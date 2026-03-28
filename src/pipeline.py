import argparse
import logging
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self, Type

import cv2
import numpy
from tqdm import tqdm

from src.analytics import EventAnalytics
from src.configs import DetectionConfig
from src.debounced_table_fsm import DebouncedTableStateMachine
from src.domain import TableEvent, TableZone
from src.person_detector import PersonDetector
from src.visualization import (
    draw_person_boxes,
    draw_table_status,
    select_table_zone_on_first_frame,
)

logger = logging.getLogger(__name__)


@contextmanager
def frames_pb(
    n_frames: int,
    no_progress: bool,
) -> Iterator[tqdm]:
    total = n_frames if n_frames > 0 else None
    show_bar = not no_progress and sys.stderr.isatty()
    with tqdm(
        total=total,
        unit="fr",
        desc="Frames",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.25,
        disable=not show_bar,
    ) as pbar:
        yield pbar


class TableVideoPipeline:
    """Pipeline: video → detection → FSM → visualization → report."""

    def __init__(
        self,
        args: argparse.Namespace,
        config: DetectionConfig = DetectionConfig(),
    ) -> None:
        self._args = args
        self._config = config

        # Video capture
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 25.0
        self._width: int = 0
        self._height: int = 0
        self._n_frames: int = 0

        self._fourcc: int | None = None
        self._writer: cv2.VideoWriter | None = None

        # Artifacts paths
        self._output_video_path: Path | None = None
        self._report_path: Path | None = None
        self._problem_frame_path: Path | None = None

    def __enter__(self) -> Self:
        video_path = Path(self._args.video)
        self._open_video(video_path)

        if self._cap is None or not self._cap.isOpened():
            raise SystemExit(f"Failed to open video: {video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._fourcc = cv2.VideoWriter.fourcc(*"mp4v")

        self._output_video_path = self._resolve_artifact_path(
            self._args.output, self._config.output_video_dir
        )
        self._report_path = self._resolve_artifact_path(
            self._args.report, self._config.report_dir
        )
        self._problem_frame_path = self._resolve_artifact_path(
            self._args.problem_frame, self._config.problem_frame_dir
        )

        for artifact in (
            self._output_video_path,
            self._report_path,
            self._problem_frame_path,
        ):
            artifact.parent.mkdir(parents=True, exist_ok=True)

        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()

        if self._cap is not None:
            self._cap.release()

        cv2.destroyAllWindows()

        if exc_type is not None:
            for path in (
                self._output_video_path,
                self._report_path,
                self._problem_frame_path,
            ):
                if path is None:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def run(self) -> None:
        """Run the pipeline: video → detection → ROI → FSM → visualization → report."""
        logger.info("Running the pipeline...")

        table_zone = self._select_table_zone()

        self._create_writer()
        if self._writer is None:
            raise RuntimeError("Failed to create writer")

        logger.info("Loading YOLOv8n (will be downloaded on the first run)...")
        fsm = DebouncedTableStateMachine(self._config.debounce_frames)

        self._run_frame_loop(
            table_zone=table_zone,
            detector=PersonDetector(self._config, table_zone),
            fsm=fsm,
        )
        self._emit_report_and_paths(table_zone, fsm.events)

    def _open_video(self, video_path: Path) -> None:
        """Open the video file."""
        logger.info(f"Opening video from {video_path} path...")

        if not video_path.is_file():
            raise SystemExit(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(video_path))
        logger.info("Video file successfully opened")

    @staticmethod
    def _resolve_artifact_path(cli_value: str, default_dir: Path) -> Path:
        if (path := Path(cli_value).expanduser()).is_absolute():
            return path.resolve()

        return (default_dir / cli_value).resolve()

    def _create_writer(self) -> None:
        """Create the video writer."""
        logger.info("Creating writer...")

        try:
            if self._fourcc is None:
                raise ValueError("FourCC is not set")

            self._writer = cv2.VideoWriter(
                filename=str(self._output_video_path),
                fourcc=self._fourcc,
                fps=float(self._fps),
                frameSize=(self._width, self._height),
            )

            if not self._writer.isOpened():
                raise ValueError(
                    f"Failed to create output video (codec/path?): "
                    f"{self._output_video_path}"
                )

        except Exception as exc:
            logger.error("Failed to create writer", exc_info=exc)
            raise

        logger.info("Writer successfully created")

    def _select_table_zone(self) -> TableZone:
        """Select the table zone on the first frame."""
        logger.info("Selecting table zone...")

        if self._cap is None:
            raise RuntimeError("Video is not opened")

        ok, first = self._cap.read()
        if not ok or first is None:
            raise ValueError("Failed to read first frame")

        logger.info(
            "Select the table with the mouse in the window and press Enter or Space."
        )
        table_zone = select_table_zone_on_first_frame(first, "Select the table zone")
        logger.info("Table zone selected", extra={"table_zone": table_zone})

        return table_zone

    def _run_frame_loop(
        self,
        table_zone: TableZone,
        detector: PersonDetector,
        fsm: DebouncedTableStateMachine,
    ) -> tuple[numpy.ndarray | None, bool]:
        """Run the frame loop."""
        if self._cap is None or self._writer is None:
            raise RuntimeError("Video capture or writer is not initialized")

        last_vis_frame: numpy.ndarray | None = None
        saved_problem = False
        frame_idx = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        problem_frame_idx = int(
            (self._n_frames or 1) * max(0.0, min(1.0, self._args.problem_at))
        )

        with frames_pb(
            n_frames=self._n_frames,
            no_progress=self._args.no_progress,
        ) as pbar:
            while True:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    break

                raw_in_zone, person_boxes = detector.detect_people_in_table_zone(frame)
                fsm.update(frame_idx, self._fps, raw_in_zone)

                draw_table_status(frame, table_zone, fsm.logical_occupied)
                draw_person_boxes(frame, person_boxes)

                self._writer.write(frame)
                last_vis_frame = frame

                if not saved_problem and frame_idx >= int(problem_frame_idx):
                    self._save_problem_frame(frame)
                    saved_problem = True

                frame_idx += 1
                pbar.update()

        return last_vis_frame, saved_problem

    def _save_problem_frame(
        self,
        last_frame: numpy.ndarray | None,
    ) -> None:
        """Save the problem frame."""
        if last_frame is None:
            return

        if self._problem_frame_path is None:
            raise ValueError("Failed to save problem frame: path is not set")

        cv2.imwrite(str(self._problem_frame_path), last_frame)

    def _emit_report_and_paths(
        self,
        table_zone: TableZone,
        events: Sequence[TableEvent],
    ) -> None:
        """Emit the report and paths."""
        analytics = EventAnalytics(events)
        analytics.log_summary()
        analytics.write_report(
            report_path=self._report_path,
            video_path=Path(self._args.video),
            table_zone=table_zone,
            config=self._config,
        )
        logger.info("")
        logger.info(f"Video saved: {self._output_video_path}")
        logger.info(f"Report: {self._report_path}")
