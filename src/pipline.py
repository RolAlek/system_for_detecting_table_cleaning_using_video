import argparse
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self, Type

import cv2
import numpy
from cv2 import VideoCapture, VideoWriter, destroyAllWindows, imwrite
from pandas import DataFrame

from src.analitics import EventAnalytics
from src.configs import DetectionConfig
from src.debounced_table_fsm import DebouncedTableStateMachine
from src.domain import TableEvent
from src.person_detector import PersonDetector
from src.visualization import (
    draw_person_boxes,
    draw_table_status,
    select_table_zone_on_first_frame,
)


class TableVideoPipeline:
    """Сборка: видео → детекция → FSM → визуализация → отчёт."""

    def __init__(
        self,
        args: argparse.Namespace,
        config: DetectionConfig = DetectionConfig(),
    ) -> None:
        self._args = args
        self._config = config
        self._analytics = EventAnalytics()

        self._cap: VideoCapture | None = None
        self._fourcc: int | None = None
        self._writer: VideoWriter | None = None
        self._fps: float = 25.0
        self._width: int = 0
        self._height: int = 0
        self._n_frames: int = 0

    def __enter__(self) -> Self:
        video_path = Path(self._args.video)
        self._open_video(video_path)

        if self._cap is None or not self._cap.isOpened():
            raise SystemExit(f"Не удалось открыть видео: {video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        self._fourcc = VideoWriter.fourcc(*"mp4v")

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

        destroyAllWindows()

    def run(self) -> None:
        if self._cap is None:
            raise RuntimeError("Video is not opened")

        ok, first = self._cap.read()
        if not ok or first is None:
            self._cap.release()
            raise SystemExit("Failed to read first frame")

        print("Выделите стол мышью в окне и нажмите Enter или Space.")
        table_zone = select_table_zone_on_first_frame(first, "Выделите зону столика")

        self._create_writer(Path(self._args.output))
        if self._writer is None:
            raise RuntimeError("Failed to create writer")

        print("Загрузка YOLOv8n (при первом запуске скачается вес)...")
        detector = PersonDetector(self._config)
        fsm = DebouncedTableStateMachine(self._config.debounce_frames)

        problem_frame_idx = int(
            (self._n_frames or 1) * max(0.0, min(1.0, self._args.problem_at))
        )
        saved_problem = False
        last_vis_frame: numpy.ndarray | None = None
        frame_idx = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break

            raw_in_zone, person_boxes = detector.detect_people_in_table_zone(
                frame, table_zone
            )
            fsm.update(frame_idx, self._fps, raw_in_zone)

            draw_table_status(frame, table_zone, fsm.logical_occupied)
            draw_person_boxes(frame, person_boxes)

            self._writer.write(frame)
            last_vis_frame = frame

            if not saved_problem and frame_idx >= problem_frame_idx:
                cv2.imwrite(str(Path(self._args.problem_frame)), frame)
                saved_problem = True

            frame_idx += 1

        if not saved_problem and last_vis_frame is not None:
            imwrite(filename=str(Path(self._args.problem_frame)), img=last_vis_frame)

        dataframe = self._convert_events_to_dataframe(fsm.events)
        mean_delay = self._analytics.mean_delay_empty_to_approach(dataframe)
        self._analytics.print_summary(dataframe, mean_delay)
        self._analytics.write_report(
            Path(self._args.report),
            Path(self._args.video),
            table_zone,
            self._config,
            dataframe,
            mean_delay,
        )
        print(f"\nВидео сохранено: {Path(self._args.output).resolve()}")
        print(f"Отчёт: {Path(self._args.report).resolve()}")

    def _open_video(self, video_path: Path) -> None:
        if not video_path.is_file():
            raise SystemExit(f"Файл видео не найден: {video_path}")

        self._cap = VideoCapture(str(video_path))

    def _create_writer(self, out_path: Path) -> None:
        if self._fourcc is None:
            raise ValueError("FourCC is not set")

        self._writer = VideoWriter(
            filename=str(out_path),
            fourcc=self._fourcc,
            fps=self._fps,
            frameSize=(self._width, self._height),
        )

        if not self._writer.isOpened():
            raise ValueError(f"Failed to create output video: {out_path}")

    @staticmethod
    def _convert_events_to_dataframe(events: Sequence[TableEvent]) -> DataFrame:
        if not events:
            return DataFrame(columns=["frame", "time_sec", "event"])
        return DataFrame([event.as_dict() for event in events])
