import logging
from pathlib import Path
from typing import Sequence

import numpy
from pandas import DataFrame

from src.configs import DetectionConfig
from src.domain import TableEvent, TableEventKind, TableZone

logger = logging.getLogger(__name__)


class EventAnalytics:
    """Table of events and metric "empty table -> next approach"."""

    SECTION_DELAY_METRICS = "Metrics: empty table -> next approach"
    TEMPLATE_EMPTY_TO_APPROACH_DELAYS = (
        "  Intervals empty -> next approach (sec): {values}"
    )
    TEMPLATE_MEAN_DELAY = "  Mean interval (empty -> next approach): {mean:.3f} sec"
    TEMPLATE_NO_SUITABLE_INTERVALS = (
        "  Mean interval (empty -> next approach): N/A (no suitable intervals)"
    )

    def __init__(self, events: Sequence[TableEvent]) -> None:
        self._df = self._convert_events_to_dataframe(events)

    def log_summary(self) -> None:
        if self._df.empty:
            logger.info("Events not recorded (video too short or no state change).")
            return

        logger.info("")
        logger.info("--- Events (first rows) ---")
        logger.info(f"\n{self._df.head(20).to_string(index=False)}")
        logger.info("")
        logger.info(f"Total events: {len(self._df)}")

        delays, mean_delay = self._empty_to_approach_delays()
        logger.info("")

        for line in self._delay_metrics_lines(delays, mean_delay):
            logger.info(line)

    def write_report(
        self,
        report_path: Path | None,
        video_path: Path,
        table_zone: TableZone,
        config: DetectionConfig,
    ) -> None:
        if report_path is None:
            raise ValueError("Failed to write report: path is not set")

        lines: list[str] = [
            "Report: prototype detection at the table",
            f"Video: {video_path}",
            f"Table zone (x, y, width, height): ({table_zone.x}, {table_zone.y}, {table_zone.w}, {table_zone.h})",
            f"Debounce frames: {config.debounce_frames}, YOLO conf: {config.yolo_conf}, overlap ratio: {config.person_table_overlap_ratio}",
            "",
        ]

        if not self._df.empty:
            lines.append(self._df.to_csv(index=False))
            lines.append("")

        delays, mean_delay = self._empty_to_approach_delays()
        lines.extend(self._delay_metrics_lines(delays, mean_delay))

        report_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _convert_events_to_dataframe(events: Sequence[TableEvent]) -> DataFrame:
        """Convert the list of events to a pandas DataFrame."""
        if not events:
            return DataFrame(columns=["frame", "time_sec", "event"])
        return DataFrame([event.as_dict() for event in events])

    def _empty_to_approach_delays(self) -> tuple[list[float], float | None]:
        """Calculate the delays between empty table and next approach."""
        if self._df.empty:
            return [], None

        empty_times = list(
            self._df.loc[self._df["event"] == TableEventKind.EMPTY.value, "time_sec"]
        )
        approach_times = list(
            self._df.loc[self._df["event"] == TableEventKind.APPROACH.value, "time_sec"]
        )

        delays: list[float] = []
        approach_index = 0
        for empty_time in empty_times:
            while (
                approach_index < len(approach_times)
                and approach_times[approach_index] <= empty_time
            ):
                approach_index += 1

            if approach_index < len(approach_times):
                delays.append(approach_times[approach_index] - empty_time)
                approach_index += 1

        if not delays:
            return [], None

        return delays, float(numpy.mean(delays))

    def _delay_metrics_lines(
        self,
        delays: list[float],
        mean_delay: float | None,
    ) -> list[str]:
        """Same metric lines for console log and report file."""
        lines: list[str] = [self.SECTION_DELAY_METRICS]
        if delays:
            lines.append(
                self.TEMPLATE_EMPTY_TO_APPROACH_DELAYS.format(
                    values=", ".join(f"{v:.3f}" for v in delays)
                )
            )
        if mean_delay is not None:
            lines.append(self.TEMPLATE_MEAN_DELAY.format(mean=round(mean_delay, 3)))
        else:
            lines.append(self.TEMPLATE_NO_SUITABLE_INTERVALS)

        return lines
