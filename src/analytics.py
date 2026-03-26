import logging
from pathlib import Path

import numpy
from pandas import DataFrame

from src.configs import DetectionConfig
from src.domain import TableEventKind, TableZone

logger = logging.getLogger(__name__)


class EventAnalytics:
    """Table of events and metric "empty table -> next approach"."""

    @staticmethod
    def empty_to_approach_delays(df: DataFrame) -> list[float]:
        if df.empty:
            return []

        empty_times = list(
            df.loc[df["event"] == TableEventKind.EMPTY.value, "time_sec"]
        )
        approach_times = list(
            df.loc[df["event"] == TableEventKind.APPROACH.value, "time_sec"]
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

        return delays

    def mean_delay_empty_to_approach(self, df: DataFrame) -> float | None:
        if not (delays := self.empty_to_approach_delays(df)):
            return None

        return float(numpy.mean(delays))

    def log_summary(self, df: DataFrame, mean_delay: float | None) -> None:
        if df.empty:
            logger.info("Events not recorded (video too short or no state change).")
            return

        logger.info("")
        logger.info("--- Events (first rows) ---")
        logger.info("\n%s", df.head(20).to_string(index=False))
        logger.info("")
        logger.info("Total events: %s", len(df))

        delays = self.empty_to_approach_delays(df)
        if delays:
            logger.info(
                "Empty table -> next approach (sec): %s",
                ", ".join(f"{v:.3f}" for v in delays),
            )
            if mean_delay is not None:
                logger.info(
                    "Average time between empty table and next approach: %.3f sec",
                    mean_delay,
                )
        else:
            logger.info("No pairs «empty -> approach» for average delay calculation.")

    @staticmethod
    def write_report(
        path: Path,
        video_path: Path,
        table_zone: TableZone,
        config: DetectionConfig,
        df: DataFrame,
        mean_delay: float | None,
    ) -> None:
        lines: list[str] = [
            "Report: prototype detection at the table",
            f"Video: {video_path}",
            f"Table zone (x, y, width, height): ({table_zone.x}, {table_zone.y}, {table_zone.w}, {table_zone.h})",
            f"Debounce frames: {config.debounce_frames}, YOLO conf: {config.yolo_conf}, overlap ratio: {config.person_table_overlap_ratio}",
            "",
        ]

        if not df.empty:
            lines.append(df.to_csv(index=False))
            lines.append("")

        delays = EventAnalytics.empty_to_approach_delays(df)
        if delays:
            lines.append(
                f"Empty table -> next approach (sec): {', '.join(f'{v:.3f}' for v in delays)}"
            )
            lines.append("")

        if mean_delay is not None:
            lines.append(
                f"Average time between empty table and next approach: {mean_delay:.3f} sec"
            )

        else:
            lines.append("Average time: N/A (no suitable intervals)")

        path.write_text("\n".join(lines), encoding="utf-8")
