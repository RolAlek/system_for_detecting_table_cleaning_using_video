from pathlib import Path

import numpy
from pandas import DataFrame

from src.configs import DetectionConfig
from src.domain import TableEventKind, TableZone


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

    def print_summary(self, df: DataFrame, mean_delay: float | None) -> None:
        if df.empty:
            print(
                "Событий не зафиксировано (видео слишком короткое или нет смены состояния)."
            )
            return

        print("\n--- События (первые строки) ---")
        print(df.head(20).to_string(index=False))
        print(f"\nВсего событий: {len(df)}")

        delays = self.empty_to_approach_delays(df)
        if delays:
            print(f"Интервалы пустой стол -> следующий подход (сек): {delays}")
            print(
                "Среднее время между уходом (пустой стол) и подходом следующего: "
                f"{mean_delay:.3f} с"
            )
        else:
            print("Нет пар «empty -> approach» для расчёта средней задержки.")

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
            "Отчёт: прототип детекции у столика",
            f"Видео: {video_path}",
            f"Зона столика (x, y, ширина, высота): ({table_zone.x}, {table_zone.y}, {table_zone.w}, {table_zone.h})",
            f"Дебаунс кадров: {config.debounce_frames}, YOLO conf: {config.yolo_conf}, overlap ratio: {config.person_table_overlap_ratio}",
            "",
        ]

        if not df.empty:
            lines.append(df.to_csv(index=False))
            lines.append("")

        if mean_delay is not None:
            lines.append(
                f"Среднее время пустой стол -> следующий подход: {mean_delay:.3f} сек"
            )

        else:
            lines.append("Среднее время: н/д (нет подходящих интервалов)")

        path.write_text("\n".join(lines), encoding="utf-8")
