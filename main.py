from dataclasses import dataclass
from enum import StrEnum


class TableEventKind(StrEnum):
    """Тип события для журнала и отчёта (значение = строка в CSV)."""

    EMPTY = "empty"
    OCCUPIED = "occupied"
    APPROACH = "approach"


@dataclass(frozen=True)
class BoundingBox:
    """
    Прямоугольная рамка на кадре видео: левый верхний и правый нижний углы (в пикселях).

    Такой формат выдаёт детектор людей: по сути «рамка вокруг человека» на изображении.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def intersection_area(self, other: "BoundingBox") -> float:
        """Границы общей части двух рамок: внутренний прямоугольник перекрытия"""
        overlap_left = max(self.x1, other.x1)
        overlap_top = max(self.y1, other.y1)
        overlap_right = min(self.x2, other.x2)
        overlap_bottom = min(self.y2, other.y2)

        return max(0.0, overlap_right - overlap_left) * max(
            0.0, overlap_bottom - overlap_top
        )

    def overlaps_table_zone(
        self,
        table_zone: "TableZone",
        min_overlap_ratio: float,
    ) -> bool:
        """
        True, если достаточная часть рамки (человека) попадает в выделенную зону столика.

        Сравнение по доле площади рамки человека, перекрывающейся с зоной стола.
        """
        if self.area <= 1e-6:
            return False
        inter = self.intersection_area(table_zone.to_bounding_box())
        return (inter / self.area) >= min_overlap_ratio


@dataclass(frozen=True)
class TableZone:
    """
    Зона одного столика на видео: где он находится и какого размера.

    Задаётся вручную при запуске (выделение мышью), в тех же координатах, что и кадр.
    Поля: левый верхний угол (x, y), ширина и высота прямоугольника.
    """

    x: int
    y: int
    w: int
    h: int

    def to_bounding_box(self) -> BoundingBox:
        """Та же область в формате «две противоположные угловые точки» — для сравнения с рамками людей."""
        return BoundingBox(
            float(self.x),
            float(self.y),
            float(self.x + self.w),
            float(self.y + self.h),
        )


@dataclass(frozen=True)
class DetectionConfig:
    """Параметры детекции и антидребезга (иммутабельная конфигурация)."""

    debounce_frames: int = 10
    person_table_overlap_ratio: float = 0.15
    yolo_conf: float = 0.35
    person_class_id: int = 0
