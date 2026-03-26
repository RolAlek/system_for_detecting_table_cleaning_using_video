from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Detection and anti-bounce parameters."""

    debounce_frames: int = 10
    person_table_overlap_ratio: float = 0.15
    yolo_conf: float = 0.35
    person_class_id: int = 0
