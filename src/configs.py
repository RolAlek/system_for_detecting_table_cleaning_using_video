from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DetectionConfig:
    """Detection and anti-bounce parameters."""

    debounce_frames: int = 10
    person_table_overlap_ratio: float = 0.15
    yolo_conf: float = 0.35
    person_class_id: int = 0

    @property
    def output_video_dir(self) -> Path:
        return BASE_DIR / "output_videos"

    @property
    def report_dir(self) -> Path:
        return BASE_DIR / "reports"

    @property
    def problem_frame_dir(self) -> Path:
        return BASE_DIR / "problem_frames"
