import numpy
from ultralytics import YOLO

from src.configs import DetectionConfig
from src.domain import BoundingBox, TableZone


class PersonDetector:
    """Wrapper over YOLO: people on the frame and flag "person in the table zone"."""

    def __init__(
        self,
        config: DetectionConfig,
        table_zone: TableZone,
        weights: str = "yolov8n.pt",
    ) -> None:
        self._cfg = config
        self._table_zone = table_zone
        self._model = YOLO(weights)

    def detect_people_in_table_zone(
        self,
        frame_bgr: numpy.ndarray,
    ) -> tuple[bool, list[BoundingBox]]:
        """Detect people in the table zone."""

        results = self._model.predict(
            frame_bgr,
            conf=self._cfg.yolo_conf,
            classes=[self._cfg.person_class_id],
            verbose=False,
        )
        if not results or not results[0].boxes:
            return False, []

        person_boxes_corner_coords = results[0].boxes.xyxy.cpu().numpy()
        in_zone = False
        boxes: list[BoundingBox] = []

        for corner_coords in person_boxes_corner_coords:
            box = BoundingBox(*corner_coords[:4].tolist())
            boxes.append(box)

            if box.overlaps_table_zone(
                table_zone=self._table_zone,
                min_overlap_ratio=self._cfg.person_table_overlap_ratio,
            ):
                in_zone = True

        return in_zone, boxes
