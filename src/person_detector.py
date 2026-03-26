from typing import List

import numpy
from ultralytics import YOLO

from src.configs import DetectionConfig
from src.domain import BoundingBox, TableZone


class PersonDetector:
    """Wrapper over YOLO: people on the frame and flag "person in the table zone"."""

    def __init__(self, config: DetectionConfig, weights: str = "yolov8n.pt") -> None:
        self._cfg = config
        self._model = YOLO(weights)

    def detect_people_in_table_zone(
        self,
        frame_bgr: numpy.ndarray,
        table_zone: TableZone,
    ) -> tuple[bool, list[BoundingBox]]:
        in_zone = False
        boxes: List[BoundingBox] = []

        results = self._model.predict(
            frame_bgr,
            conf=self._cfg.yolo_conf,
            classes=[self._cfg.person_class_id],
            verbose=False,
        )
        if not results:
            return in_zone, boxes

        detected_boxes = results[0].boxes
        if not detected_boxes:
            return in_zone, boxes

        person_boxes_corner_coords = detected_boxes.xyxy.cpu().numpy()
        ratio = self._cfg.person_table_overlap_ratio

        for corner_coords in person_boxes_corner_coords:
            x1, y1, x2, y2 = map(float, corner_coords[:4].tolist())
            box = BoundingBox(x1, y1, x2, y2)
            boxes.append(box)

            if box.overlaps_table_zone(table_zone, ratio):
                in_zone = True

        return in_zone, boxes
