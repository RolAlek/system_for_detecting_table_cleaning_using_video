import cv2
import numpy as np

from src.domain import TableZone


def select_table_zone_on_first_frame(
    first_frame: np.ndarray,
    window_title: str,
) -> TableZone:
    """Interactive selection of the table zone on the first frame (mouse, like in OpenCV)."""
    roi = cv2.selectROI(window_title, first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_title)
    x, y, w, h = map(int, roi[:4])

    if w <= 0 or h <= 0:
        raise SystemExit(
            "Table zone not selected: width and height must be greater than zero."
        )

    return TableZone(x, y, w, h)
