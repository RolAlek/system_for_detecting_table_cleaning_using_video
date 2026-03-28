from enum import StrEnum

import cv2
import numpy

from src.domain import BoundingBox, TableZone

BGR_RED: tuple[int, int, int] = (0, 0, 255)
BGR_GREEN: tuple[int, int, int] = (0, 255, 0)
BGR_BLUE: tuple[int, int, int] = (255, 128, 0)

FONT_SCALE: float = 0.6
TEXT_THICKNESS: int = 2
BOX_THICKNESS: int = 1
TEXT_OFFSET: int = 8


class Label(StrEnum):
    EMPTY = "table: EMPTY"
    OCCUPIED = "table: OCCUPIED"


def _close_select_roi_window(window_title: str) -> None:
    """Close the select ROI window."""
    try:
        cv2.destroyWindow(window_title)
    except cv2.error:
        pass

    for _ in range(20):
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def select_table_zone_on_first_frame(
    first_frame: numpy.ndarray,
    window_title: str,
) -> TableZone:
    """Interactive selection of the table zone on the first frame (mouse, like in OpenCV)."""
    roi = cv2.selectROI(window_title, first_frame, showCrosshair=True, fromCenter=False)
    _close_select_roi_window(window_title)

    return TableZone(*roi[:4])


def draw_table_status(
    frame_bgr: numpy.ndarray,
    table_zone: TableZone,
    logical_occupied: bool | None,
) -> None:
    """Draw the table status on the frame."""
    color = BGR_RED if logical_occupied is True else BGR_GREEN

    cv2.rectangle(
        frame_bgr,
        (table_zone.x, table_zone.y),
        (table_zone.x + table_zone.w, table_zone.y + table_zone.h),
        color,
        2,
    )

    cv2.putText(
        img=frame_bgr,
        text=Label.EMPTY if not logical_occupied else Label.OCCUPIED,
        org=(table_zone.x, max(0, table_zone.y - TEXT_OFFSET)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=FONT_SCALE,
        color=color,
        thickness=TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def draw_person_boxes(
    frame_bgr: numpy.ndarray,
    person_boxes: list[BoundingBox],
) -> None:
    """Draw the person boxes on the frame."""
    for box in person_boxes:
        cv2.rectangle(
            img=frame_bgr,
            pt1=(int(box.x1), int(box.y1)),
            pt2=(int(box.x2), int(box.y2)),
            color=BGR_BLUE,
            thickness=BOX_THICKNESS,
        )
