"""Draw detections and export annotated media."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.postprocessing import Detection

# BGR colors for common labels
_COLORS = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (219, 199, 49),
    (239, 142, 49),
]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return _COLORS[class_id % len(_COLORS)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        color = _color_for_class(d.class_id)
        x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.display_name} {d.confidence:.2f}"
        if d.track_id is not None:
            label += f" ID:{d.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
