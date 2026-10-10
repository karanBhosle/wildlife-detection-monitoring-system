"""Filter and normalize detection outputs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Detection:
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None
    frame_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def filter_detections(
    detections: list[Detection],
    min_confidence: float,
    allowed_class_names: set[str] | None,
) -> list[Detection]:
    out: list[Detection] = []
    for d in detections:
        if d.confidence < min_confidence:
            continue
        if allowed_class_names is not None and d.class_name not in allowed_class_names:
            continue
        out.append(d)
    return out
