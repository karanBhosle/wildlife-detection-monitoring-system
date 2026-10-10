"""Frame and image preprocessing for YOLO inference."""

from __future__ import annotations

import numpy as np


def ensure_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if len(frame.shape) == 2:
        return np.stack([frame] * 3, axis=-1)
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    return frame
