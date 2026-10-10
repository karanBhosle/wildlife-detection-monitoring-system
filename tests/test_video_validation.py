from pathlib import Path

import cv2
import numpy as np

from src.utils.file_validation import validate_video_readable


def test_invalid_video_empty(tmp_path: Path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"not a video")
    assert not validate_video_readable(p).ok


def test_valid_short_video(tmp_path: Path):
    p = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(p),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()
    assert validate_video_readable(p).ok
