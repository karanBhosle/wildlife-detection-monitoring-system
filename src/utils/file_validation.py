"""Input file validation utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""
    path: Path | None = None


def sanitize_filename(name: str, max_len: int = 200) -> str:
    base = Path(name).name
    safe = re.sub(r"[^\w.\-]", "_", base)
    return safe[:max_len] or "unnamed"


def validate_extension(path: Path, allowed: list[str]) -> ValidationResult:
    ext = path.suffix.lower()
    normalized = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed]
    if ext not in normalized:
        return ValidationResult(
            ok=False,
            message=f"Extension '{ext}' not allowed. Allowed: {', '.join(normalized)}",
            path=path,
        )
    return ValidationResult(ok=True, path=path)


def validate_file_size(path: Path, max_mb: int) -> ValidationResult:
    if not path.exists():
        return ValidationResult(ok=False, message="File does not exist.", path=path)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return ValidationResult(
            ok=False,
            message=f"File size {size_mb:.1f} MB exceeds limit {max_mb} MB.",
            path=path,
        )
    return ValidationResult(ok=True, path=path)


def validate_image_readable(path: Path) -> ValidationResult:
    try:
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            return ValidationResult(ok=False, message="OpenCV could not decode image.", path=path)
        return ValidationResult(ok=True, path=path)
    except Exception as exc:
        return ValidationResult(ok=False, message=str(exc), path=path)


def validate_video_readable(path: Path, min_frames: int = 1) -> ValidationResult:
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return ValidationResult(ok=False, message="Could not open video.", path=path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None and frame.size > 0:
                count += 1
            if count >= min_frames:
                break
        if count < min_frames:
            return ValidationResult(ok=False, message="Video has no valid frames.", path=path)
        return ValidationResult(ok=True, path=path)
    except Exception as exc:
        return ValidationResult(ok=False, message=str(exc), path=path)
    finally:
        if cap is not None:
            cap.release()


def file_content_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def extract_observation_timestamp(path: Path) -> str | None:
    """Try EXIF DateTimeOriginal, else parse common filename patterns."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if exif:
                for tag_id, value in exif.items():
                    from PIL.ExifTags import TAGS

                    if TAGS.get(tag_id) == "DateTimeOriginal" and value:
                        # EXIF format: YYYY:MM:DD HH:MM:SS
                        return str(value).replace(":", "-", 2)
    except Exception:
        pass

    name = path.stem
    # YYYY-MM-DD or YYYY_MM_DD with optional time
    m = re.search(
        r"(\d{4})[-_](\d{2})[-_](\d{2})(?:[-_T](\d{2})[-_]?(\d{2})[-_]?(\d{2}))?",
        name,
    )
    if m:
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if m.group(4):
            return f"{date} {m.group(4)}:{m.group(5) or '00'}:{m.group(6) or '00'}"
        return date
    return None
