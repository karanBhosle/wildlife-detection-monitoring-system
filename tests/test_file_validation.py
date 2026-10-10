from pathlib import Path

import numpy as np

from src.utils.file_validation import (
    extract_observation_timestamp,
    sanitize_filename,
    validate_extension,
    validate_image_readable,
)


def test_sanitize_filename():
    assert sanitize_filename("../../evil/name?.jpg") == "name_.jpg"


def test_validate_extension_ok(tmp_path: Path):
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    r = validate_extension(p, [".png", ".jpg"])
    assert r.ok


def test_validate_extension_fail(tmp_path: Path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"x")
    r = validate_extension(p, [".png"])
    assert not r.ok


def test_validate_image_readable(sample_bgr_image: Path):
    assert validate_image_readable(sample_bgr_image).ok


def test_extract_timestamp_from_filename(tmp_path: Path):
    p = tmp_path / "cam_2024-03-15_143022.jpg"
    p.write_bytes(b"x")
    ts = extract_observation_timestamp(p)
    assert ts is not None
    assert "2024-03-15" in ts


def test_corrupt_image_fails(tmp_path: Path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not-an-image")
    r = validate_image_readable(p)
    assert not r.ok
