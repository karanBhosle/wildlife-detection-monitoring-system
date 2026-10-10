import pytest

from src.config import WildlifeConfig, load_config, get_device_string


def test_load_config():
    cfg = load_config()
    assert cfg.model.confidence_threshold > 0
    assert cfg.paths.output_dir.endswith("output") or "output" in cfg.paths.output_dir


def test_invalid_confidence():
    with pytest.raises(Exception):
        WildlifeConfig.model_validate({"model": {"confidence_threshold": 1.5}})


def test_get_device_string_cpu():
    assert get_device_string("cpu") == "cpu"
