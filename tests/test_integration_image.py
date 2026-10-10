"""Integration test with mocked YOLO — no weight download."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.config import load_config
from src.detection.detector import WildlifeDetector
from src.processing.image_processor import ImageProcessor
from src.storage.repository import WildlifeRepository


def test_image_processor_end_to_end(tmp_path, sample_bgr_image, mock_yolo_results):
    cfg = load_config()
    cfg.paths.database_path = str(tmp_path / "wildlife.db")
    cfg.paths.annotated_images_dir = str(tmp_path / "annotated")
    cfg.ensure_directories()

    db = WildlifeRepository(cfg.paths.database_path)
    detector = WildlifeDetector(cfg)
    with patch("ultralytics.YOLO") as yolo_cls:
        model = MagicMock()
        model.names = {0: "bear"}
        model.predict.return_value = mock_yolo_results
        yolo_cls.return_value = model
        detector.load()

    proc = ImageProcessor(cfg, detector, db)
    result = proc.process(sample_bgr_image, confidence=0.25)
    assert result.error is None
    assert len(result.detections) == 1
    assert result.annotated_path is not None
    df = db.fetch_detections_df()
    assert len(df) == 1
