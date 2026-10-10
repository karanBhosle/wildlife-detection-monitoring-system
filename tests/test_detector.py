from unittest.mock import MagicMock, patch

import numpy as np

from src.config import load_config
from src.detection.detector import WildlifeDetector
from src.detection.postprocessing import Detection, filter_detections


def test_filter_detections_confidence_and_class():
    dets = [
        Detection(0, "bear", "bear", 0.9, 0, 0, 10, 10),
        Detection(1, "person", "person", 0.4, 0, 0, 10, 10),
    ]
    out = filter_detections(dets, min_confidence=0.5, allowed_class_names={"bear"})
    assert len(out) == 1
    assert out[0].class_name == "bear"


def test_detector_parse_and_filter(mock_yolo_results):
    cfg = load_config()
    cfg.model.supported_classes = ["bear"]
    detector = WildlifeDetector(cfg)

    with patch("ultralytics.YOLO") as yolo_cls:
        model = MagicMock()
        model.names = {0: "bear", 1: "person"}
        model.predict.return_value = mock_yolo_results
        yolo_cls.return_value = model
        detector.load()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets, stats = detector.detect_image(frame, confidence=0.25)
    assert len(dets) == 1
    assert dets[0].class_name == "bear"
    assert stats.num_detections == 1


def test_empty_detections():
    cfg = load_config()
    detector = WildlifeDetector(cfg)

    class EmptyBoxes:
        def __len__(self):
            return 0

    class Result:
        boxes = EmptyBoxes()

    with patch("ultralytics.YOLO") as yolo_cls:
        model = MagicMock()
        model.names = {0: "bear"}
        model.predict.return_value = [Result()]
        yolo_cls.return_value = model
        detector.load()

    dets, stats = detector.detect_image(np.zeros((64, 64, 3), dtype=np.uint8))
    assert dets == []
    assert stats.num_detections == 0


def test_unsupported_class_warning():
    cfg = load_config()
    cfg.model.supported_classes = ["tiger"]
    detector = WildlifeDetector(cfg)
    with patch("ultralytics.YOLO") as yolo_cls:
        model = MagicMock()
        model.names = {0: "bear"}
        yolo_cls.return_value = model
        detector.load()
    assert any("tiger" in w for w in detector.model_info.unsupported_filter_warnings)
