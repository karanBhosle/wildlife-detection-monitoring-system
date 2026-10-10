import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_bgr_image(tmp_path: Path) -> Path:
    import cv2

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[30:90, 40:120] = (0, 128, 255)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def mock_yolo_results():
    class Boxes:
        def __init__(self):
            import torch

            self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
            self.conf = torch.tensor([0.9])
            self.cls = torch.tensor([0])
            self.id = None

        def __len__(self):
            return 1

    class Result:
        boxes = Boxes()

    return [Result()]
