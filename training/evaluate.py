"""Evaluate a YOLO11 checkpoint on a labeled validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_device_string, load_config
from src.logging_config import setup_logging
from src.storage.repository import WildlifeRepository
from training.train import validate_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 wildlife model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "training" / "dataset.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logging(cfg)
    dataset_yaml = Path(args.data)
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    validate_dataset_yaml(dataset_yaml)

    from ultralytics import YOLO

    device = args.device or get_device_string(cfg.model.device)
    model = YOLO(str(weights))
    logger.info("Running validation on %s", dataset_yaml)
    metrics_obj = model.val(data=str(dataset_yaml), imgsz=args.imgsz, device=device)

    metrics = {
        "precision": float(getattr(metrics_obj.box, "mp", 0) if hasattr(metrics_obj, "box") else 0),
        "recall": float(getattr(metrics_obj.box, "mr", 0) if hasattr(metrics_obj, "box") else 0),
        "map50": float(getattr(metrics_obj.box, "map50", 0) if hasattr(metrics_obj, "box") else 0),
        "map50_95": float(getattr(metrics_obj.box, "map", 0) if hasattr(metrics_obj, "box") else 0),
    }
    if hasattr(metrics_obj, "results_dict"):
        metrics.update({k: float(v) for k, v in metrics_obj.results_dict.items() if isinstance(v, (int, float))})

    repo = WildlifeRepository(cfg.paths.database_path)
    run_id = repo.register_model_run("evaluation", str(weights), metrics, notes=f"data={dataset_yaml}")
    logger.info("Evaluation complete. Run ID: %s", run_id)
    print(json.dumps({"run_id": run_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
