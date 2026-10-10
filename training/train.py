"""Fine-tune YOLO11 on a custom wildlife dataset (Ultralytics format)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.logging_config import setup_logging
from src.storage.repository import WildlifeRepository


def validate_dataset_yaml(dataset_yaml: Path) -> None:
    import yaml

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    with open(dataset_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base = dataset_yaml.parent
    path = data.get("path")
    if not path:
        raise ValueError("dataset.yaml must define 'path'.")

    root = (base / path).resolve() if not Path(path).is_absolute() else Path(path)
    for split_key in ("train", "val"):
        split = data.get(split_key)
        if not split:
            raise ValueError(f"dataset.yaml missing split: {split_key}")
        split_path = root / split if not Path(split).is_absolute() else Path(split)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_path}. Prepare YOLO annotations before training."
            )

    names = data.get("names")
    if not names:
        raise ValueError("dataset.yaml must define class 'names'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11 wildlife detector")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "training" / "dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None, help="auto|cpu|cuda|0")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logging(cfg)
    dataset_yaml = Path(args.data)

    validate_dataset_yaml(dataset_yaml)
    logger.info("Dataset validation passed: %s", dataset_yaml)

    from ultralytics import YOLO

    from src.config import get_device_string

    device = args.device or get_device_string(cfg.model.device)
    epochs = args.epochs or cfg.training.epochs
    batch = args.batch or cfg.training.batch_size
    base = cfg.training.default_base_weights

    model = YOLO(base)
    project = PROJECT_ROOT / cfg.training.project_dir
    project.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training: base=%s epochs=%s batch=%s", base, epochs, batch)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=device,
        project=str(project),
        name=cfg.training.name,
        workers=cfg.training.workers,
    )

    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = dict(results.results_dict)

    repo = WildlifeRepository(cfg.paths.database_path)
    run_id = repo.register_model_run(
        "training",
        weights_path=str(project / cfg.training.name / "weights" / "best.pt"),
        metrics=metrics,
        notes=f"data={dataset_yaml}",
    )
    logger.info("Training finished. Run ID: %s", run_id)
    print(json.dumps({"run_id": run_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
