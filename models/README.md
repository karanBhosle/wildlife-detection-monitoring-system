# Model weights

Place custom-trained wildlife checkpoints here (`.pt` files). Git ignores `*.pt` to avoid committing large binaries.

## Default inference

`configs/config.yaml` defaults to `yolo11n.pt`, which Ultralytics downloads on first run. That checkpoint is trained on **COCO** and recognizes general animal categories, not most reserve-specific species (tiger, leopard, wild boar, etc.).

## Custom wildlife model

1. Prepare a YOLO-format dataset (see `training/dataset.yaml`).
2. Run `python training/train.py --data training/dataset.yaml`.
3. Point `model.weights_path` in `configs/config.yaml` to `models/runs/wildlife_train/weights/best.pt` (or copy `best.pt` here).

Document dataset source, class list, and license for any checkpoint you use in production.
