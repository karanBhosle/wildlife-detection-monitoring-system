"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ModelConfig(BaseModel):
    weights_path: str = "yolo11n.pt"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32, le=2048)
    supported_classes: list[str] = Field(default_factory=list)
    class_aliases: dict[str, str] = Field(default_factory=dict)


class TrackingConfig(BaseModel):
    tracker: str = "bytetrack.yaml"
    persist: bool = True
    frame_sample_interval: int = Field(default=1, ge=1)


class VideoConfig(BaseModel):
    max_upload_mb: int = Field(default=500, ge=1)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi"])


class ImageConfig(BaseModel):
    max_upload_mb: int = Field(default=50, ge=1)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])


class BatchConfig(BaseModel):
    max_files: int = Field(default=500, ge=1)
    input_dir: str = "data/input"
    dedupe_by_content_hash: bool = True


class PathsConfig(BaseModel):
    output_dir: str = "data/output"
    annotated_images_dir: str = "data/output/annotated_images"
    annotated_videos_dir: str = "data/output/annotated_videos"
    exports_dir: str = "data/output/exports"
    database_path: str = "data/output/wildlife.db"

    def resolve(self, root: Path) -> PathsConfig:
        """Return paths resolved relative to project root."""
        data = self.model_dump()

        def _r(p: str) -> str:
            path = Path(p)
            if path.is_absolute():
                return str(path)
            return str((root / path).resolve())

        return PathsConfig(
            output_dir=_r(data["output_dir"]),
            annotated_images_dir=_r(data["annotated_images_dir"]),
            annotated_videos_dir=_r(data["annotated_videos_dir"]),
            exports_dir=_r(data["exports_dir"]),
            database_path=_r(data["database_path"]),
        )


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/wildlife.log"
    max_bytes: int = 10485760
    backup_count: int = 3


class AppConfig(BaseModel):
    brand_name: str = "karanverse"
    brand_tagline: str = "Wildlife Detection & Monitoring System"
    title: str = "karanverse Wildlife Detection & Monitoring System"
    page_icon: str = "🌌"


class TrainingConfig(BaseModel):
    default_base_weights: str = "yolo11n.pt"
    project_dir: str = "models/runs"
    name: str = "wildlife_train"
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=16, ge=1)
    workers: int = Field(default=4, ge=0)


class WildlifeConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    @field_validator("paths", mode="before")
    @classmethod
    def _paths_default(cls, v: Any) -> Any:
        return v or {}

    @model_validator(mode="after")
    def resolve_paths(self) -> WildlifeConfig:
        self.paths = self.paths.resolve(self.project_root)
        return self

    def resolve_weights_path(self) -> Path:
        wp = Path(self.model.weights_path)
        if wp.is_absolute() and wp.exists():
            return wp
        candidate = self.project_root / "models" / wp.name
        if candidate.exists():
            return candidate
        if wp.exists():
            return wp.resolve()
        # Ultralytics may download by name (e.g. yolo11n.pt)
        return wp

    def ensure_directories(self) -> None:
        for attr in (
            "output_dir",
            "annotated_images_dir",
            "annotated_videos_dir",
            "exports_dir",
        ):
            Path(getattr(self.paths, attr)).mkdir(parents=True, exist_ok=True)
        log_path = Path(self.logging.file)
        if not log_path.is_absolute():
            log_path = self.project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path | str | None = None) -> WildlifeConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    db_override = os.environ.get("WILDLIFE_DB_PATH")
    if db_override:
        raw.setdefault("paths", {})["database_path"] = db_override

    cfg = WildlifeConfig(**raw)
    cfg.ensure_directories()
    return cfg


def get_device_string(device_setting: str) -> str:
    import torch

    if device_setting == "cpu":
        return "cpu"
    if device_setting == "cuda":
        if torch.cuda.is_available():
            return "0"
        raise RuntimeError("CUDA requested but not available.")
    if device_setting == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
