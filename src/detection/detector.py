"""YOLO11 detection engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import WildlifeConfig, get_device_string
from src.detection.postprocessing import Detection, filter_detections
from src.detection.preprocessing import ensure_bgr_uint8

logger = logging.getLogger("wildlife.detector")


@dataclass
class InferenceStats:
    inference_time_ms: float
    num_detections: int
    device: str


@dataclass
class ModelInfo:
    weights_path: str
    device: str
    class_names: dict[int, str]
    supported_filter: list[str]
    unsupported_filter_warnings: list[str]
    loaded: bool


class WildlifeDetector:
    """Lazy-loaded YOLO11 detector with class validation."""

    def __init__(self, config: WildlifeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._device: str = "cpu"
        self._class_names: dict[int, str] = {}
        self._allowed_names: set[str] | None = None
        self._warnings: list[str] = []

    def load(self) -> None:
        if self._model is not None:
            return

        from ultralytics import YOLO

        weights = self.config.resolve_weights_path()
        weights_str = str(weights)
        if not Path(weights_str).exists() and "/" not in weights_str:
            logger.info("Weights not found locally; Ultralytics may download: %s", weights_str)

        self._device = get_device_string(self.config.model.device)
        logger.info("Loading YOLO model from %s on device %s", weights_str, self._device)
        self._model = YOLO(weights_str)
        names = self._model.names or {}
        self._class_names = {int(k): str(v) for k, v in names.items()}
        self._configure_class_filter()
        logger.info("Model loaded with %d classes.", len(self._class_names))

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            weights_path=str(self.config.resolve_weights_path()),
            device=self._device,
            class_names=dict(self._class_names),
            supported_filter=list(self.config.model.supported_classes),
            unsupported_filter_warnings=list(self._warnings),
            loaded=self._model is not None,
        )

    def _configure_class_filter(self) -> None:
        requested = [c.strip().lower() for c in self.config.model.supported_classes if c.strip()]
        if not requested:
            self._allowed_names = None
            self._warnings = []
            return

        model_names_lower = {v.lower(): v for v in self._class_names.values()}
        resolved: set[str] = set()
        warnings: list[str] = []
        for req in requested:
            if req in model_names_lower:
                resolved.add(model_names_lower[req])
            else:
                warnings.append(
                    f"Configured class '{req}' is not in the loaded model. "
                    "Train or fine-tune on wildlife data or use matching weights."
                )
        self._allowed_names = resolved if resolved else None
        self._warnings = warnings

    def _display_name(self, class_name: str) -> str:
        aliases = self.config.model.class_aliases
        return aliases.get(class_name, aliases.get(class_name.lower(), class_name))

    def _parse_results(
        self,
        results: Any,
        frame_index: int | None = None,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cid = int(cls[i])
            cname = self._class_names.get(cid, str(cid))
            tid = int(track_ids[i]) if track_ids is not None else None
            x1, y1, x2, y2 = map(float, xyxy[i])
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=cname,
                    display_name=self._display_name(cname),
                    confidence=float(conf[i]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    track_id=tid,
                    frame_index=frame_index,
                )
            )
        return detections

    def detect_image(
        self,
        image: np.ndarray | str | Path,
        confidence: float | None = None,
    ) -> tuple[list[Detection], InferenceStats]:
        self.load()
        conf = confidence if confidence is not None else self.config.model.confidence_threshold
        frame = image
        if isinstance(image, (str, Path)):
            import cv2

            frame = cv2.imread(str(image))
        else:
            frame = ensure_bgr_uint8(image)

        start = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=conf,
            iou=self.config.model.iou_threshold,
            imgsz=self.config.model.image_size,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        raw = self._parse_results(results)
        filtered = filter_detections(raw, conf, self._allowed_names)
        stats = InferenceStats(
            inference_time_ms=elapsed_ms,
            num_detections=len(filtered),
            device=self._device,
        )
        return filtered, stats

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        confidence: float | None = None,
        persist: bool | None = None,
        reset_tracker: bool = False,
    ) -> tuple[list[Detection], InferenceStats]:
        self.load()
        conf = confidence if confidence is not None else self.config.model.confidence_threshold
        persist_flag = persist if persist is not None else self.config.tracking.persist
        frame = ensure_bgr_uint8(frame)

        if reset_tracker:
            persist_flag = False

        start = time.perf_counter()
        results = self._model.track(
            source=frame,
            conf=conf,
            iou=self.config.model.iou_threshold,
            imgsz=self.config.model.image_size,
            device=self._device,
            tracker=self.config.tracking.tracker,
            persist=persist_flag,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        raw = self._parse_results(results, frame_index=frame_index)
        filtered = filter_detections(raw, conf, self._allowed_names)
        stats = InferenceStats(
            inference_time_ms=elapsed_ms,
            num_detections=len(filtered),
            device=self._device,
        )
        return filtered, stats
