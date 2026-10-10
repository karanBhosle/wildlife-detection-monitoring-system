"""Single-image inference workflow."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config import WildlifeConfig
from src.detection.detector import WildlifeDetector
from src.detection.postprocessing import Detection
from src.storage.repository import WildlifeRepository
from src.utils.file_validation import extract_observation_timestamp, sanitize_filename
from src.utils.visualization import draw_detections

logger = logging.getLogger("wildlife.image_processor")


@dataclass
class ImageProcessResult:
    job_id: str
    input_path: Path
    annotated_path: Path | None
    detections: list[Detection]
    inference_time_ms: float
    observation_timestamp: str | None
    error: str | None = None


class ImageProcessor:
    def __init__(
        self,
        config: WildlifeConfig,
        detector: WildlifeDetector,
        repository: WildlifeRepository,
    ) -> None:
        self.config = config
        self.detector = detector
        self.repository = repository

    def process(
        self,
        input_path: Path,
        confidence: float | None = None,
        save_annotated: bool = True,
    ) -> ImageProcessResult:
        input_path = Path(input_path)
        start = time.perf_counter()
        obs_ts = extract_observation_timestamp(input_path)
        if obs_ts is None:
            obs_ts = "unknown"

        job_id = self.repository.create_job(
            input_filename=input_path.name,
            input_type="image",
            model_weights=str(self.config.resolve_weights_path()),
            metadata={"observation_timestamp": obs_ts},
        )

        try:
            detections, stats = self.detector.detect_image(input_path, confidence=confidence)
            frame = cv2.imread(str(input_path))
            annotated_path = None
            if save_annotated and frame is not None:
                annotated = draw_detections(frame, detections)
                out_name = sanitize_filename(f"annotated_{input_path.stem}.jpg")
                annotated_path = Path(self.config.paths.annotated_images_dir) / out_name
                cv2.imwrite(str(annotated_path), annotated)

            self.repository.insert_detections(
                job_id=job_id,
                input_filename=input_path.name,
                input_type="image",
                detections=detections,
                observation_timestamp=obs_ts if obs_ts != "unknown" else None,
                inference_time_ms=stats.inference_time_ms,
                model_weights=str(self.config.resolve_weights_path()),
                frame_index=0,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            self.repository.finish_job(job_id, "completed", duration_ms=duration_ms)
            return ImageProcessResult(
                job_id=job_id,
                input_path=input_path,
                annotated_path=annotated_path,
                detections=detections,
                inference_time_ms=stats.inference_time_ms,
                observation_timestamp=obs_ts if obs_ts != "unknown" else None,
            )
        except Exception as exc:
            logger.exception("Image processing failed: %s", input_path)
            duration_ms = (time.perf_counter() - start) * 1000
            self.repository.finish_job(job_id, "failed", duration_ms=duration_ms, error=str(exc))
            return ImageProcessResult(
                job_id=job_id,
                input_path=input_path,
                annotated_path=None,
                detections=[],
                inference_time_ms=0.0,
                observation_timestamp=None,
                error=str(exc),
            )
