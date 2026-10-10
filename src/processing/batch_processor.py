"""Camera-trap batch image processing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import WildlifeConfig
from src.detection.detector import WildlifeDetector
from src.processing.image_processor import ImageProcessor
from src.storage.repository import WildlifeRepository
from src.utils.file_validation import (
    file_content_hash,
    validate_extension,
    validate_file_size,
    validate_image_readable,
)

logger = logging.getLogger("wildlife.batch_processor")


@dataclass
class BatchFileStatus:
    filename: str
    status: str
    detections: int = 0
    message: str = ""
    skipped: bool = False


@dataclass
class BatchProcessResult:
    job_ids: list[str] = field(default_factory=list)
    file_statuses: list[BatchFileStatus] = field(default_factory=list)
    total_detections: int = 0
    species_counts: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0


class BatchProcessor:
    def __init__(
        self,
        config: WildlifeConfig,
        detector: WildlifeDetector,
        repository: WildlifeRepository,
    ) -> None:
        self.config = config
        self.image_processor = ImageProcessor(config, detector, repository)
        self.repository = repository

    def process_paths(
        self,
        paths: list[Path],
        confidence: float | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchProcessResult:
        start = time.perf_counter()
        result = BatchProcessResult()
        allowed = self.config.image.allowed_extensions
        max_mb = self.config.image.max_upload_mb
        max_files = self.config.batch.max_files

        paths = paths[:max_files]
        total = len(paths)

        for i, path in enumerate(paths):
            path = Path(path)
            if progress_callback:
                progress_callback(i, total, path.name)

            validation_failed = False
            for check in (
                validate_extension(path, allowed),
                validate_file_size(path, max_mb),
                validate_image_readable(path),
            ):
                if not check.ok:
                    result.file_statuses.append(
                        BatchFileStatus(filename=path.name, status="failed", message=check.message)
                    )
                    validation_failed = True
                    break
            if validation_failed:
                continue

            if self.config.batch.dedupe_by_content_hash:
                content_hash = file_content_hash(path)
                existing = self.repository.job_exists_for_hash(content_hash)
                if existing:
                    result.file_statuses.append(
                        BatchFileStatus(
                            filename=path.name,
                            status="skipped",
                            skipped=True,
                            message=f"Duplicate content (job {existing}).",
                        )
                    )
                    continue
            else:
                content_hash = None

            # Create job with hash via direct repository call in image processor - extend image processor
            proc = self._process_with_hash(path, confidence, content_hash)
            if proc.error:
                result.file_statuses.append(
                    BatchFileStatus(filename=path.name, status="failed", message=proc.error)
                )
            else:
                counts = {}
                for d in proc.detections:
                    counts[d.display_name] = counts.get(d.display_name, 0) + 1
                    result.species_counts[d.display_name] = (
                        result.species_counts.get(d.display_name, 0) + 1
                    )
                result.total_detections += len(proc.detections)
                result.job_ids.append(proc.job_id)
                result.file_statuses.append(
                    BatchFileStatus(
                        filename=path.name,
                        status="completed",
                        detections=len(proc.detections),
                    )
                )

        result.duration_ms = (time.perf_counter() - start) * 1000
        if progress_callback:
            progress_callback(total, total, "done")
        return result

    def _process_with_hash(self, path: Path, confidence: float | None, content_hash: str | None):
        # Re-create job with hash by wrapping repository.create_job
        job_id = self.repository.create_job(
            input_filename=path.name,
            input_type="batch_image",
            model_weights=str(self.config.resolve_weights_path()),
            input_hash=content_hash,
        )
        # ImageProcessor creates its own job - use lower-level flow
        from src.processing.image_processor import ImageProcessResult
        import cv2
        from src.utils.file_validation import extract_observation_timestamp, sanitize_filename
        from src.utils.visualization import draw_detections

        obs_ts = extract_observation_timestamp(path)
        obs_store = obs_ts if obs_ts else None
        try:
            detections, stats = self.image_processor.detector.detect_image(path, confidence=confidence)
            frame = cv2.imread(str(path))
            annotated_path = None
            if frame is not None:
                annotated = draw_detections(frame, detections)
                out_name = sanitize_filename(f"annotated_{path.stem}.jpg")
                annotated_path = Path(self.config.paths.annotated_images_dir) / out_name
                cv2.imwrite(str(annotated_path), annotated)

            self.repository.insert_detections(
                job_id=job_id,
                input_filename=path.name,
                input_type="batch_image",
                detections=detections,
                observation_timestamp=obs_store,
                inference_time_ms=stats.inference_time_ms,
                model_weights=str(self.config.resolve_weights_path()),
                frame_index=0,
            )
            self.repository.finish_job(job_id, "completed")
            return ImageProcessResult(
                job_id=job_id,
                input_path=path,
                annotated_path=annotated_path,
                detections=detections,
                inference_time_ms=stats.inference_time_ms,
                observation_timestamp=obs_store,
            )
        except Exception as exc:
            self.repository.finish_job(job_id, "failed", error=str(exc))
            return ImageProcessResult(
                job_id=job_id,
                input_path=path,
                annotated_path=None,
                detections=[],
                inference_time_ms=0.0,
                observation_timestamp=None,
                error=str(exc),
            )

    def process_directory(
        self,
        directory: Path | None = None,
        confidence: float | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchProcessResult:
        dir_path = Path(directory or self.config.batch.input_dir)
        if not dir_path.is_absolute():
            dir_path = self.config.project_root / dir_path
        allowed = {e.lower() for e in self.config.image.allowed_extensions}
        paths = sorted(
            p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in allowed
        )
        return self.process_paths(paths, confidence=confidence, progress_callback=progress_callback)
