"""Video inference with ByteTrack tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from src.config import WildlifeConfig
from src.detection.detector import WildlifeDetector
from src.detection.postprocessing import Detection
from src.storage.repository import WildlifeRepository
from src.tracking.tracker import TrackingSession
from src.utils.file_validation import sanitize_filename
from src.utils.visualization import draw_detections

logger = logging.getLogger("wildlife.video_processor")


@dataclass
class VideoProcessResult:
    job_id: str
    input_path: Path
    output_path: Path | None
    tracking_summary: dict
    total_detections: int
    frames_processed: int
    error: str | None = None


class VideoProcessor:
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
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> VideoProcessResult:
        input_path = Path(input_path)
        start = time.perf_counter()
        job_id = self.repository.create_job(
            input_filename=input_path.name,
            input_type="video",
            model_weights=str(self.config.resolve_weights_path()),
        )

        cap: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        session = TrackingSession(video_name=input_path.name)
        total_detections = 0
        frames_processed = 0
        output_path: Path | None = None

        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            interval = self.config.tracking.frame_sample_interval

            out_name = sanitize_filename(f"tracked_{input_path.stem}.mp4")
            output_path = Path(self.config.paths.annotated_videos_dir) / out_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise ValueError("Could not create output video writer.")

            frame_idx = 0
            processed_idx = 0
            # Reset tracker state for new video
            first_track = True

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    frame_idx += 1
                    continue

                run_inference = frame_idx % interval == 0
                annotated = frame

                if run_inference:
                    detections, stats = self.detector.track_frame(
                        frame,
                        frame_index=processed_idx,
                        confidence=confidence,
                        reset_tracker=first_track,
                    )
                    first_track = False
                    session.update(detections, processed_idx)
                    total_detections += len(detections)

                    for d in detections:
                        if d.track_id is not None:
                            self.repository.upsert_video_track(
                                job_id,
                                input_path.name,
                                d.track_id,
                                d.display_name,
                                processed_idx,
                            )

                    self.repository.insert_detections(
                        job_id=job_id,
                        input_filename=input_path.name,
                        input_type="video",
                        detections=detections,
                        observation_timestamp=None,
                        inference_time_ms=stats.inference_time_ms,
                        model_weights=str(self.config.resolve_weights_path()),
                    )
                    annotated = draw_detections(frame, detections)
                    processed_idx += 1
                    frames_processed = processed_idx

                writer.write(annotated)

                if progress_callback and total_frames > 0 and frame_idx % max(1, interval) == 0:
                    progress_callback(min(1.0, (frame_idx + 1) / total_frames), f"Frame {frame_idx + 1}/{total_frames}")

                frame_idx += 1

            if processed_idx == 0:
                raise ValueError("No valid frames were processed.")

            duration_ms = (time.perf_counter() - start) * 1000
            self.repository.finish_job(job_id, "completed", duration_ms=duration_ms)
            if progress_callback:
                progress_callback(1.0, "Complete")

            return VideoProcessResult(
                job_id=job_id,
                input_path=input_path,
                output_path=output_path,
                tracking_summary=session.summary(),
                total_detections=total_detections,
                frames_processed=frames_processed,
            )
        except Exception as exc:
            logger.exception("Video processing failed: %s", input_path)
            duration_ms = (time.perf_counter() - start) * 1000
            self.repository.finish_job(job_id, "failed", duration_ms=duration_ms, error=str(exc))
            if output_path and output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            return VideoProcessResult(
                job_id=job_id,
                input_path=input_path,
                output_path=None,
                tracking_summary=session.summary(),
                total_detections=total_detections,
                frames_processed=frames_processed,
                error=str(exc),
            )
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
