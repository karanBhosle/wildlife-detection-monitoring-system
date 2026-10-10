"""Tracking session helpers (ByteTrack via Ultralytics in WildlifeDetector)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackingSession:
    """Session-level track statistics. IDs are not permanent animal identities."""

    video_name: str
    unique_track_ids: set[int] = field(default_factory=set)
    frame_count: int = 0
    detection_count: int = 0

    def update(self, detections: list, frame_index: int) -> None:
        self.frame_count = max(self.frame_count, frame_index + 1)
        for d in detections:
            self.detection_count += 1
            if d.track_id is not None:
                self.unique_track_ids.add(int(d.track_id))

    def summary(self) -> dict:
        return {
            "video_name": self.video_name,
            "frames_processed": self.frame_count,
            "detection_events": self.detection_count,
            "unique_track_ids": len(self.unique_track_ids),
            "track_ids": sorted(self.unique_track_ids),
        }
