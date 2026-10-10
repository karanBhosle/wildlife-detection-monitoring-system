"""SQLite persistence for jobs, detections, and tracks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

from src.detection.postprocessing import Detection


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detection_event_id(
    job_id: str,
    input_filename: str,
    frame_index: int | None,
    detection: Detection,
) -> str:
    """Deterministic ID so retries do not duplicate rows (incl. NULL track_id)."""
    parts = (
        job_id,
        input_filename,
        str(frame_index if frame_index is not None else detection.frame_index),
        str(detection.class_id),
        f"{detection.x1:.2f}",
        f"{detection.y1:.2f}",
        f"{detection.x2:.2f}",
        f"{detection.y2:.2f}",
        f"{detection.confidence:.4f}",
        str(detection.track_id if detection.track_id is not None else -1),
    )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


class WildlifeRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    input_filename TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    input_hash TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    processing_duration_ms REAL,
                    model_weights TEXT,
                    error_message TEXT,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS detection_events (
                    event_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    input_filename TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    frame_index INTEGER,
                    observation_timestamp TEXT,
                    processed_at TEXT NOT NULL,
                    species_name TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    track_id INTEGER,
                    inference_time_ms REAL,
                    model_weights TEXT,
                    UNIQUE(job_id, input_filename, frame_index, class_id, x1, y1, x2, y2, confidence, track_id)
                );

                CREATE INDEX IF NOT EXISTS idx_detection_job ON detection_events(job_id);
                CREATE INDEX IF NOT EXISTS idx_detection_species ON detection_events(species_name);
                CREATE INDEX IF NOT EXISTS idx_detection_obs_ts ON detection_events(observation_timestamp);

                CREATE TABLE IF NOT EXISTS video_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    input_filename TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    species_name TEXT,
                    first_frame INTEGER,
                    last_frame INTEGER,
                    detection_count INTEGER DEFAULT 0,
                    UNIQUE(job_id, input_filename, track_id)
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    weights_path TEXT,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL,
                    notes TEXT
                );
                """
            )

    def create_job(
        self,
        input_filename: str,
        input_type: str,
        model_weights: str,
        input_hash: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO processing_jobs
                (job_id, input_filename, input_type, input_hash, status, started_at, model_weights, metadata_json)
                VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    job_id,
                    input_filename,
                    input_type,
                    input_hash,
                    _utc_now(),
                    model_weights,
                    json.dumps(metadata or {}),
                ),
            )
        return job_id

    def finish_job(
        self,
        job_id: str,
        status: str,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = ?, finished_at = ?, processing_duration_ms = ?, error_message = ?
                WHERE job_id = ?
                """,
                (status, _utc_now(), duration_ms, error, job_id),
            )

    def insert_detections(
        self,
        job_id: str,
        input_filename: str,
        input_type: str,
        detections: Iterable[Detection],
        observation_timestamp: str | None,
        inference_time_ms: float | None,
        model_weights: str,
        frame_index: int | None = None,
    ) -> int:
        inserted = 0
        processed_at = _utc_now()
        with self._conn() as conn:
            for d in detections:
                fi = d.frame_index if d.frame_index is not None else frame_index
                event_id = _detection_event_id(job_id, input_filename, fi, d)
                try:
                    conn.execute(
                        """
                        INSERT INTO detection_events (
                            event_id, job_id, input_filename, input_type, frame_index,
                            observation_timestamp, processed_at, species_name, class_id, class_name,
                            confidence, x1, y1, x2, y2, track_id, inference_time_ms, model_weights
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            job_id,
                            input_filename,
                            input_type,
                            fi,
                            observation_timestamp,
                            processed_at,
                            d.display_name,
                            d.class_id,
                            d.class_name,
                            d.confidence,
                            d.x1,
                            d.y1,
                            d.x2,
                            d.y2,
                            d.track_id,
                            inference_time_ms,
                            model_weights,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
        return inserted

    def upsert_video_track(
        self,
        job_id: str,
        input_filename: str,
        track_id: int,
        species_name: str,
        frame_index: int,
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, first_frame, last_frame, detection_count FROM video_tracks
                WHERE job_id = ? AND input_filename = ? AND track_id = ?
                """,
                (job_id, input_filename, track_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO video_tracks
                    (job_id, input_filename, track_id, species_name, first_frame, last_frame, detection_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (job_id, input_filename, track_id, species_name, frame_index, frame_index),
                )
            else:
                conn.execute(
                    """
                    UPDATE video_tracks SET
                        last_frame = CASE WHEN last_frame > ? THEN last_frame ELSE ? END,
                        first_frame = CASE WHEN first_frame < ? THEN first_frame ELSE ? END,
                        detection_count = detection_count + 1,
                        species_name = COALESCE(?, species_name)
                    WHERE id = ?
                    """,
                    (frame_index, frame_index, frame_index, frame_index, species_name, row["id"]),
                )

    def job_exists_for_hash(self, input_hash: str) -> str | None:
        if not input_hash:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT job_id FROM processing_jobs WHERE input_hash = ? AND status = 'completed' LIMIT 1",
                (input_hash,),
            ).fetchone()
        return row["job_id"] if row else None

    def fetch_detections_df(self):
        import pandas as pd

        with self._conn() as conn:
            df = pd.read_sql_query("SELECT * FROM detection_events ORDER BY processed_at DESC", conn)
        return df

    def fetch_jobs_df(self):
        import pandas as pd

        with self._conn() as conn:
            return pd.read_sql_query("SELECT * FROM processing_jobs ORDER BY started_at DESC", conn)

    def fetch_all_detections_as_records(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM detection_events").fetchall()
        return [dict(r) for r in rows]

    def register_model_run(self, run_type: str, weights_path: str, metrics: dict, notes: str = "") -> str:
        run_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO model_runs (run_id, run_type, weights_path, metrics_json, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, run_type, weights_path, json.dumps(metrics), _utc_now(), notes),
            )
        return run_id

    def latest_model_run(self, run_type: str = "evaluation") -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM model_runs WHERE run_type = ? ORDER BY created_at DESC LIMIT 1
                """,
                (run_type,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["metrics"] = json.loads(data.pop("metrics_json") or "{}")
        return data
