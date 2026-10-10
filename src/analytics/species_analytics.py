"""Analytics over stored detection records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AnalyticsSummary:
    total_images: int
    total_videos: int
    total_frames_with_detections: int
    total_detection_events: int
    unique_species: int
    unique_track_ids: int
    avg_confidence: float
    avg_inference_ms: float
    most_common_species: str | None
    metrics_note: str


def _empty_summary(note: str) -> AnalyticsSummary:
    return AnalyticsSummary(
        total_images=0,
        total_videos=0,
        total_frames_with_detections=0,
        total_detection_events=0,
        unique_species=0,
        unique_track_ids=0,
        avg_confidence=0.0,
        avg_inference_ms=0.0,
        most_common_species=None,
        metrics_note=note,
    )


def compute_summary(detections_df: pd.DataFrame, jobs_df: pd.DataFrame | None = None) -> AnalyticsSummary:
    if detections_df is None or detections_df.empty:
        return _empty_summary("No detection records yet.")

    df = detections_df.copy()
    images = df[df["input_type"].isin(["image", "batch_image"])]["input_filename"].nunique()
    videos = df[df["input_type"] == "video"]["input_filename"].nunique()
    frames = (
        df[df["input_type"] == "video"]["frame_index"].nunique()
        if "frame_index" in df.columns
        else 0
    )
    species_n = df["species_name"].nunique()
    track_ids = df["track_id"].dropna().nunique()
    avg_conf = float(df["confidence"].mean()) if len(df) else 0.0
    avg_inf = float(df["inference_time_ms"].dropna().mean()) if df["inference_time_ms"].notna().any() else 0.0
    top = df["species_name"].value_counts().index[0] if len(df) else None

    note = (
        "Detection count = model detection events stored. "
        "Track count = unique tracker IDs per video session (not permanent animal IDs). "
        "These are not population estimates."
    )
    return AnalyticsSummary(
        total_images=int(images),
        total_videos=int(videos),
        total_frames_with_detections=int(frames),
        total_detection_events=len(df),
        unique_species=int(species_n),
        unique_track_ids=int(track_ids),
        avg_confidence=avg_conf,
        avg_inference_ms=avg_inf,
        most_common_species=str(top) if top is not None else None,
        metrics_note=note,
    )


def species_distribution(detections_df: pd.DataFrame) -> pd.DataFrame:
    if detections_df.empty:
        return pd.DataFrame(columns=["species_name", "count"])
    counts = detections_df["species_name"].value_counts().reset_index()
    counts.columns = ["species_name", "count"]
    return counts


def confidence_distribution(detections_df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if detections_df.empty:
        return pd.DataFrame(columns=["bin", "count"])
    hist, edges = pd.cut(detections_df["confidence"], bins=bins, retbins=True)
    out = hist.value_counts().sort_index().reset_index()
    out.columns = ["bin", "count"]
    return out


def activity_by_hour(detections_df: pd.DataFrame) -> pd.DataFrame:
    if detections_df.empty or "observation_timestamp" not in detections_df.columns:
        return pd.DataFrame(columns=["hour", "detections"])

    ts = detections_df["observation_timestamp"].dropna()
    if ts.empty:
        return pd.DataFrame(columns=["hour", "detections"])

    parsed = pd.to_datetime(ts, errors="coerce", utc=True)
    valid = parsed.dropna()
    if valid.empty:
        return pd.DataFrame(columns=["hour", "detections"])

    hours = valid.dt.hour.value_counts().sort_index().reset_index()
    hours.columns = ["hour", "detections"]
    return hours


def activity_by_date(detections_df: pd.DataFrame) -> pd.DataFrame:
    if detections_df.empty or "observation_timestamp" not in detections_df.columns:
        return pd.DataFrame(columns=["date", "detections"])

    ts = detections_df["observation_timestamp"].dropna()
    parsed = pd.to_datetime(ts, errors="coerce", utc=True)
    valid = parsed.dropna()
    if valid.empty:
        return pd.DataFrame(columns=["date", "detections"])

    dates = valid.dt.date.value_counts().sort_index().reset_index()
    dates.columns = ["date", "detections"]
    return dates


def per_input_stats(detections_df: pd.DataFrame) -> pd.DataFrame:
    if detections_df.empty:
        return pd.DataFrame()
    g = (
        detections_df.groupby(["input_type", "input_filename"])
        .agg(
            detections=("event_id", "count"),
            species=("species_name", "nunique"),
            avg_confidence=("confidence", "mean"),
            unique_tracks=("track_id", lambda s: s.dropna().nunique()),
        )
        .reset_index()
    )
    return g


def filter_detections_df(
    df: pd.DataFrame,
    species: list[str] | None = None,
    min_confidence: float | None = None,
    input_type: str | None = None,
    source_file: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if species:
        out = out[out["species_name"].isin(species)]
    if min_confidence is not None:
        out = out[out["confidence"] >= min_confidence]
    if input_type:
        out = out[out["input_type"] == input_type]
    if source_file:
        out = out[out["input_filename"].str.contains(source_file, case=False, na=False)]
    return out
