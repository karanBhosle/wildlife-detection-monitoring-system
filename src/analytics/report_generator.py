"""Export and summary reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.species_analytics import compute_summary, species_distribution


class ReportGenerator:
    def __init__(self, exports_dir: str | Path) -> None:
        self.exports_dir = Path(exports_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def export_detections_csv(self, df: pd.DataFrame, filename: str = "detections.csv") -> Path:
        path = self.exports_dir / filename
        df.to_csv(path, index=False)
        return path

    def export_detections_json(self, records: list[dict[str, Any]], filename: str = "detections.json") -> Path:
        path = self.exports_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        return path

    def export_species_summary_csv(self, df: pd.DataFrame, filename: str = "species_summary.csv") -> Path:
        summary = species_distribution(df)
        path = self.exports_dir / filename
        summary.to_csv(path, index=False)
        return path

    def build_job_report(
        self,
        detections_df: pd.DataFrame,
        jobs_df: pd.DataFrame,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        ddf = detections_df
        jdf = jobs_df
        if job_id:
            ddf = ddf[ddf["job_id"] == job_id]
            jdf = jdf[jdf["job_id"] == job_id]

        summary = compute_summary(ddf, jdf)
        species = species_distribution(ddf).to_dict(orient="records")
        failures = []
        if not jdf.empty:
            failures = jdf[jdf["status"] == "failed"][["job_id", "input_filename", "error_message"]].to_dict(
                orient="records"
            )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "summary": summary.__dict__,
            "species_counts": species,
            "failed_jobs": failures,
            "disclaimer": (
                "Counts reflect stored detection events and session track IDs, "
                "not verified unique animals or population sizes."
            ),
        }

    def save_report_json(self, report: dict[str, Any], filename: str = "processing_report.json") -> Path:
        path = self.exports_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return path
