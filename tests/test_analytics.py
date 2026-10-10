import pandas as pd

from src.analytics.species_analytics import (
    activity_by_hour,
    compute_summary,
    species_distribution,
)


def test_compute_summary_empty():
    s = compute_summary(pd.DataFrame())
    assert s.total_detection_events == 0


def test_species_distribution_and_hour():
    df = pd.DataFrame(
        {
            "event_id": ["1", "2", "3"],
            "input_type": ["image", "image", "batch_image"],
            "input_filename": ["a.jpg", "b.jpg", "c.jpg"],
            "species_name": ["bear", "bear", "elephant"],
            "confidence": [0.9, 0.8, 0.7],
            "track_id": [None, None, None],
            "frame_index": [0, 0, 0],
            "inference_time_ms": [10.0, 12.0, 11.0],
            "observation_timestamp": ["2024-01-01 14:00:00", None, "2024-01-02 09:00:00"],
        }
    )
    summary = compute_summary(df)
    assert summary.total_detection_events == 3
    assert summary.unique_species == 2
    assert summary.total_images == 3

    dist = species_distribution(df)
    assert dist.loc[dist["species_name"] == "bear", "count"].iloc[0] == 2

    hours = activity_by_hour(df)
    assert not hours.empty


def test_missing_timestamps_hour_chart_empty():
    df = pd.DataFrame(
        {
            "observation_timestamp": [None, None],
            "species_name": ["bear", "bear"],
            "confidence": [0.5, 0.6],
        }
    )
    hours = activity_by_hour(df)
    assert hours.empty
