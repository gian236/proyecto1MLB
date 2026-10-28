"""Tests for demand forecasting nodes."""

import numpy as np
import pandas as pd

from bike_sharing_forecast.pipelines.demand_forecasting.nodes import (
    build_features,
    train_and_evaluate,
)

WORKWEEK_DAYS = 5
EXPECTED_METRIC_ROWS = 2


def make_hourly_data(n_hours=600):
    """Create a deterministic hourly dataset for unit tests."""
    timestamps = pd.date_range(
        "2012-01-01",
        periods=n_hours,
        freq="h",
    )
    hour = timestamps.hour.to_numpy()

    return pd.DataFrame(
        {
            "datetime": timestamps,
            "season": 1,
            "holiday": 0,
            "workingday": (timestamps.dayofweek < WORKWEEK_DAYS).astype(int),
            "weather": 1,
            "temp": 15 + 8 * np.sin(2 * np.pi * hour / 24),
            "atemp": 16 + 7 * np.sin(2 * np.pi * hour / 24),
            "humidity": 60,
            "windspeed": 10,
            "count": (
                100 + 40 * np.sin(2 * np.pi * hour / 24) + np.arange(n_hours) * 0.05
            )
            .round()
            .astype(int),
        }
    )


def test_build_features_uses_exact_timestamps():
    """Rows without the immediately preceding hour are excluded."""
    raw = make_hourly_data()
    missing_timestamp = pd.Timestamp("2012-01-10 04:00:00")
    raw = raw.loc[raw["datetime"] != missing_timestamp]

    features = build_features(raw)
    timestamp_after_gap = missing_timestamp + pd.Timedelta(hours=1)

    assert timestamp_after_gap not in set(features["datetime"])
    assert features["datetime"].is_unique
    assert features["count_lag_1"].notna().all()
    assert features["count_lag_24"].notna().all()
    assert features["count_lag_168"].notna().all()


def test_training_returns_metrics_predictions_and_model():
    """The training node produces valid evaluation artifacts."""
    raw = make_hourly_data()
    features = build_features(raw)

    metrics, predictions, model = train_and_evaluate(
        features,
        {
            "validation_start": "2012-01-17",
            "iterations": 20,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "random_seed": 42,
        },
    )

    assert len(metrics) == EXPECTED_METRIC_ROWS
    assert {"RMSLE", "MAE", "RMSE", "R2"}.issubset(metrics.columns)
    assert metrics["RMSLE"].notna().all()
    assert predictions["prediction"].notna().all()
    assert (predictions["prediction"] >= 0).all()
    assert len(predictions) > 0
    assert hasattr(model, "predict")
