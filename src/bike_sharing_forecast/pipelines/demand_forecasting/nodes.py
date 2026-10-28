"""Nodes for hourly bike demand forecasting."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

FEATURE_COLUMNS = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
    "count_lag_1",
    "count_lag_24",
    "count_lag_168",
    "count_mean_24",
    "count_mean_168",
]


def build_features(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, weather, and exact-timestamp lag features."""
    data = train_raw.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values("datetime").reset_index(drop=True)

    if data["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found in training data.")

    if data["count"].isna().any() or (data["count"] < 0).any():
        raise ValueError("Target count contains missing or negative values.")

    timestamps = data["datetime"]
    counts = data.set_index("datetime")["count"].sort_index()

    data["year"] = timestamps.dt.year
    data["month"] = timestamps.dt.month
    data["day"] = timestamps.dt.day
    data["hour"] = timestamps.dt.hour
    data["dayofweek"] = timestamps.dt.dayofweek
    data["dayofyear"] = timestamps.dt.dayofyear

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
    data["year_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
    data["year_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)

    for hours in [1, 24, 168]:
        previous = timestamps - pd.Timedelta(hours=hours)
        data[f"count_lag_{hours}"] = counts.reindex(previous).to_numpy()

    hourly_index = pd.date_range(
        counts.index.min(),
        counts.index.max(),
        freq="h",
    )
    complete_series = counts.reindex(hourly_index)

    data["count_mean_24"] = (
        complete_series.rolling(24, min_periods=24)
        .mean()
        .shift(1)
        .reindex(timestamps)
        .to_numpy()
    )
    data["count_mean_168"] = (
        complete_series.rolling(168, min_periods=168)
        .mean()
        .shift(1)
        .reindex(timestamps)
        .to_numpy()
    )

    # Require complete, strictly historical lag windows.
    data = data.dropna(subset=FEATURE_COLUMNS).copy()

    if data.empty:
        raise ValueError("No rows remain after building lag features.")

    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Missing values remain in model features.")

    return data


def train_and_evaluate(
    features: pd.DataFrame,
    training_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, CatBoostRegressor]:
    """Train CatBoost on earlier dates and evaluate on later dates."""

    # Normalizar datetime después de cargar el CSV intermedio
    features = features.copy()
    features["datetime"] = pd.to_datetime(
        features["datetime"],
        errors="raise",
    )

    cutoff = pd.Timestamp(training_params["validation_start"])

    train_df = features.loc[features["datetime"] < cutoff].copy()

    valid_df = features.loc[features["datetime"] >= cutoff].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Temporal split produced an empty partition.")

    # Ensure validation has no timestamps in the training partition.
    if train_df["datetime"].max() >= valid_df["datetime"].min():
        raise ValueError("Invalid temporal split.")

    x_train = train_df[FEATURE_COLUMNS]
    x_valid = valid_df[FEATURE_COLUMNS]
    y_train = np.log1p(train_df["count"].to_numpy())
    y_valid = valid_df["count"].to_numpy()

    model = CatBoostRegressor(
        iterations=int(training_params["iterations"]),
        depth=int(training_params["depth"]),
        learning_rate=float(training_params["learning_rate"]),
        l2_leaf_reg=float(training_params["l2_leaf_reg"]),
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        random_seed=int(training_params["random_seed"]),
    )
    model.fit(x_train, y_train)

    predicted_log = model.predict(x_valid)
    predicted = np.clip(np.expm1(predicted_log), 0, None)
    baseline = np.clip(valid_df["count_lag_168"].to_numpy(), 0, None)

    def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "model": name,
            "n_validation": len(y_true),
            "RMSLE": float(np.sqrt(mean_squared_log_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    metrics = pd.DataFrame(
        [
            metrics_row("Lag_168_baseline", y_valid, baseline),
            metrics_row("CatBoost_log_lags", y_valid, predicted),
        ]
    )

    predictions = valid_df[["datetime", "count"]].copy()
    predictions["prediction"] = predicted
    predictions["baseline_prediction"] = baseline
    predictions["residual"] = predictions["count"] - predictions["prediction"]
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["hour"] = predictions["datetime"].dt.hour

    return metrics, predictions, model
