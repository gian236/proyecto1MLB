"""Generate deterministic synthetic data for the Kedro CI smoke test."""

from pathlib import Path

import numpy as np
import pandas as pd

WORKWEEK_DAYS = 5


def main() -> None:
    """Write a small, continuous hourly training dataset."""
    timestamps = pd.date_range(
        "2012-01-01",
        "2012-10-05 23:00:00",
        freq="h",
    )
    rng = np.random.default_rng(42)
    hour = timestamps.hour.to_numpy()
    dayofweek = timestamps.dayofweek.to_numpy()
    day_index = np.arange(len(timestamps))

    daily_pattern = 80 + 60 * np.sin(2 * np.pi * (hour - 6) / 24)
    commute_peak = 90 * np.exp(-(((hour - 8) / 2.0) ** 2)) + 100 * np.exp(
        -(((hour - 17) / 2.5) ** 2)
    )
    weekday_effect = np.where(dayofweek < WORKWEEK_DAYS, 50, -10)
    trend = 0.015 * day_index
    noise = rng.normal(0, 8, size=len(timestamps))

    target = (
        np.maximum(
            0,
            daily_pattern + commute_peak + weekday_effect + trend + noise,
        )
        .round()
        .astype(int)
    )

    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "season": timestamps.month.map(lambda month: ((month % 12) // 3) + 1),
            "holiday": 0,
            "workingday": (dayofweek < WORKWEEK_DAYS).astype(int),
            "weather": 1,
            "temp": 15 + 10 * np.sin(2 * np.pi * timestamps.dayofyear / 365.25),
            "atemp": 16 + 9 * np.sin(2 * np.pi * timestamps.dayofyear / 365.25),
            "humidity": rng.integers(35, 85, size=len(timestamps)),
            "windspeed": rng.uniform(0, 30, size=len(timestamps)),
            "count": target,
        }
    )

    output = Path("data/01_raw/train.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
