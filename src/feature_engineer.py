"""
feature_engineer.py — Create informative features from raw sensor readings.

WHY THIS MATTERS:
  Raw sensor values alone may not capture the trend of degradation clearly.
  Rolling statistics smooth noise and reveal the direction of change over
  time. Rate-of-change features flag when a sensor is accelerating toward
  failure. Exponential weighted means give more weight to recent data,
  which is critical near failure.

  Important: all features are computed PER ENGINE GROUP so we never mix
  data across engines.
"""

import pandas as pd
import numpy as np
from src.data_loader import get_sensor_columns


# Rolling window sizes (in cycles) for statistical features
ROLL_WINDOWS = [5, 10, 20]


def engineer_features(df: pd.DataFrame,
                      sensor_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Add rolling statistics, EWM, and rate-of-change features to the DataFrame.

    Args:
        df         : DataFrame with sensor columns and 'engine_id', 'cycle'
        sensor_cols: Columns to engineer. Defaults to informative sensors.

    Returns:
        DataFrame with additional feature columns appended.
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns()

    df = df.sort_values(["engine_id", "cycle"]).copy()

    new_features: list[pd.DataFrame] = []

    for col in sensor_cols:
        group = df.groupby("engine_id")[col]

        # ── Rolling mean & std (noise reduction + trend) ───────────────────
        for w in ROLL_WINDOWS:
            new_features.append(
                group.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
                .rename(f"{col}_rmean_{w}")
            )
            new_features.append(
                group.transform(lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0))
                .rename(f"{col}_rstd_{w}")
            )

        # ── Exponential Weighted Mean (EWM) — weights recent cycles more ───
        # span=10 means the last 10 cycles carry ~86% of the weight
        new_features.append(
            group.transform(lambda x: x.ewm(span=10, min_periods=1).mean())
            .rename(f"{col}_ewm")
        )

        # ── Rate of change (diff) — detects acceleration toward failure ────
        new_features.append(
            group.transform(lambda x: x.diff().fillna(0))
            .rename(f"{col}_diff")
        )

    # Concatenate all new columns at once (much faster than column-by-column)
    df = pd.concat([df] + new_features, axis=1)

    # Drop any remaining NaNs that appeared at window boundaries
    df.fillna(0.0, inplace=True)

    return df


def get_feature_columns(sensor_cols: list[str] | None = None) -> list[str]:
    """
    Return ALL feature column names that engineer_features() will produce,
    in the same order. Useful for indexing into model input tensors.
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns()

    cols = list(sensor_cols)
    for col in sensor_cols:
        for w in ROLL_WINDOWS:
            cols += [f"{col}_rmean_{w}", f"{col}_rstd_{w}"]
        cols += [f"{col}_ewm", f"{col}_diff"]
    return cols
