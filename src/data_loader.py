"""
data_loader.py — Load and label the NASA CMAPSS turbofan engine dataset.

WHY THIS MATTERS:
  The CMAPSS dataset has no header. We must assign column names manually.
  RUL labels must be computed from the data — engines don't come pre-labelled.
  We apply a "piecewise linear" RUL cap because engines are assumed healthy
  for the first portion of their life, which reduces noise in early-cycle data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_DIR, DataConfig


# ── Column names for CMAPSS ───────────────────────────────────────────────────
COLUMNS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

# Sensors with near-zero or constant variance across FD001 — safely dropped.
# These sensors carry no degradation signal and only add noise.
LOW_INFO_SENSORS = ["sensor_01", "sensor_05", "sensor_10", "sensor_16",
                    "sensor_18", "sensor_19"]


def load_raw(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load train, test, and ground-truth RUL files for a given CMAPSS subset.

    Args:
        subset: One of "FD001", "FD002", "FD003", "FD004"

    Returns:
        train_df  : Training time-series (all engines, all cycles)
        test_df   : Test time-series (truncated — last cycle before failure)
        rul_series: True RUL at the last observed cycle for each test engine
    """
    read = lambda fname: pd.read_csv(
        DATA_DIR / fname, sep=r"\s+", header=None, names=COLUMNS
    )

    train_df = read(f"train_{subset}.txt")
    test_df  = read(f"test_{subset}.txt")
    rul_series = pd.read_csv(
        DATA_DIR / f"RUL_{subset}.txt", header=None
    ).squeeze()
    return train_df, test_df, rul_series


def add_rul_labels(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """
    Compute and attach piecewise-linear RUL labels to a training DataFrame.

    Piecewise linear capping:
      - For cycles far from failure, we cap RUL at `rul_cap`.
      - This reflects that early, healthy cycles have similar sensor signatures
        regardless of their exact distance to failure.
      - Without capping, the model wastes capacity fitting a wide range of
        'healthy' RUL values that look identical in the sensor data.

    Args:
        df     : DataFrame with 'engine_id' and 'cycle' columns
        rul_cap: Maximum RUL value (cycles) to assign

    Returns:
        DataFrame with an added 'rul' column
    """
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_test_rul_labels(test_df: pd.DataFrame,
                        rul_series: pd.Series,
                        rul_cap: int = 125) -> pd.DataFrame:
    """
    Attach RUL labels to the test set using ground-truth RUL for the last cycle,
    then back-fill for earlier cycles within each engine.

    Args:
        test_df   : Test DataFrame
        rul_series: Ground-truth RUL at each engine's last observed cycle
        rul_cap   : Maximum RUL cap

    Returns:
        Test DataFrame with 'rul' column
    """
    last_rul = rul_series.values  # index = engine_id - 1
    groups = []
    for engine_id, group in test_df.groupby("engine_id"):
        group = group.copy().sort_values("cycle")
        final_rul = last_rul[engine_id - 1]
        n = len(group)
        # RUL decreases by 1 each cycle
        group["rul"] = (final_rul + np.arange(n - 1, -1, -1)).clip(0, rul_cap)
        groups.append(group)
    return pd.concat(groups).reset_index(drop=True)


def get_sensor_columns(drop_low_info: bool = True) -> list[str]:
    """Return the list of sensor columns to use as model features."""
    sensors = [f"sensor_{i:02d}" for i in range(1, 22)]
    if drop_low_info:
        sensors = [s for s in sensors if s not in LOW_INFO_SENSORS]
    return sensors
