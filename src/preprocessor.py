"""
preprocessor.py — Sensor normalization and missing-value handling.

WHY THIS MATTERS:
  Neural networks are sensitive to input scale. A sensor reading in the
  thousands will dominate a sensor reading in the range 0-1 unless we
  normalise. We use MinMaxScaler fitted ONLY on training data to avoid
  data leakage into the test/validation sets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import get_sensor_columns


class RULPreprocessor:
    """
    Fit-on-train, transform-on-all normaliser for CMAPSS sensor data.

    Usage:
        pp = RULPreprocessor()
        train_df = pp.fit_transform(train_df)
        test_df  = pp.transform(test_df)
    """

    def __init__(self, feature_cols: list[str] | None = None):
        """
        Args:
            feature_cols: Sensor columns to normalise. Defaults to the
                          informative sensors returned by get_sensor_columns().
        """
        self.feature_cols: list[str] = feature_cols or get_sensor_columns()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on df and return normalised copy (use on training data)."""
        df = self._handle_missing(df)
        df = df.copy()
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted scaler to df (use on val/test data)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        df = self._handle_missing(df)
        df = df.copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

    def inverse_transform_rul(self, rul: np.ndarray) -> np.ndarray:
        """RUL is NOT scaled (it stays in cycle units). This is a no-op passthrough."""
        return rul

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values.
        Strategy:
          1. Forward-fill within each engine (carry last known reading forward).
          2. Zero-fill any remaining NaNs (start of sequence edge case).
        """
        df = df.copy()
        # Sort so each engine's rows are contiguous, then forward-fill.
        # This is equivalent to per-engine ffill and works correctly in pandas 2.x
        # without any risk of dropping the 'engine_id' column.
        df = df.sort_values(["engine_id", "cycle"])
        df = df.ffill()
        df.fillna(0.0, inplace=True)
        return df
