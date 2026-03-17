"""
dataset.py — PyTorch Dataset and DataLoader factory for CMAPSS RUL prediction.

WHY THIS MATTERS:
  Deep learning models learn from fixed-length input sequences. Since engines
  run for varying numbers of cycles, we use a SLIDING WINDOW approach:
  for each cycle t, we take the previous `window_size` cycles as input and
  the RUL at cycle t as the label. This significantly multiplies the number
  of training samples and teaches the model to predict RUL at any stage.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.config import DataConfig, TrainConfig
from src.feature_engineer import get_feature_columns


class CMAPSSDataset(Dataset):
    """
    Sliding-window Dataset for CMAPSS RUL prediction.

    Each item is:
        x: Tensor of shape (window_size, num_features) — the input sequence
        y: Scalar Tensor — the RUL at the last cycle of the window
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], window_size: int):
        """
        Args:
            df          : Preprocessed + feature-engineered DataFrame with 'rul' column
            feature_cols: Input feature column names
            window_size : Number of cycles per input sequence
        """
        self.window_size = window_size
        self.sequences: list[np.ndarray] = []
        self.labels:    list[float]      = []

        self._build_sequences(df, feature_cols)

    def _build_sequences(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Extract sliding-window (X, y) pairs per engine."""
        for _, engine_df in df.groupby("engine_id"):
            engine_df = engine_df.sort_values("cycle")
            data = engine_df[feature_cols].values.astype(np.float32)
            rul  = engine_df["rul"].values.astype(np.float32)
            n = len(data)

            if n < self.window_size:
                # Engine has fewer cycles than window — pad with zeros at the front
                pad   = np.zeros((self.window_size - n, data.shape[1]), dtype=np.float32)
                data  = np.vstack([pad, data])
                rul   = np.concatenate([np.zeros(self.window_size - n), rul])
                n     = self.window_size

            for end in range(self.window_size, n + 1):
                self.sequences.append(data[end - self.window_size : end])
                self.labels.append(rul[end - 1])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.sequences[idx])          # (window, features)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def get_dataloaders(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    sensor_cols: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train, validation, and test DataLoaders from preprocessed DataFrames.

    Validation split: random subset of training sequences (not engines).
    This is the simplest strategy; for a more rigorous split you could
    hold out specific engine IDs.

    Returns:
        train_loader, val_loader, test_loader, feature_cols
    """
    feature_cols = get_feature_columns(sensor_cols)

    # ── Build full training dataset, then split ───────────────────────────
    full_train_ds = CMAPSSDataset(train_df, feature_cols, data_cfg.window_size)
    n_val   = int(len(full_train_ds) * data_cfg.val_split)
    n_train = len(full_train_ds) - n_val

    torch.manual_seed(data_cfg.seed)
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

    # ── Test dataset ──────────────────────────────────────────────────────
    test_ds = CMAPSSDataset(test_df, feature_cols, data_cfg.window_size)

    loader_kwargs = dict(
        batch_size  = train_cfg.batch_size,
        num_workers = train_cfg.num_workers,
        pin_memory  = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"Dataset sizes → train: {n_train:,}  val: {n_val:,}  test: {len(test_ds):,}")
    print(f"Input shape   → (batch, {data_cfg.window_size}, {len(feature_cols)})")

    return train_loader, val_loader, test_loader, feature_cols
