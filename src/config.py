"""
config.py — Central configuration for the RUL predictive maintenance pipeline.
All hyperparameters and paths live here. Change values here to run experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FIG_DIR    = OUTPUT_DIR / "figures"


@dataclass
class DataConfig:
    # Which CMAPSS subset to use: "FD001", "FD002", "FD003", "FD004"
    subset: str = "FD001"

    # Piece-wise linear RUL cap (cycles).
    # Engines are often healthy at the start; we cap max RUL at this value.
    rul_cap: int = 125

    # Sliding-window sequence length (number of cycles per sample)
    window_size: int = 30

    # Fraction of training engines held out for validation
    val_split: float = 0.2

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ModelConfig:
    # Architecture: "bilstm" | "cnn_lstm"
    arch: str = "bilstm"

    # LSTM hidden size
    hidden_size: int = 128

    # Number of LSTM layers
    num_layers: int = 2

    # Dropout probability (applied between LSTM layers and before head)
    dropout: float = 0.3

    # CNN filters (only used by cnn_lstm)
    cnn_filters: int = 64

    # CNN kernel size
    cnn_kernel: int = 3

    # Use attention on top of LSTM hidden states
    use_attention: bool = False


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4         # AdamW regularisation
    patience: int = 15                  # Early stopping patience (epochs)
    min_delta: float = 0.1             # Minimum improvement to reset patience
    scheduler: str = "cosine"          # "cosine" | "step" | "none"
    clip_grad_norm: float = 1.0        # Gradient clipping max norm
    num_workers: int = 0               # DataLoader workers (0 = main process)
    device: str = "cpu"               # "auto" | "cpu"


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ── Default instance ──────────────────────────────────────────────────────────
cfg = Config()
