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
    learning_rate: float = 5e-4        # Reduced for more stable training
    weight_decay: float = 1e-4         # AdamW regularisation
    patience: int = 20                 # Increased patience for better convergence
    min_delta: float = 0.1             # Minimum improvement to reset patience
    scheduler: str = "cosine"          # "cosine" | "step" | "none"
    clip_grad_norm: float = 1.0        # Gradient clipping max norm
    num_workers: int = 0               # DataLoader workers (0 = main process)
    device: str = "auto"               # "auto" | "cpu" | "cuda" Though our codebase mainly focuses on cpu.


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ── Default instance ──────────────────────────────────────────────────────────
cfg = Config()

# ── Benchmark results (from README) ──────────────────────────────────────────
# Keyed by subset → arch → metric. Used by the API metadata endpoint.
BENCHMARK_METRICS: dict[str, dict[str, dict[str, float]]] = {
    "FD001": {"bilstm": {"rmse": 14.28, "mae": 10.24}, "cnn_lstm": {"rmse": 15.26, "mae": 11.03}},
    "FD002": {"bilstm": {"rmse": 19.84, "mae": 14.35}, "cnn_lstm": {"rmse": 19.22, "mae": 13.30}},
    "FD003": {"bilstm": {"rmse": 15.31, "mae": 10.59}, "cnn_lstm": {"rmse": 15.32, "mae": 10.23}},
    "FD004": {"bilstm": {"rmse": 25.28, "mae": 18.11}, "cnn_lstm": {"rmse": 28.41, "mae": 20.87}},
}
