"""
cnn_lstm.py — CNN feature extractor + LSTM temporal model for RUL prediction.

WHY CNN+LSTM?
  Convolutional layers excel at extracting LOCAL temporal patterns — e.g.,
  a sharp spike in vibration over 5 cycles, or a gradual pressure increase
  over 10 cycles. These local features become the input to the LSTM, which
  then models LONG-RANGE dependencies. This hierarchical approach often
  outperforms a pure LSTM because the CNN acts as a learned feature
  extractor, reducing noise before the sequence model sees the data.

Architecture:
    Input (batch, window, features)
        ↓  [transpose to (batch, features, window) for Conv1d]
    Conv1d(features → cnn_filters, kernel) → BatchNorm → ReLU
    Conv1d(cnn_filters → cnn_filters*2, kernel) → BatchNorm → ReLU
        ↓  [transpose back to (batch, new_seq, cnn_filters*2)]
    LSTM × num_layers
        ↓
    AdditiveAttention (optional)  OR  last hidden state
        ↓
    Dropout → Linear(hidden → 64) → ReLU → Linear(64 → 1)
        ↓
    Predicted RUL (scalar, in cycles)
"""

import torch
import torch.nn as nn
from src.models.attention import AdditiveAttention
from src.config import ModelConfig


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size: int, cfg: ModelConfig):
        """
        Args:
            input_size: Number of input features (= len(feature_cols))
            cfg       : ModelConfig instance
        """
        super().__init__()
        self.use_attention = cfg.use_attention
        F = cfg.cnn_filters
        K = cfg.cnn_kernel
        hidden = cfg.hidden_size

        # ── 1D Convolutional feature extractor ───────────────────────────
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_size, F,   kernel_size=K, padding=K // 2),
            nn.BatchNorm1d(F),
            nn.ReLU(),
            nn.Dropout(cfg.dropout * 0.5),         # lighter dropout in CNN

            # Layer 2 — doubles filter count to capture richer patterns
            nn.Conv1d(F, F * 2,        kernel_size=K, padding=K // 2),
            nn.BatchNorm1d(F * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout * 0.5),
        )

        lstm_input_size = F * 2

        # ── LSTM temporal model ──────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = lstm_input_size,
            hidden_size = hidden,
            num_layers  = cfg.num_layers,
            batch_first = True,
            dropout     = cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        if self.use_attention:
            self.attention = AdditiveAttention(hidden)

        # ── Regression head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (batch, window_size, input_size)
            return_attention: If True, returns (rul, attention_weights)

        Returns:
            rul: (batch,) — predicted RUL in cycles
            weights: (batch, window_size) — attention weights (if return_attention=True)
        """
        # Conv1d expects (batch, channels, seq) — transpose
        x = x.transpose(1, 2)             # → (batch, features, window)
        x = self.cnn(x)                   # → (batch, F*2, window)
        x = x.transpose(1, 2)             # → (batch, window, F*2)

        out, _ = self.lstm(x)             # → (batch, window, hidden)

        if self.use_attention:
            context, weights = self.attention(out)    # context: (batch, hidden), weights: (batch, window)
        else:
            context = out[:, -1, :]             # last step
            # Post-hoc pseudo-attention: L2 norm of each hidden state, softmax-normalised.
            weights = torch.softmax(out.norm(dim=-1), dim=-1)  # (batch, seq)

        rul = self.head(context).squeeze(-1)    # (batch,)
        
        if return_attention:
            return rul, weights
        return rul
