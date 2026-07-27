"""
bilstm.py — Bidirectional LSTM baseline for RUL prediction.

WHY BiLSTM?
  A standard LSTM only sees past cycles. Within a fixed sliding window,
  a BiLSTM also reads the sequence in reverse, effectively giving each
  time step context from both directions. For a window of 30 cycles,
  this means the model can see "what came after cycle 5" to better
  characterise the health state at cycle 5. In practice this gives a
  consistent 5–10% RMSE improvement over a unidirectional LSTM.

Architecture:
    Input (batch, window, features)
        ↓
    BiLSTM × num_layers  [hidden*2 because bidirectional]
        ↓
    AdditiveAttention  (optional)  OR  last hidden state
        ↓
    Dropout → Linear(hidden*2 → 64) → ReLU → Linear(64 → 1)
        ↓
    Predicted RUL (scalar, in cycles)
"""

import torch
import torch.nn as nn
from src.models.attention import AdditiveAttention
from src.config import ModelConfig


class BiLSTMModel(nn.Module):
    def __init__(self, input_size: int, cfg: ModelConfig):
        """
        Args:
            input_size: Number of input features (= len(feature_cols))
            cfg       : ModelConfig instance
        """
        super().__init__()
        self.use_attention = cfg.use_attention
        hidden = cfg.hidden_size

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden,
            num_layers   = cfg.num_layers,
            batch_first  = True,
            dropout      = cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional= True,
        )

        lstm_out_dim = hidden * 2  # bidirectional doubles the hidden size

        if self.use_attention:
            self.attention = AdditiveAttention(lstm_out_dim)

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(lstm_out_dim, 64),
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
        out, _ = self.lstm(x)     # out: (batch, seq, hidden*2)

        if self.use_attention:
            context, weights = self.attention(out)   # context: (batch, hidden*2), weights: (batch, seq)
        else:
            context = out[:, -1, :]            # take last time step
            # Post-hoc pseudo-attention: L2 norm of each hidden state, softmax-normalised.
            # Reflects how strongly each time step activated the LSTM — a valid proxy
            # for importance when the model was trained without an explicit attention layer.
            weights = torch.softmax(out.norm(dim=-1), dim=-1)  # (batch, seq)

        rul = self.head(context).squeeze(-1)   # (batch,)

        if return_attention:
            return rul, weights
        return rul
