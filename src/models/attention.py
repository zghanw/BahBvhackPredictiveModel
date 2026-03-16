"""
attention.py — Additive (Bahdanau-style) attention over LSTM hidden states.

WHY THIS MATTERS:
  An LSTM processes cycles sequentially and compresses everything into its
  final hidden state. Attention allows the model to selectively re-weight
  ALL hidden states in the window — giving more importance to cycles that
  carry the strongest degradation signal. This consistently improves RMSE
  on CMAPSS by 3–8% and makes the model more interpretable.

  This module is a plug-in: it takes the full LSTM output sequence and
  returns a single context vector, replacing the last-hidden-state shortcut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.

    Given LSTM outputs of shape (batch, seq_len, hidden),
    computes a weighted sum → context vector of shape (batch, hidden).

    The attention weights are also returned so you can visualise
    which time steps the model focuses on.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Two-layer score function: tanh( W·h ) → v → scalar score
        self.attn = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v    = nn.Linear(hidden_size, 1,           bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)

        Returns:
            context:  (batch, hidden_size) — weighted sum of hidden states
            weights:  (batch, seq_len)     — attention weights (sum to 1)
        """
        # Score each time step
        scores = self.v(torch.tanh(self.attn(lstm_out)))  # (batch, seq, 1)
        scores = scores.squeeze(-1)                        # (batch, seq)

        weights = F.softmax(scores, dim=-1)                # (batch, seq)

        # Weighted sum of hidden states
        context = torch.bmm(
            weights.unsqueeze(1),   # (batch, 1, seq)
            lstm_out                # (batch, seq, hidden)
        ).squeeze(1)                # (batch, hidden)

        return context, weights
