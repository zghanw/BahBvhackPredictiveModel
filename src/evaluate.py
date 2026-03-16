"""
evaluate.py — Compute evaluation metrics and generate RUL prediction curves.

Metrics used:
  MAE   — Mean Absolute Error (in cycles). Intuitive: "off by N cycles on average."
  RMSE  — Root Mean Squared Error. Penalises large errors more; standard benchmark.
  Score — NASA asymmetric scoring function. Penalises LATE predictions more severely
          than early predictions (better to predict failure early than late!).
          Score = Σ exp(d/13) - 1  if d < 0  (early)
                  Σ exp(d/10) - 1  if d >= 0 (late)  ← steeper penalty
          Lower is better.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import cfg, CKPT_DIR
from src.train import prepare_data, build_model, get_device
from src.dataset import get_dataloaders
from src.visualize import plot_rul_curves, plot_error_distribution


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def compute_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def compute_nasa_score(preds: np.ndarray, targets: np.ndarray) -> float:
    """NASA asymmetric scoring — lower is better."""
    d = preds - targets  # positive d = late prediction (underestimating RUL)
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run model over a DataLoader and return (predictions, targets) arrays."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(ckpt_path: Path = CKPT_DIR / "best_model.pt"):
    device = get_device(cfg.train.device)

    print("\nPreparing data ...")
    train_df, test_df, sensor_cols = prepare_data(cfg)
    _, _, test_loader, feature_cols = get_dataloaders(
        train_df, test_df, cfg.data, cfg.train
    )

    print(f"Loading model from: {ckpt_path}")
    model = build_model(len(feature_cols), cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\nRunning test inference ...")
    preds, targets = run_inference(model, test_loader, device)

    mae   = compute_mae(preds, targets)
    rmse_ = compute_rmse(preds, targets)
    score = compute_nasa_score(preds, targets)

    print(f"\n{'='*40}")
    print(f"  Test Results ({cfg.data.subset})")
    print(f"{'='*40}")
    print(f"  MAE   : {mae:.4f} cycles")
    print(f"  RMSE  : {rmse_:.4f} cycles")
    print(f"  Score : {score:.2f}  (↓ lower is better)")
    print(f"{'='*40}\n")

    plot_rul_curves(preds, targets)
    plot_error_distribution(preds, targets)

    return {"mae": mae, "rmse": rmse_, "score": score}


if __name__ == "__main__":
    evaluate()
