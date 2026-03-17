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

    # FIXED: Auto-detect architecture from checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if any('cnn.' in key for key in checkpoint.keys()):
        cfg.model.arch = 'cnn_lstm'
        print(f"Detected CNN-LSTM architecture from checkpoint")
    else:
        cfg.model.arch = 'bilstm'
        print(f"Detected BiLSTM architecture from checkpoint")

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

    # ── Official CMAPSS benchmark extraction ──────────────────────────────────
    # The official benchmark only evaluates the VERY LAST cycle of each engine.
    # Since test_loader has shuffle=False, we can find the indices of the last cycle.
    import numpy as np
    engine_seq_counts = [max(1, len(group) - cfg.data.window_size + 1) for _, group in test_df.groupby("engine_id")]
    last_indices = np.cumsum(engine_seq_counts) - 1

    last_preds = preds[last_indices]
    last_targets = targets[last_indices]

    # Metrics on ALL cycles (continuous monitoring)
    mae_all   = compute_mae(preds, targets)
    rmse_all  = compute_rmse(preds, targets)
    score_all = compute_nasa_score(preds, targets)

    # Metrics on LAST cycles (official benchmark)
    mae_official   = compute_mae(last_preds, last_targets)
    rmse_official  = compute_rmse(last_preds, last_targets)
    score_official = compute_nasa_score(last_preds, last_targets)

    print(f"\n{'='*50}")
    print(f"  Test Results ({cfg.data.subset}) — CONTINUOUS (All {len(preds)} samples)")
    print(f"{'='*50}")
    print(f"  MAE   : {mae_all:.4f} cycles")
    print(f"  RMSE  : {rmse_all:.4f} cycles")
    print(f"  Score : {score_all:.2f}  (Naturally large due to summation)")
    
    print(f"\n{'='*50}")
    print(f"  ** OFFICIAL CMAPSS BENCHMARK (Last {len(last_preds)} samples only)")
    print(f"{'='*50}")
    print(f"  MAE   : {mae_official:.4f} cycles")
    print(f"  RMSE  : {rmse_official:.4f} cycles")
    print(f"  Score : {score_official:.2f}  (Official NASA Score)")
    print(f"{'='*50}\n")

    plot_rul_curves(preds, targets)
    plot_error_distribution(preds, targets)

    return {"mae": mae_official, "rmse": rmse_official, "score": score_official}

if __name__ == "__main__":
    evaluate()
