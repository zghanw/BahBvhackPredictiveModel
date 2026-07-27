"""
train.py — Full training pipeline for the RUL predictive maintenance model.

Includes:
  - Device selection (auto-detects CUDA)
  - Model factory (BiLSTM or CNN+LSTM)
  - MSELoss + AdamW + CosineAnnealingLR
  - Training loop with gradient clipping
  - Early stopping on validation RMSE
  - Best model checkpoint saving
  - Smoke-test mode for quick validation
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import cfg, Config, CKPT_DIR
from src.data_loader import load_raw, add_rul_labels, add_test_rul_labels, get_sensor_columns
from src.preprocessor import RULPreprocessor
from src.feature_engineer import engineer_features
from src.dataset import get_dataloaders
from src.models import BiLSTMModel, CNNLSTMModel


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def build_model(input_size: int, config: Config) -> nn.Module:
    """Instantiate model based on config.model.arch."""
    arch = config.model.arch.lower()
    if arch == "bilstm":
        return BiLSTMModel(input_size, config.model)
    elif arch == "cnn_lstm":
        return CNNLSTMModel(input_size, config.model)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose 'bilstm' or 'cnn_lstm'.")


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(((predictions - targets) ** 2).mean().item())


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(config: Config):
    """Full data pipeline: load → label → preprocess → feature engineer."""
    print(f"\n[1/4] Loading CMAPSS {config.data.subset} ...")
    train_raw, test_raw, rul_series = load_raw(config.data.subset)

    print("[2/4] Adding RUL labels ...")
    train_raw = add_rul_labels(train_raw, config.data.rul_cap)
    test_raw  = add_test_rul_labels(test_raw, rul_series, config.data.rul_cap)

    print("[3/4] Normalising sensors ...")
    pp = RULPreprocessor()
    train_raw = pp.fit_transform(train_raw)
    test_raw  = pp.transform(test_raw)

    print("[4/4] Engineering features ...")
    sensor_cols = get_sensor_columns()
    train_df = engineer_features(train_raw, sensor_cols)
    test_df  = engineer_features(test_raw,  sensor_cols)

    return train_df, test_df, sensor_cols


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation RMSE stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 0.01,  # Reduced min_delta
                 ckpt_path: Path = CKPT_DIR / "best_model.pt"):
        self.patience  = patience
        self.min_delta = min_delta
        self.ckpt_path = ckpt_path
        self.best_rmse = float("inf")
        self.counter   = 0
        self.early_stop = False

    def __call__(self, val_rmse: float, model: nn.Module) -> bool:
        if val_rmse < self.best_rmse - self.min_delta:
            self.best_rmse = val_rmse
            self.counter   = 0
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.ckpt_path)
            print(f"  + New best val RMSE: {val_rmse:.4f} — checkpoint saved.")
            return False
        else:
            self.counter += 1
            print(f"  - No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                print("  x Early stopping triggered.")
                self.early_stop = True
                return True
            return False


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: Config = cfg, smoke_test: bool = False):
    device = get_device(config.train.device)
    print(f"\n{'='*50}")
    print(f" Predictive Maintenance — RUL Training")
    print(f" Device : {device}")
    print(f" Arch   : {config.model.arch}")
    print(f" Subset : {config.data.subset}")
    print(f"{'='*50}")

    # ── Build data ────────────────────────────────────────────────────────
    if smoke_test:
        print("\n[SMOKE TEST] Using tiny synthetic data for 2 epochs ...")
        import numpy as np, pandas as pd
        n_engines, n_cycles = 5, 60
        rows = []
        for eid in range(1, n_engines + 1):
            for c in range(1, n_cycles + 1):
                row = {"engine_id": eid, "cycle": c}
                for s in range(1, 22):
                    row[f"sensor_{s:02d}"] = np.random.randn()
                row.update({"op_setting_1": 0, "op_setting_2": 0, "op_setting_3": 0})
                rows.append(row)
        df = pd.DataFrame(rows)
        from src.data_loader import add_rul_labels
        from src.feature_engineer import engineer_features, get_feature_columns
        df = add_rul_labels(df, config.data.rul_cap)
        from src.preprocessor import RULPreprocessor
        pp = RULPreprocessor()
        df = pp.fit_transform(df)
        from src.data_loader import get_sensor_columns
        sensor_cols = get_sensor_columns()
        df = engineer_features(df, sensor_cols)
        train_df = test_df = df
        config.train.epochs = 2
        config.train.patience = 2
    else:
        train_df, test_df, sensor_cols = prepare_data(config)

    train_loader, val_loader, test_loader, feature_cols = get_dataloaders(
        train_df, test_df, config.data, config.train
    )

    input_size = len(feature_cols)

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(input_size, config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {config.model.arch}  |  Parameters: {n_params:,}")

    criterion  = nn.SmoothL1Loss()  # More robust than MSE for outliers
    optimizer  = AdamW(model.parameters(),
                       lr=config.train.learning_rate,
                       weight_decay=config.train.weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6) \
                 if config.train.scheduler == "cosine" else None

    stopper    = EarlyStopping(config.train.patience, config.train.min_delta,
                               ckpt_path=CKPT_DIR / f"best_model_{config.data.subset}.pt")

    # ── Epoch loop ────────────────────────────────────────────────────────
    print(f"\nTraining for up to {config.train.epochs} epochs ...\n")
    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu()
                all_preds.append(preds)
                all_targets.append(yb)

        val_preds   = torch.cat(all_preds)
        val_targets = torch.cat(all_targets)
        val_rmse    = rmse(val_preds, val_targets)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | {elapsed:.1f}s")

        history["train_loss"].append(avg_loss)
        history["val_rmse"].append(val_rmse)

        # ── Early stopping and scheduler step ─────────────────────────────
        if stopper(val_rmse, model):
            break
            
        # Only step scheduler if we're continuing training
        if scheduler:
            scheduler.step()

    print(f"\nBest Val RMSE: {stopper.best_rmse:.4f}")
    return model, history, stopper.ckpt_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick 2-epoch test with synthetic data")
    parser.add_argument("--arch", default=None, choices=["bilstm", "cnn_lstm"],
                        help="Override model architecture")
    parser.add_argument("--subset", default=None, choices=["FD001","FD002","FD003","FD004"],
                        help="Override CMAPSS subset")
    args = parser.parse_args()

    if args.arch:
        cfg.model.arch = args.arch
    if args.subset:
        cfg.data.subset = args.subset

    train(cfg, smoke_test=args.smoke_test)
