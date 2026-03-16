"""
visualize.py — Plotting utilities for RUL prediction results.

Generates:
  1. RUL prediction curve: predicted vs. true RUL over test samples
  2. Error distribution: histogram of (predicted - true) errors
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.config import FIG_DIR


FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style defaults ────────────────────────────────────────────────────────────
PRED_COLOR  = "#4C9BE8"    # blue  — predictions
TRUE_COLOR  = "#E8734C"    # orange — ground truth
GRID_ALPHA  = 0.25
FONT_TITLE  = 13
FONT_LABEL  = 11


def plot_rul_curves(
    predictions: np.ndarray,
    targets:     np.ndarray,
    n_samples:   int  = 500,
    save:        bool = True,
) -> None:
    """
    Plot predicted vs. true RUL for up to `n_samples` test samples.

    A good model will have blue and orange lines tracking closely.
    Systematic patterns in the gap indicate bias (e.g. under-predicting
    long RUL values near the start of an engine's life).
    """
    idx = np.arange(min(n_samples, len(predictions)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, targets[idx],     color=TRUE_COLOR,  label="True RUL",      linewidth=1.2)
    ax.plot(idx, predictions[idx], color=PRED_COLOR,  label="Predicted RUL",
            linewidth=1.2, linestyle="--", alpha=0.85)

    ax.set_title("RUL Prediction vs. Ground Truth", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=FONT_LABEL)
    ax.set_ylabel("Remaining Useful Life (cycles)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LABEL)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    if save:
        path = FIG_DIR / "rul_prediction_curve.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets:     np.ndarray,
    bins:        int  = 40,
    save:        bool = True,
) -> None:
    """
    Histogram of prediction errors: (predicted - true).
    An ideal model has a distribution centred at 0 with small spread.
    Right-skew = tendency to predict too-late (dangerous in maintenance!).
    """
    errors = predictions - targets

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors, bins=bins, color=PRED_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(0,            color="black",         linewidth=1.5, linestyle="--", label="Perfect (0 error)")
    ax.axvline(errors.mean(), color=TRUE_COLOR,     linewidth=1.5, linestyle="-",  label=f"Mean error: {errors.mean():.2f}")

    ax.set_title("Prediction Error Distribution  [Predicted − True RUL]",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Error (cycles)", fontsize=FONT_LABEL)
    ax.set_ylabel("Count",          fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LABEL)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    if save:
        path = FIG_DIR / "error_distribution.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def plot_training_history(
    train_losses: list[float],
    val_rmses:    list[float],
    save:         bool = True,
) -> None:
    """
    Plot training loss and validation RMSE over epochs.
    Useful for diagnosing overfitting vs. underfitting:
      - Val RMSE diverging up while train loss drops → overfitting
      - Both flat → underfitting / learning rate too low
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, color=PRED_COLOR, linewidth=1.5)
    ax1.set_title("Training Loss (MSE)", fontsize=FONT_TITLE, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax1.set_ylabel("MSE Loss", fontsize=FONT_LABEL)
    ax1.grid(True, alpha=GRID_ALPHA)

    ax2.plot(epochs, val_rmses, color=TRUE_COLOR, linewidth=1.5)
    ax2.set_title("Validation RMSE", fontsize=FONT_TITLE, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax2.set_ylabel("RMSE (cycles)", fontsize=FONT_LABEL)
    ax2.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    if save:
        path = FIG_DIR / "training_history.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()
