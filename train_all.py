"""
train_all.py — Train and evaluate all four CMAPSS subsets sequentially.

Usage:
    python train_all.py                        # bilstm on all subsets
    python train_all.py --arch cnn_lstm        # cnn_lstm on all subsets
    python train_all.py --subsets FD001 FD002  # specific subsets only
"""

import argparse
import time
from copy import deepcopy

from src.config import cfg, CKPT_DIR
from src.train import train
from src.evaluate import evaluate

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="bilstm", choices=["bilstm", "cnn_lstm"])
    parser.add_argument("--subsets", nargs="+", default=SUBSETS,
                        choices=SUBSETS, metavar="SUBSET")
    args = parser.parse_args()

    summary = {}
    total_start = time.time()

    for subset in args.subsets:
        print(f"\n{'#'*60}")
        print(f"  {subset}  ({args.subsets.index(subset)+1}/{len(args.subsets)})")
        print(f"{'#'*60}")

        # Configure for this run
        cfg.model.arch   = args.arch
        cfg.data.subset  = subset

        # ── Train ──────────────────────────────────────────────────────
        t0 = time.time()
        _, _, ckpt_path = train(cfg)
        train_mins = (time.time() - t0) / 60

        # ── Evaluate ───────────────────────────────────────────────────
        metrics = evaluate(ckpt_path)

        summary[subset] = {
            "rmse":       round(metrics["rmse"],  2),
            "mae":        round(metrics["mae"],   2),
            "score":      round(metrics["score"], 1),
            "train_mins": round(train_mins, 1),
        }

    # ── Summary table ──────────────────────────────────────────────────────
    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY  —  arch: {args.arch}")
    print(f"{'='*60}")
    print(f"  {'Subset':<8} {'RMSE':>8} {'MAE':>8} {'NASA Score':>12} {'Time (min)':>12}")
    print(f"  {'-'*50}")
    for subset, m in summary.items():
        print(f"  {subset:<8} {m['rmse']:>8.2f} {m['mae']:>8.2f} {m['score']:>12.1f} {m['train_mins']:>12.1f}")
    print(f"{'='*60}")
    print(f"  Total time: {total_mins:.1f} min")
    print(f"  Checkpoints saved to: {CKPT_DIR}/")


if __name__ == "__main__":
    main()
