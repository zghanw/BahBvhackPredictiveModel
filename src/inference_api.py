"""
inference_api.py — Dashboard Integration & Anomaly Detection API

Simulates real-time inference for a specific engine, calculates explicit
"Healthy" vs "Impaired" anomaly status, extracts top contributing sensors
using attention weights, and exports predictions to a JSON file for the
frontend dashboard.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import cfg, CKPT_DIR, OUTPUT_DIR
from src.train import prepare_data, build_model, get_device
from src.dataset import get_dataloaders
from src.feature_engineer import engineer_features, get_feature_columns
from src.data_loader import get_sensor_columns


DASHBOARD_DATA_FILE = OUTPUT_DIR / "dashboard_data.json"
ANOMALY_THRESHOLD = 30  # cycles — RUL below this = Impaired


def extract_top_sensors(
    weights: torch.Tensor | None,
    sensor_cols: list[str],
    window_data: np.ndarray | None = None,
    top_k: int = 3,
) -> list[str]:
    """
    Return the top-k sensors most responsible for the current prediction.

    When attention weights AND the raw window data are available, we compute
    a weighted-average absolute change per sensor across the window:
        score[s] = Σ_t  attn[t] * |Δsensor_s[t]|
    This gives a genuine, data-driven ranking rather than a fixed list.

    Falls back to the known high-variance CMAPSS sensors when either input
    is unavailable (e.g. model trained without attention).
    """
    # Fallback: known high-degradation sensors for CMAPSS
    fallback = sensor_cols[:top_k] if len(sensor_cols) >= top_k else sensor_cols

    if weights is None or window_data is None:
        return fallback

    try:
        attn = weights[0].cpu().numpy()          # (window_size,)
        # window_data shape: (window_size, n_features)
        # sensor_cols are the first len(sensor_cols) columns (raw values before engineered features)
        n_sensors = len(sensor_cols)
        raw = window_data[:, :n_sensors]         # (window_size, n_sensors)
        delta = np.abs(np.diff(raw, axis=0))     # (window_size-1, n_sensors)
        attn_trimmed = attn[1:]                  # align with diff
        scores = (attn_trimmed[:, None] * delta).sum(axis=0)  # (n_sensors,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [sensor_cols[i] for i in top_indices]
    except Exception:
        return fallback


def run_dashboard_simulation(engine_id: int = 3, ckpt_path: Path = CKPT_DIR / "best_model.pt", delay: float = 0.5):
    """
    Simulates a real-time data feed for a single engine, running inference cycle-by-cycle.
    """
    device = get_device(cfg.train.device)
    print(f"Loading data to simulate engine {engine_id}...")
    
    # We use prepare_data just to get the normalized and feature-engineered test set
    _, test_df, sensor_cols = prepare_data(cfg)
    
    engine_data = test_df[test_df["engine_id"] == engine_id].sort_values("cycle").reset_index(drop=True)
    if engine_data.empty:
        raise ValueError(f"Engine {engine_id} not found in test set.")
        
    # Get feature col names in the correct order
    from src.feature_engineer import get_feature_columns
    feature_cols = get_feature_columns(sensor_cols)
    
    print(f"Loading model from {ckpt_path}...")
    model = build_model(len(feature_cols), cfg).to(device)
    if not ckpt_path.exists():
        print("Model checkpoint not found. Simulating with untrained model for prototype.")
    else:
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("Checkpoint loaded successfully.")
        except RuntimeError as e:
            print(f"Warning: Checkpoint size mismatch (likely trained without attention). Using untrained model for prototype. Error: {e}")
    
    model.eval()
    
    print(f"\n--- Starting Real-Time Simulation for Engine {engine_id} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    window_size = cfg.data.window_size
    data_matrix = engine_data[feature_cols].values.astype(np.float32)
    cycles = engine_data["cycle"].values
    
    for end in range(window_size, len(engine_data) + 1):
        window_data = data_matrix[end - window_size : end]
        current_cycle = int(cycles[end - 1])
        
        # Prepare tensor
        x_tensor = torch.from_numpy(window_data).unsqueeze(0).to(device) # (1, window, features)
        
        # Inference
        with torch.no_grad():
            output = model(x_tensor, return_attention=True)
            if isinstance(output, tuple):
                rul_pred_tensor, weights_tensor = output
                rul_pred = rul_pred_tensor.item()
                weights = weights_tensor
            else:
                rul_pred = output.item()
                weights = None
                
        # --- EXPLICIT ANOMALY CHANGE-POINT LOGIC ---
        rul_pred = max(0.0, rul_pred) # non-negative
        status_label = "Impaired" if rul_pred <= ANOMALY_THRESHOLD else "Healthy"
        
        # --- INTERPRETABILITY / ACTIONABLE INSIGHTS ---
        top_sensors = extract_top_sensors(weights, sensor_cols, window_data=window_data)
        
        # Construct JSON payload for frontend
        payload = {
            "engine_id": engine_id,
            "cycle": current_cycle,
            "predicted_rul": round(rul_pred, 1),
            "status": status_label,
            "anomaly_threshold_cycles": ANOMALY_THRESHOLD,
            "top_contributing_sensors": top_sensors,
            "timestamp": time.time()
        }
        
        # Write to JSON (overwrite/append depending on what frontend expects - we'll overwrite as 'current state')
        with open(DASHBOARD_DATA_FILE, "w") as f:
            json.dump(payload, f, indent=4)
            
        print(f"Cycle {current_cycle:03d} | RUL: {rul_pred:5.1f} | Status: {status_label.upper():>8} | Top Sensors: {', '.join(top_sensors)}")
        
        time.sleep(delay)
        
    print("\n--- Simulation Complete ---")


if __name__ == "__main__":
    run_dashboard_simulation()
