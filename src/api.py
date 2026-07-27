"""
api.py — FastAPI backend for the Predictive Maintenance Dashboard.

Endpoints:
  GET  /                        → health check
  GET  /api/engines             → list available test engines for a subset
  GET  /api/predict/{engine_id} → run inference for one engine (full history)
  GET  /api/fleet               → latest RUL snapshot for all engines
  GET  /api/model/metadata      → architecture, subset, metrics info
  WS   /ws/simulate/{engine_id} → WebSocket: streams cycle-by-cycle predictions

Run with:
  uvicorn src.api:app --reload --port 8000
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import cfg, CKPT_DIR, OUTPUT_DIR, BENCHMARK_METRICS
from src.train import prepare_data, build_model, get_device
from src.feature_engineer import get_feature_columns
from src.inference_api import extract_top_sensors, ANOMALY_THRESHOLD

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount dashboard static files ──────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

# ── Module-level cache so we don't reload data/model on every request ─────────
_cache: dict[str, Any] = {}


def _load_pipeline(subset: str = "FD001"):
    """Load and cache the data pipeline + model for a given subset."""
    key = f"pipeline_{subset}"
    if key in _cache:
        return _cache[key]

    cfg.data.subset = subset
    device = get_device(cfg.train.device)

    train_df, test_df, sensor_cols = prepare_data(cfg)
    feature_cols = get_feature_columns(sensor_cols)

    model = build_model(len(feature_cols), cfg).to(device)
    ckpt = CKPT_DIR / f"best_model_{subset}.pt"
    # Fall back to the generic checkpoint if a subset-specific one doesn't exist yet
    if not ckpt.exists():
        ckpt = CKPT_DIR / "best_model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        # Auto-detect arch from checkpoint keys
        if any("cnn." in k for k in state.keys()):
            cfg.model.arch = "cnn_lstm"
            model = build_model(len(feature_cols), cfg).to(device)
        model.load_state_dict(state)
    model.eval()

    _cache[key] = {
        "test_df": test_df,
        "feature_cols": feature_cols,
        "sensor_cols": sensor_cols,
        "model": model,
        "device": device,
        "arch": cfg.model.arch,
        "subset": subset,
    }
    return _cache[key]


def _predict_engine(pipeline: dict, engine_id: int) -> list[dict]:
    """Run inference over all cycles of a single engine. Returns list of cycle records."""
    test_df      = pipeline["test_df"]
    feature_cols = pipeline["feature_cols"]
    sensor_cols  = pipeline["sensor_cols"]
    model        = pipeline["model"]
    device       = pipeline["device"]
    window_size  = cfg.data.window_size

    engine_data = test_df[test_df["engine_id"] == engine_id].sort_values("cycle").reset_index(drop=True)
    if engine_data.empty:
        return []

    data_matrix = engine_data[feature_cols].values.astype(np.float32)
    cycles = engine_data["cycle"].values
    records = []

    for end in range(window_size, len(engine_data) + 1):
        window = data_matrix[end - window_size: end]
        x = torch.from_numpy(window).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x, return_attention=True)
            if isinstance(out, tuple):
                rul_pred, weights = out
                rul_val = float(max(0.0, rul_pred.item()))
                attn = weights[0].cpu().numpy().tolist() if weights is not None else []
            else:
                rul_val = float(max(0.0, out.item()))
                weights = None
                attn = []

        top_sensors = extract_top_sensors(weights, sensor_cols, window_data=window)
        status = "Impaired" if rul_val <= ANOMALY_THRESHOLD else "Healthy"
        records.append({
            "engine_id": engine_id,
            "cycle": int(cycles[end - 1]),
            "predicted_rul": round(rul_val, 2),
            "status": status,
            "attention_weights": attn,
            "top_sensors": top_sensors,
            "timestamp": time.time(),
        })

    return records


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "Predictive Maintenance API"}


@app.get("/api/engines")
def list_engines(subset: str = "FD001"):
    """Return sorted list of engine IDs available in the test set."""
    pipeline = _load_pipeline(subset)
    ids = sorted(pipeline["test_df"]["engine_id"].unique().tolist())
    return {"subset": subset, "engine_ids": ids, "count": len(ids)}


@app.get("/api/predict/{engine_id}")
def predict_engine(engine_id: int, subset: str = "FD001"):
    """Full cycle-by-cycle RUL history for one engine."""
    pipeline = _load_pipeline(subset)
    records = _predict_engine(pipeline, engine_id)
    if not records:
        return {"error": f"Engine {engine_id} not found in {subset}"}
    return {
        "engine_id": engine_id,
        "subset": subset,
        "arch": pipeline["arch"],
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "cycles": records,
    }


@app.get("/api/fleet")
def fleet_snapshot(subset: str = "FD001", max_engines: int = 20):
    """Latest RUL prediction for each engine — fleet overview."""
    pipeline = _load_pipeline(subset)
    ids = sorted(pipeline["test_df"]["engine_id"].unique().tolist())[:max_engines]
    snapshot = []
    for eid in ids:
        records = _predict_engine(pipeline, eid)
        if records:
            last = records[-1]
            snapshot.append({
                "engine_id": eid,
                "predicted_rul": last["predicted_rul"],
                "status": last["status"],
                "last_cycle": last["cycle"],
            })
    return {"subset": subset, "engines": snapshot}


@app.get("/api/model/metadata")
def model_metadata(subset: str = "FD001"):
    """Return model architecture info and known benchmark metrics."""
    pipeline = _load_pipeline(subset)
    subset_ckpt = CKPT_DIR / f"best_model_{subset}.pt"
    generic_ckpt = CKPT_DIR / "best_model.pt"
    ckpt_exists = subset_ckpt.exists() or generic_ckpt.exists()
    ckpt_is_subset_specific = subset_ckpt.exists()
    arch = pipeline["arch"]
    subset_metrics = BENCHMARK_METRICS.get(subset, {}).get(arch, {})
    return {
        "arch": arch,
        "subset": subset,
        "checkpoint_exists": ckpt_exists,
        "checkpoint_subset_specific": ckpt_is_subset_specific,
        "rul_cap": cfg.data.rul_cap,
        "window_size": cfg.data.window_size,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "rmse": subset_metrics.get("rmse"),
        "mae": subset_metrics.get("mae"),
        "feature_count": len(pipeline["feature_cols"]),
    }


@app.websocket("/ws/simulate/{engine_id}")
async def ws_simulate(websocket: WebSocket, engine_id: int, subset: str = "FD001", delay: float = 0.3):
    """Stream cycle-by-cycle predictions over WebSocket."""
    await websocket.accept()
    try:
        pipeline = _load_pipeline(subset)
        records = _predict_engine(pipeline, engine_id)
        if not records:
            await websocket.send_json({"error": f"Engine {engine_id} not found"})
            return
        for record in records:
            await websocket.send_json(record)
            await asyncio.sleep(delay)
        await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        pass
