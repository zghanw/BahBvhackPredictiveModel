# Predictive Maintenance for SME Resilience: AI-Driven RUL Estimator

**Team Britney and her Bodyguards** present a hackathon-grade deep learning pipeline designed to solve critical maintenance challenges for ASEAN SMEs. By analyzing multivariate sensor time-series data, our model predicts the Remaining Useful Life (RUL) of industrial machinery, shifting factories from inefficient reactive fixes to strategic proactive planning.

**Track:** Machine Learning (Time-Series / Remaining Useful Life Estimation)  
**Primary Goal:** SDG 9: Industry, Innovation, and Infrastructure (Target 9.4)

---

## 🎯 The Problem & Our Solution

### Real-World Context
Small and Medium Enterprises (SMEs) are the backbone of the ASEAN economy, yet they often operate with aging machinery and thin margins. A single motor failure in a rural plant can halt production for weeks. Current maintenance is either **reactive** (fixing after failure) or **preventative** (replacing parts too early). Both drain resources.

### Our AI-Driven Approach
We developed a robust, scalable PyTorch pipeline tailored to the constraints of real-world industrial settings. Our solution addresses all four technical challenges of the hackathon:

1. **Temporal Feature Engineering (Robust Noise Handling):** Raw sensors are noisy. We utilize rolling statistics and Exponentially Weighted Moving Averages (EWMA) per engine to separate actual degradation trends from environmental noise.
2. **Advanced RUL Regression Modeling:** We implemented both **BiLSTM** and **CNN-LSTM** hybrid architectures. The CNN acts as a learned feature extractor for local spikes, while the LSTM models long-range degradation dependencies.
3. **Explicit Anomaly Change-Point Detection:** Our `inference_api.py` doesn't just output a number; it explicitly monitors the exact moment a machine transitions from "Healthy" to "Impaired" (e.g., when predicted RUL drops below a critical threshold).
4. **Actionable Actionable & Interpretability:** Factory operators need to trust the AI. We integrated a custom `AdditiveAttention` layer that highlights *exactly* which past cycles and which specific sensors (e.g., Temperature, Speed) the model is focusing on to warn of an impending failure.

---

## 🏆 Achieved Results (CMAPSS FD001)

Our pipeline significantly outperforms the hackathon targets, proving its readiness for real-world SME deployment.

| Metric     | Target | **Our Result (CNN-LSTM)** |
|------------|--------|---------------------------|
| **RMSE**   | < 30   | **14.12** cycles          |
| **MAE**    | < 20   | **9.27** cycles           |
| **NASA Score** | -  | **~950**                  |

*Note: The NASA Score is calculated strictly on the last recorded cycle of each test engine (Official CMAPSS Benchmark standard), while MAE and RMSE showcase our model's performance on continuous real-time sequences.*
*Achieved a Best Validation RMSE of **8.52** during training.*

---

## 📁 Architecture & Dashboard Handoff

Our modular architecture ensures the model is scalable to different machinery with minimal retraining, and provides a seamless handoff to frontend dashboards.

```text
vhackusm/
├── data/raw/               # CMAPSS Dataset files
├── src/
│   ├── config.py           # Centralized hyperparameters for easy tuning
│   ├── data_loader.py      # Data loading and Piecewise-Linear RUL labeling
│   ├── feature_engineer.py # Signal smoothing and rate-of-change extraction
│   ├── models/             # PyTorch Architectures (BiLSTM, CNN_LSTM, Attention)
│   ├── train.py            # Training sequence with early stopping & gradient clipping
│   ├── evaluate.py         # Standardized Target Metrics (MAE, RMSE, NASA Score)
│   ├── visualize.py        # Generates RUL curves, Error Distributions, & Attention Maps
│   └── inference_api.py    # 🔴 REAL-TIME DASHBOARD API: Exports predictive JSON feed
└── outputs/
    ├── dashboard_data.json # Live feed for frontend (RUL, Status, Top Sensors)
    └── figures/            # Visualizations of model interpretability
```

---

## 🚀 Quick Start

We invite the judges to run our pipeline and verify our performance target.

### 1. Setup & Data
```bash
# Install dependencies
pip install -r requirements.txt

# Download NASA CMAPSS Dataset and place test_FD00(1-4).txt, train_FD00(1-4).txt, RUL_FD00(1-4).txt in:
# -> data/raw/
```

### 2. Full Training & Evaluation Pipeline
```bash
# Train the model
python -m src.train

# Evaluate predictions and generate interpretability plots
python -m src.evaluate
```

This will automatically save all visualizations to the `outputs/figures/` directory, including:
- `rul_prediction_curve.png`: Predicted vs Ground Truth RUL.
- `error_distribution.png`: Histogram of the prediction errors.

### 3. Live Dashboard Simulation
To see how our model integrates with a frontend UI by generating explicit anomaly states and actionable insights (Top Contributing Sensors):
```bash
python -m src.inference_api
```

**Expected Output:**
```text
Loading data to simulate engine 1...
...
--- Starting Real-Time Simulation for Engine 1 ---
Cycle 030 | RUL: 121.4 | Status:  HEALTHY | Top Sensors: sensor_11, sensor_14, sensor_09
Cycle 031 | RUL: 120.8 | Status:  HEALTHY | Top Sensors: sensor_11, sensor_14, sensor_09
...
```
*(This generates a continuous feed in `outputs/dashboard_data.json` simulating a live engine).*

---

## 🧠 Model Comparison

| Feature | BiLSTM | CNN-LSTM (Default) |
| :--- | :--- | :--- |
| **Logic** | Reads the 30-cycle window in both directions to understand health based on history and future context. | Uses a CNN to extract local severity patterns and an LSTM to model overall long-range degradation. |
| **Complexity** | High (~683K parameters) | Medium (~323K parameters) |
| **Focus** | Temporal Context | Pattern Extraction & Computational Efficiency |
*Switch architectures (`cfg.model.arch`) or evaluate different CMAPSS subsets (`cfg.data.subset` from "FD001" to "FD004") easily by updating `src/config.py`.*
---
*Built for the vHack USM Hackathon 2026. Data sourced from the [NASA PCoE Dataset Repository](https://www.nasa.gov/).*
