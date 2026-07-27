# Predictive Maintenance for ASEAN SMEs — Project Overview

---

## The Problem We're Solving

Small and medium factories across ASEAN run on aging machinery with thin margins. When a motor or turbine fails unexpectedly, production halts for days or weeks. The two common approaches today are both wasteful:

- **Reactive maintenance** — you fix it after it breaks. Expensive, unpredictable.
- **Preventative maintenance** — you replace parts on a schedule, whether they need it or not. Also wasteful.

We built a third option: **know exactly when a machine is going to fail, before it does.**

---

## What This System Does

It analyzes real-time sensor data from industrial machinery — temperature, pressure, speed, flow — and predicts the **Remaining Useful Life (RUL)**: how many operating cycles are left before the machine needs maintenance.

The model was trained and validated on NASA's CMAPSS turbofan engine dataset, which is the industry-standard benchmark for this type of problem. Our results beat the target thresholds across all four dataset variants.

---

## How It Works — End to End

**1. Sensor data comes in**
Each machine reports 21 sensor readings every cycle. We drop 6 sensors that carry no useful signal, leaving 15 informative ones. We then engineer ~135 features per cycle — rolling averages, trend rates, exponential smoothing — to extract the degradation signal from noisy raw readings.

**2. The AI model predicts RUL**
We have two deep learning architectures:
- **BiLSTM** — reads the last 30 cycles in both directions to understand the full degradation context. Higher accuracy, ~683K parameters.
- **CNN-LSTM** — a CNN first extracts local spike patterns, then an LSTM models the long-range trend. More efficient, ~323K parameters, nearly identical accuracy.

Both models output a single number: how many cycles remain.

**3. The system flags anomalies**
When predicted RUL drops below 30 cycles, the machine status flips from **Healthy** to **Impaired**. That's the actionable signal for the maintenance team.

**4. The dashboard shows everything**
A live web dashboard gives factory managers a full picture:
- Fleet overview — all machines at a glance, color-coded by health
- Per-machine RUL trend over time
- Which sensors are driving the prediction
- An alert log that records every status change

---

## Why Operators Can Trust It

The model doesn't just output a number — it shows its reasoning. The **attention weight chart** highlights which of the last 30 cycles the model focused on most heavily. If it's flagging a machine, you can see *which cycles triggered that concern* and cross-check the physical sensor readings yourself. That transparency is what builds trust with non-technical operators.

---

## How to Use It

**Step 1 — Train the model** (one-time, ~30 min per dataset on CPU)
```
python -m src.train --arch bilstm --subset FD001
```

**Step 2 — Start the API server**
```
uvicorn src.api:app --port 8000
```

**Step 3 — Open the dashboard**
Open `dashboard/index.html` in any browser. No installation needed.

From there, a manager can:
- Select a dataset (FD001–FD004, representing different operating conditions)
- Browse all engines in the fleet panel on the left
- Click any engine to see its full RUL history and current health status
- Hit **Live Simulate** to watch predictions stream in cycle-by-cycle, as if it were a real-time feed

---

## Results

| Dataset | Difficulty | Our RMSE | Target |
|---|---|---|---|
| FD001 | Baseline | 14.3 cycles | < 30 |
| FD002 | Multi-condition | 19.8 cycles | < 30 |
| FD003 | Baseline | 15.3 cycles | < 30 |
| FD004 | Multi-condition + multi-fault | 25.3 cycles | < 30 |

Every variant beats the target. FD004 is the hardest scenario — multiple operating conditions and two distinct fault modes — and we still come in under 30.

---

## What Makes This Scalable

The pipeline is modular by design. The sensor columns, feature engineering, model architecture, and anomaly threshold are all configuration values — not hardcoded assumptions. Adapting this to a different type of machinery means providing new sensor data and retraining. The dashboard and API don't change at all.