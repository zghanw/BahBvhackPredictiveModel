# Predictive Maintenance — Remaining Useful Life (RUL) Estimator

**Team Britney and her Bodyguards' Predictive Model**
A hackathon-grade deep learning pipeline using **PyTorch** to predict RUL of industrial machinery from multivariate sensor time-series (NASA CMAPSS dataset).

## Project Structure

```
vhackusm/
├── data/raw/               # Place raw CMAPSS .txt files here
├── notebooks/              # EDA notebooks (optional)
├── src/
│   ├── config.py           # All hyperparameters & paths
│   ├── data_loader.py      # Load & label CMAPSS data
│   ├── preprocessor.py     # Normalization, sensor selection
│   ├── feature_engineer.py # Rolling stats, trend, degradation features
│   ├── dataset.py          # PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── bilstm.py       # Bidirectional LSTM baseline
│   │   ├── cnn_lstm.py     # CNN + LSTM hybrid
│   │   └── attention.py    # Additive attention module
│   ├── train.py            # Training loop + early stopping
│   ├── evaluate.py         # MAE, RMSE, Score, RUL curves
│   └── visualize.py        # Prediction plots
└── outputs/
    ├── checkpoints/        # Saved model weights
    └── figures/            # Generated plots
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download CMAPSS Dataset
Download from [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  
Place in `data/raw/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

### 3. Train
```bash
python -m src.train
```

### 4. Evaluate
```bash
python -m src.evaluate
```

## Models

| Feature | BiLSTM | CNN_LSTM |
| :--- | :--- | :--- |
| **Logic** | Reads the 30-cycle window twice (early-to-late and late-to-early), understanding health based on history and future context within the window. | Uses a CNN to extract local patterns (e.g., spikes/trends) and an LSTM to model long-range dependencies from those patterns. |
| **Complexity** | High (~683K parameters) | Medium (~323K parameters) |
| **Training Speed** | Slower | Faster |
| **Best For** | Maximum Accuracy | Speed and Efficiency (especially on CPU) |
| **Focus** | Temporal Context (past & future) | Pattern Extraction (detecting spikes/trends) |

*Note: You can easily switch models by updating the `arch` setting in `src/config.py`.*

## Target Metrics (FD001)
| Metric | Target |
|--------|--------|
| RMSE   | < 30   |
| MAE    | < 20   |
