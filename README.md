
# DDoS Multi-Class Detection — Ensemble Learning (Production-Grade)

This repository implements a production-grade version of the multi-class DDoS detection system inspired by your paper. It uses:

- **Preprocessing:**
  - ID column dropping
  - Basic missing-value handling
  - One-hot encoding for categorical features
  - Optional PCA-based dimensionality reduction

- **Model:**
  - Soft-voting ensemble of RandomForest and ExtraTrees
  - Class-weighted training to mitigate class imbalance
  - Configurable number of estimators and depth

- **Evaluation:**
  - Accuracy, Macro F1, Weighted F1
  - Multi-class ROC-AUC (OVR)
  - Full classification report
  - Confusion matrix (stored as an array in metrics JSON)

- **Deployment:**
  - FastAPI service exposing `/predict`
  - Accepts JSON with arbitrary feature dict
  - Returns predicted class and probabilities

## Data Layout

The code expects a single CSV file:

```
data/
  ddos.csv
```

Where `ddos.csv` contains:

- Feature columns (numeric + categorical)
- A label column named `label` by default (configurable in `config.py`)
- Optional ID columns: `id`, `flow_id`, `timestamp` (these are dropped if present)

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

This will:

- Load `data/ddos.csv`
- Preprocess and split into train/validation
- Train the ensemble model
- Save the trained pipeline to `artifacts/ddos_ensemble.pkl`
- Save evaluation metrics to `artifacts/metrics.json`

## Evaluation

```bash
python eval.py
```

This reloads the dataset and model and prints validation metrics.

## Inference (Local)

```bash
python inference.py
```

Edit the `sample` dict in `inference.py` to match your feature schema.

## FastAPI Deployment

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

Then send a POST request to `/predict` with a JSON body like:

```json
{
  "features": {
    "src_port": 12345,
    "dst_port": 80,
    "protocol": 6,
    "pkt_count": 10,
    "byte_count": 1500
  }
}
```

You will receive:

```json
{
  "prediction": "1",
  "probabilities": {
    "0": 0.01,
    "1": 0.95,
    "2": 0.03,
    "3": 0.01
  }
}
```

This project is designed to be:
- Easy to extend with more models (XGBoost, LightGBM, etc.)
- Easy to plug into a real-time pipeline (Kafka → FastAPI → this model).
