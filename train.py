
import joblib
import numpy as np

from config import CFG
from data import load_ddos_data
from model import build_default_ensemble
from metrics import compute_metrics
from utils import set_seed, ensure_dir, save_json

def main():
    cfg = CFG()
    set_seed(cfg.random_state)

    print("[INFO] Loading data...")
    X_train, X_val, y_train, y_val = load_ddos_data()
    print(f"[INFO] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    print("[INFO] Building model...")
    pipe = build_default_ensemble(n_features=X_train.shape[1])

    print("[INFO] Fitting model...")
    pipe.fit(X_train, y_train)

    print("[INFO] Evaluating on validation set...")
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val) if hasattr(pipe, "predict_proba") else None

    label_names = sorted(np.unique(y_train))
    metrics = compute_metrics(
        y_val, y_pred, y_proba, label_names=[str(l) for l in label_names]
    )

    print("[INFO] Metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Weighted F1:", metrics["weighted_f1"])
    if metrics.get("macro_roc_auc_ovr") is not None:
        print("Macro ROC-AUC (OVR):", metrics["macro_roc_auc_ovr"])

    ensure_dir("artifacts")
    joblib.dump(pipe, cfg.model_path)
    save_json(metrics, cfg.metrics_path)

    print(f"[INFO] Saved model to {cfg.model_path}")
    print(f"[INFO] Saved metrics to {cfg.metrics_path}")

if __name__ == "__main__":
    main()
