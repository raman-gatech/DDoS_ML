
import joblib
import numpy as np

from config import CFG
from data import load_ddos_data
from metrics import compute_metrics

def evaluate():
    cfg = CFG()

    print("[INFO] Loading validation data...")
    _, X_val, _, y_val = load_ddos_data()

    print("[INFO] Loading model...")
    pipe = joblib.load(cfg.model_path)

    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val) if hasattr(pipe, "predict_proba") else None

    label_names = sorted(np.unique(y_val))
    metrics = compute_metrics(
        y_val, y_pred, y_proba, label_names=[str(l) for l in label_names]
    )

    print("[INFO] Validation metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Weighted F1:", metrics["weighted_f1"])
    if metrics.get("macro_roc_auc_ovr") is not None:
        print("Macro ROC-AUC (OVR):", metrics["macro_roc_auc_ovr"])

if __name__ == "__main__":
    evaluate()
