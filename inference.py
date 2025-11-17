
import joblib
import pandas as pd

from config import CFG

def infer_from_row(row_dict: dict):
    cfg = CFG()
    pipe = joblib.load(cfg.model_path)

    df = pd.DataFrame([row_dict])
    pred = pipe.predict(df)[0]

    proba = None
    if hasattr(pipe, "predict_proba"):
        proba_arr = pipe.predict_proba(df)[0]
        proba = {str(i): float(p) for i, p in enumerate(proba_arr)}

    return pred, proba

if __name__ == "__main__":
    # Example usage:
    sample = {
        "src_port": 12345,
        "dst_port": 80,
        "protocol": 6,
        "pkt_count": 10,
        "byte_count": 1500,
    }
    pred, proba = infer_from_row(sample)
    print("Predicted:", pred)
    print("Probabilities:", proba)
