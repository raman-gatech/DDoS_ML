
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from config import CFG

cfg = CFG()

app = FastAPI(title="DDoS Multi-Class Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FlowSample(BaseModel):
    features: dict

@app.on_event("startup")
def load_model():
    global pipe
    pipe = joblib.load(cfg.model_path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(sample: FlowSample):
    df = pd.DataFrame([sample.features])
    pred = pipe.predict(df)[0]

    proba = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(df)[0]
        proba = {str(i): float(p) for i, p in enumerate(probs)}

    return {
        "prediction": str(pred),
        "probabilities": proba
    }
