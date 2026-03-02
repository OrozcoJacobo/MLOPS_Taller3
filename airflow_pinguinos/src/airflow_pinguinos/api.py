from fastapi import FastAPI, HTTPException
import joblib
import json
import os
from pathlib import Path
import pandas as pd

app = FastAPI(title="Penguins Species Prediction API")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))

model = None
registry = None
default_model_name = None


@app.on_event("startup")
def load_model():
    global model, registry, default_model_name

    registry_path = MODELS_DIR / "registry.json"

    if not registry_path.exists():
        print("⚠ registry.json not found. API started without model.")
        return

    with open(registry_path, "r") as f:
        registry = json.load(f)

    default_model_name = registry["default_model"]
    model_path = MODELS_DIR / f"{default_model_name}.joblib"

    if not model_path.exists():
        print("⚠ Default model not found. API started without model.")
        return

    model = joblib.load(model_path)
    print("Model loaded successfully")


@app.get("/")
def root():
    return {
        "message": "Penguins API is running",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(features: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return {"prediction": pred}