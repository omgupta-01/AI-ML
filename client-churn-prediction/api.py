"""
api.py — FastAPI inference endpoint for churn prediction model
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "outputs/churn_model.pkl"

app = FastAPI(
    title="Client Churn Prediction API",
    description="Predict client churn probability using XGBoost + SHAP",
    version="1.0.0",
)

# Load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded.")
    except FileNotFoundError:
        print("⚠️  Model not found. Run train.py first.")


class ClientFeatures(BaseModel):
    tenure_days: float
    total_transactions: int
    total_value: float
    avg_txn_value: float
    txn_per_day: float
    days_since_active: int
    product_count: int
    support_tickets: int
    age: int
    segment: int  # encoded

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_days": 365,
                "total_transactions": 48,
                "total_value": 250000.0,
                "avg_txn_value": 5208.33,
                "txn_per_day": 0.13,
                "days_since_active": 30,
                "product_count": 2,
                "support_tickets": 3,
                "age": 42,
                "segment": 1,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    features = pd.DataFrame([client.dict()])
    proba    = model.predict_proba(features)[0][1]
    pred     = bool(proba >= 0.5)

    if proba >= 0.75:
        risk = "HIGH"
    elif proba >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_prediction=pred,
        risk_level=risk,
    )
