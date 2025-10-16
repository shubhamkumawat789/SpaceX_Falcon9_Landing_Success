
import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from .schemas import LandingRequest, LandingResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="Falcon 9 Landing Predictor", version="1.0.0")

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=LandingResponse)
def predict(req: LandingRequest):
    try:
        model = _load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    df = pd.DataFrame([req.model_dump()])
    proba = float(model.predict_proba(df)[0,1])
    label = int(proba >= 0.5)
    return LandingResponse(probability=proba, predicted_label=label)
