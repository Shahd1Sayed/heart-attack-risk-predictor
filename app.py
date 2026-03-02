import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Load model bundle ────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model     = bundle["model"]
scaler    = bundle["scaler"]
le_gender = bundle["le_gender"]
le_target = bundle["le_target"]

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Heart Attack Risk Predictor")


# ── Request schema ───────────────────────────────────────────────────────────
class PatientVitals(BaseModel):
    age: int
    gender: str                 # "Male" or "Female"
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    blood_sugar: int
    ck_mb: float
    troponin: float


# ── POST /predict ────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(vitals: PatientVitals):
    # Encode gender: the dataset uses 0 = Female, 1 = Male
    gender_encoded = le_gender.transform([int(vitals.gender)])[0] \
        if vitals.gender.isdigit() \
        else (1 if vitals.gender.strip().lower() == "male" else 0)

    features = np.array([[
        vitals.age,
        gender_encoded,
        vitals.heart_rate,
        vitals.systolic_bp,
        vitals.diastolic_bp,
        vitals.blood_sugar,
        vitals.ck_mb,
        vitals.troponin,
    ]])

    features_scaled = scaler.transform(features)

    prediction    = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    risk_label  = le_target.inverse_transform([prediction])[0]

    # Map probabilities to class names
    class_probs = {
        le_target.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(probabilities)
    }

    return {
        "risk_level": risk_label,
        "probabilities": class_probs,
    }


# ── Serve frontend ──────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))
