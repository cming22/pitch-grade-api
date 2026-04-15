from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

# Allow requests from your app (CORS fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("lgbm_model.joblib")

# Feature order (must match training)
FEATURES = [
    "start_speed",
    "spin_rate",
    "extension",
    "az",
    "ax",
    "x0",
    "z0",
    "speed_diff",
    "az_diff",
    "ax_diff",
]

# Request schema
class PitchRequest(BaseModel):
    start_speed: float
    spin_rate: float
    extension: float
    az: float
    ax: float
    x0: float
    z0: float
    speed_diff: float
    az_diff: float
    ax_diff: float

# Health check
@app.get("/")
def root():
    return {"ok": True, "message": "Pitch grade API is live"}

# Prediction endpoint
@app.post("/predict")
def predict(payload: PitchRequest):
    df = pd.DataFrame(
        [[getattr(payload, f) for f in FEATURES]],
        columns=FEATURES
    )

    pred = model.predict(df)[0]

    # Convert raw prediction to tjStuff+
    raw = float(pred)
    tj_stuff_plus = 100 - (((raw - 0.35) / 0.68) * 10)

    return {
        "pitch_grade": round(tj_stuff_plus, 1)
    }
