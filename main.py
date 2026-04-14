from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("lgbm_model.joblib")

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

@app.get("/")
def root():
    return {"ok": True, "message": "Pitch grade API is live"}

@app.post("/predict")
def predict(payload: PitchRequest):
    row = [[getattr(payload, f) for f in FEATURES]]
    df = pd.DataFrame(row, columns=FEATURES)
    pred = model.predict(df)[0]
    return {"pitch_grade": float(pred)}