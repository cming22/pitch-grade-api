from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class RawPitch(BaseModel):
    pitch_type: str
    pitcher_hand: Optional[str] = "R"
    RelSpeed: float
    SpinRate: float
    Extension: float
    InducedVertBreak: float
    HorzBreak: float
    RelSide: float
    RelHeight: float


class BatchRequest(BaseModel):
    pitches: List[RawPitch]


@app.get("/")
def root():
    return {"ok": True, "message": "Pitch grade API is live"}


@app.post("/predict")
def predict(payload: PitchRequest):
    df = pd.DataFrame([[getattr(payload, f) for f in FEATURES]], columns=FEATURES)
    pred = model.predict(df)[0]
    raw = float(pred)
    tj_stuff_plus = 100 - (((raw - 0.35) / 0.68) * 10)
    return {"pitch_grade": round(tj_stuff_plus, 1)}


def normalize_pitch_type(pitch_type: str) -> str:
    pt = (pitch_type or "").strip().lower()
    mapping = {
        "ff": "FF",
        "four-seam": "FF",
        "four-seam fastball": "FF",
        "4-seam": "FF",
        "4-seam fastball": "FF",
        "si": "SI",
        "sinker": "SI",
        "two-seam": "SI",
        "two-seam fastball": "SI",
        "fc": "FC",
        "cutter": "FC",
        "sl": "SL",
        "slider": "SL",
        "cu": "CU",
        "curveball": "CU",
        "ch": "CH",
        "changeup": "CH",
    }
    return mapping.get(pt, pitch_type)


def map_ax_x0(pitch: RawPitch):
    hand = (pitch.pitcher_hand or "R").upper()
    az = pitch.InducedVertBreak

    if hand == "L":
        ax = -pitch.HorzBreak
        x0 = pitch.RelSide
    else:
        ax = pitch.HorzBreak
        x0 = -pitch.RelSide

    return az, ax, x0


@app.post("/predict_batch")
def predict_batch(payload: BatchRequest):
    if not payload.pitches:
        return {"baseline_type": None, "pitches": [], "summary": []}

    rows = []
    for p in payload.pitches:
        norm_type = normalize_pitch_type(p.pitch_type)
        az, ax, x0 = map_ax_x0(p)

        rows.append({
            "pitch_type": norm_type,
            "raw_pitch_type": p.pitch_type,
            "pitcher_hand": (p.pitcher_hand or "R").upper(),
            "start_speed": float(p.RelSpeed),
            "spin_rate": float(p.SpinRate),
            "extension": float(p.Extension),
            "az": float(az),
            "ax": float(ax),
            "x0": float(x0),
            "z0": float(p.RelHeight),
        })

    df = pd.DataFrame(rows)

    fastball_types = ["FF", "SI", "FC"]
    fb_candidates = df[df["pitch_type"].isin(fastball_types)].copy()

    if fb_candidates.empty:
        return {
            "baseline_type": None,
            "pitches": [],
            "summary": [],
            "error": "No fastball baseline found"
        }

    fb_grouped = (
        fb_candidates.groupby("pitch_type")
        .agg(
            pitch_count=("pitch_type", "size"),
            avg_speed=("start_speed", "mean"),
            avg_az=("az", "mean"),
            avg_ax=("ax", "mean"),
        )
        .reset_index()
        .sort_values(["pitch_count", "avg_speed"], ascending=[False, False])
    )

    baseline = fb_grouped.iloc[0]
    baseline_type = str(baseline["pitch_type"])
    baseline_speed = float(baseline["avg_speed"])
    baseline_az = float(baseline["avg_az"])
    baseline_ax = float(baseline["avg_ax"])

    df["speed_diff"] = df["start_speed"] - baseline_speed
    df["az_diff"] = df["az"] - baseline_az
    df["ax_diff"] = (df["ax"] - baseline_ax).abs()

    model_df = df[FEATURES].copy()
    preds = model.predict(model_df)

    df["raw_prediction"] = preds.astype(float)
    df["stuff_plus"] = 100 - (((df["raw_prediction"] - 0.35) / 0.68) * 10)
    df["stuff_plus"] = df["stuff_plus"].round(1)

    pitch_results = df.to_dict(orient="records")

    summary = (
        df.groupby("pitch_type")
        .agg(
            stuff_plus=("stuff_plus", "mean"),
            velo=("start_speed", "mean"),
            ivb=("az", "mean"),
            hb=("ax", "mean"),
            spin=("spin_rate", "mean"),
            num=("pitch_type", "size"),
        )
        .reset_index()
    )

    summary["stuff_plus"] = summary["stuff_plus"].round(1)
    summary["velo"] = summary["velo"].round(1)
    summary["ivb"] = summary["ivb"].round(1)
    summary["hb"] = summary["hb"].round(1)
    summary["spin"] = summary["spin"].round(0)

    return {
        "baseline_type": baseline_type,
        "pitches": pitch_results,
        "summary": summary.to_dict(orient="records"),
    }
