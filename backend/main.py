"""
Hourly Rainfall Prediction – FastAPI Backend
=============================================
Endpoints:
  POST /upload-model   — Upload a ZIP containing model.h5, scaler.pkl, config.json
  POST /predict        — Predict 24-hour rainfall for a given date
"""

import os
import json
import shutil
import zipfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
LATITUDE = float(os.getenv("LATITUDE", "21.1458"))
LONGITUDE = float(os.getenv("LONGITUDE", "79.0882"))

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hourly Rainfall Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session state (single-user MVP)
# ---------------------------------------------------------------------------
session: dict = {
    "model": None,
    "scaler": None,
    "config": None,
    "metadata": None,
    "filename": None,
}

REQUIRED_FILES = {"model.h5", "scaler.pkl", "config.json"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    date: str  # YYYY-MM-DD


class PredictResponse(BaseModel):
    date: str
    predictions: list[float]


# ---------------------------------------------------------------------------
# POST /upload-model
# ---------------------------------------------------------------------------
@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Accept a ZIP, validate, extract model artefacts, and keep in memory."""

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    # Save uploaded file to a temp location
    tmp_path = UPLOAD_DIR / "temp_upload.zip"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Validate ZIP contents
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            names = set(zf.namelist())
            # Also check for files inside a single top-level folder
            flat_names = {os.path.basename(n) for n in names if not n.endswith("/")}
            if not REQUIRED_FILES.issubset(flat_names):
                missing = REQUIRED_FILES - flat_names
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP is missing required files: {', '.join(missing)}",
                )

            # Extract to uploads directory
            extract_dir = UPLOAD_DIR / "model_files"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid ZIP archive.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Find the actual files (may be nested in a subfolder)
    model_dir = _find_model_dir(extract_dir)

    # Load artefacts into memory
    try:
        import tensorflow as tf

        session["model"] = tf.keras.models.load_model(str(model_dir / "model.h5"), compile=False)
        session["scaler"] = joblib.load(str(model_dir / "scaler.pkl"))

        with open(model_dir / "config.json", "r") as f:
            session["config"] = json.load(f)

        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                session["metadata"] = json.load(f)
        else:
            session["metadata"] = None

        session["filename"] = file.filename

    except Exception as exc:
        session["model"] = None
        raise HTTPException(status_code=500, detail=f"Failed to load model artefacts: {exc}")

    return {
        "status": "success",
        "filename": file.filename,
        "config": session["config"],
        "metadata": session["metadata"],
    }


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run 24-hour rainfall prediction for the requested date."""

    # --- Validation ---------------------------------------------------------
    if session["model"] is None:
        raise HTTPException(status_code=400, detail="No model uploaded yet. Please upload a model package first.")

    try:
        target_date = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    config = session["config"]
    look_back: int = config.get("look_back", 15)
    feature_order: list[str] = config.get("feature_order", [])

    # --- Fetch weather data -------------------------------------------------
    try:
        weather_data = _fetch_weather_data(target_date, look_back, feature_order)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    # --- Prepare input & predict --------------------------------------------
    try:
        predictions = _run_prediction(weather_data, look_back, feature_order)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictResponse(
        date=req.date,
        predictions=[round(float(p), 2) for p in predictions],
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": session["model"] is not None,
        "filename": session.get("filename"),
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _find_model_dir(extract_dir: Path) -> Path:
    """Locate the directory that contains model.h5 (handles nested folders)."""
    if (extract_dir / "model.h5").exists():
        return extract_dir
    for child in extract_dir.iterdir():
        if child.is_dir() and (child / "model.h5").exists():
            return child
    raise HTTPException(status_code=400, detail="Could not locate model.h5 inside the ZIP.")


FEATURE_MAP = {
    # Map config feature names → OpenWeatherMap JSON fields
    "TEMP": lambda h: h.get("temp", 0),
    "RH": lambda h: h.get("humidity", 0),
    "WIND SPEED": lambda h: h.get("wind_speed", 0),
    "WIND DIR": lambda h: h.get("wind_deg", 0),
    "WIND DIR2": lambda h: h.get("wind_deg", 0),
    "WIND SP2": lambda h: h.get("wind_speed", 0),
    "RAINR": lambda h: h.get("rain", {}).get("1h", 0) if isinstance(h.get("rain"), dict) else 0,
}


def _fetch_weather_data(
    target_date: datetime,
    look_back: int,
    feature_order: list[str],
) -> np.ndarray:
    """
    Fetch hourly weather from OpenWeatherMap for ``look_back + 24`` hours
    starting from ``look_back`` hours before the target date midnight.

    Returns an ndarray of shape ``(look_back + 24, n_features)``.
    """
    if not OPENWEATHERMAP_API_KEY or OPENWEATHERMAP_API_KEY == "your_api_key_here":
        # --- DEMO / FALLBACK: generate synthetic weather data ---------------
        return _generate_synthetic_weather(target_date, look_back, feature_order)

    total_hours = look_back + 24
    start_ts = int((target_date - timedelta(hours=look_back)).timestamp())

    # One Call API 3.0 – timemachine (or current+forecast for future dates)
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    all_hourly: list[dict] = []

    # We may need multiple day calls for timemachine
    for day_offset in range(0, (total_hours // 24) + 2):
        dt = start_ts + day_offset * 86400
        resp = requests.get(
            url,
            params={
                "lat": LATITUDE,
                "lon": LONGITUDE,
                "dt": dt,
                "appid": OPENWEATHERMAP_API_KEY,
                "units": "metric",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            # Try the forecast API for future dates
            if resp.status_code == 400 or resp.status_code == 401:
                return _fetch_forecast_weather(target_date, look_back, feature_order)
            raise RuntimeError(f"OpenWeatherMap returned {resp.status_code}: {resp.text}")

        data = resp.json()
        hourly = data.get("data", data.get("hourly", []))
        if isinstance(hourly, list):
            all_hourly.extend(hourly)

    # Build feature matrix
    rows = []
    for h in all_hourly[:total_hours]:
        row = []
        for feat in feature_order:
            extractor = FEATURE_MAP.get(feat.upper())
            if extractor:
                row.append(extractor(h))
            else:
                row.append(0)
        rows.append(row)

    # Pad if we didn't get enough data
    while len(rows) < total_hours:
        rows.append(rows[-1] if rows else [0] * len(feature_order))

    return np.array(rows, dtype=np.float32)


def _fetch_forecast_weather(
    target_date: datetime,
    look_back: int,
    feature_order: list[str],
) -> np.ndarray:
    """Fallback: use 5-day/3-hour forecast API and interpolate."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    resp = requests.get(
        url,
        params={
            "lat": LATITUDE,
            "lon": LONGITUDE,
            "appid": OPENWEATHERMAP_API_KEY,
            "units": "metric",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Forecast API error {resp.status_code}: {resp.text}")

    data = resp.json()
    total_hours = look_back + 24
    rows = []

    for item in data.get("list", []):
        main = item.get("main", {})
        wind = item.get("wind", {})
        rain = item.get("rain", {})
        h = {
            "temp": main.get("temp", 0),
            "humidity": main.get("humidity", 0),
            "wind_speed": wind.get("speed", 0),
            "wind_deg": wind.get("deg", 0),
            "rain": {"1h": rain.get("3h", 0) / 3} if rain else {"1h": 0},
        }
        row = []
        for feat in feature_order:
            extractor = FEATURE_MAP.get(feat.upper())
            row.append(extractor(h) if extractor else 0)
        # Each forecast point covers 3 hours – duplicate for interpolation
        for _ in range(3):
            rows.append(row)

    while len(rows) < total_hours:
        rows.append(rows[-1] if rows else [0] * len(feature_order))

    return np.array(rows[:total_hours], dtype=np.float32)


def _generate_synthetic_weather(
    target_date: datetime,
    look_back: int,
    feature_order: list[str],
) -> np.ndarray:
    """
    Generate plausible synthetic weather data for demo/testing
    when no API key is configured.
    """
    total_hours = look_back + 24
    rng = np.random.RandomState(int(target_date.strftime("%Y%m%d")))

    rows = []
    for i in range(total_hours):
        hour = i % 24
        row = []
        for feat in feature_order:
            fname = feat.upper()
            if fname == "TEMP":
                # Diurnal temperature pattern (°C)
                row.append(25 + 8 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 1))
            elif fname == "RH":
                # Humidity (%)
                row.append(min(100, max(30, 70 - 15 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 5))))
            elif fname in ("WIND SPEED", "WIND SP2"):
                row.append(max(0, 3 + 2 * np.sin(hour * np.pi / 12) + rng.normal(0, 1)))
            elif fname in ("WIND DIR", "WIND DIR2"):
                row.append(rng.uniform(0, 360))
            elif fname == "RAINR":
                # Sparse rainfall
                row.append(max(0, rng.exponential(0.5) if rng.random() > 0.6 else 0))
            else:
                row.append(0)
        rows.append(row)

    return np.array(rows, dtype=np.float32)


def _run_prediction(
    weather_data: np.ndarray,
    look_back: int,
    feature_order: list[str],
) -> list[float]:
    """
    Scale the weather data, create look_back sequences, and run 24 predictions.
    """
    model = session["model"]
    scaler = session["scaler"]

    n_features = len(feature_order)

    # Scale data
    scaled = scaler.transform(weather_data)

    predictions = []
    # For each of the 24 hours, create a sequence of length look_back
    for i in range(24):
        start = i  # sliding window
        end = start + look_back
        if end > len(scaled):
            # Pad with last available row
            seq = scaled[start:]
            while len(seq) < look_back:
                seq = np.vstack([seq, seq[-1:]])
        else:
            seq = scaled[start:end]

        # Reshape to (1, look_back, n_features)
        seq = seq.reshape(1, look_back, n_features)

        pred = model.predict(seq, verbose=0)

        # The model may output scaled rainfall – try to inverse-transform
        # We assume the first feature in feature_order is the target (RAINR)
        pred_val = float(pred[0][0]) if pred.ndim > 1 else float(pred[0])

        # Inverse-scale if possible: create a dummy row, set first column to pred
        try:
            dummy = np.zeros((1, n_features))
            dummy[0, 0] = pred_val
            inv = scaler.inverse_transform(dummy)
            pred_val = float(inv[0, 0])
        except Exception:
            pass

        predictions.append(max(0.0, pred_val))

    return predictions


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
