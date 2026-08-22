"""FastAPI server exposing Modèle A predictions.

Endpoints
---------
* ``GET  /health``       — liveness + model status
* ``POST /predict``      — single 12×1000 ECG window → class + risk score
* ``GET  /samples``      — list a few sample ECGs from the test fold
* ``GET  /samples/{id}`` — return one sample signal + ground-truth class
* ``GET  /``             — serve the dashboard ``webapp/index.html``

The static dashboard files are mounted at ``/static`` so the same Uvicorn
process can serve both the API and the web UI.

Run with:
    python -m scripts.modelA.api
or
    uvicorn scripts.modelA.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    CLASS_NAMES,
    INFERENCE,
    LEAD_NAMES,
    N_LEADS,
    PTBXL_CSV,
    PTBXL_DIR,
    SAMPLING_RATE_HZ,
    SIGNAL_LENGTH,
    TRAINING,
)
from .inference import ECGPredictor
from .label_mapping import assign_class_from_raw


logger = logging.getLogger("api")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = PROJECT_ROOT / "webapp"


class PredictRequest(BaseModel):
    signal: list[list[float]] = Field(
        ...,
        description=f"ECG signal shape ({N_LEADS}, {SIGNAL_LENGTH}) — lead-major.",
    )
    sampling_rate: int = Field(default=SAMPLING_RATE_HZ)


class PredictFeaturesRequest(BaseModel):
    hr_mean: float = Field(..., description="Mean heart rate (bpm)")
    hr_std: float = Field(..., description="Heart rate std deviation (bpm)")
    sdnn: float = Field(..., description="SDNN — std of NN intervals (ms)")
    rmssd: float = Field(..., description="RMSSD — root mean square of successive differences (ms)")
    pnn50: float = Field(..., description="pNN50 — proportion of NN intervals > 50 ms (%)")
    qrs_width_mean: float = Field(..., description="Mean QRS complex width (ms)")
    qrs_width_std: float = Field(..., description="QRS width std deviation (ms)")
    pr_interval_mean: float = Field(..., description="Mean PR interval (ms)")
    qt_interval_mean: float = Field(..., description="Mean QT interval (ms)")
    st_level_mean: float = Field(..., description="Mean ST segment level (mV)")
    t_amplitude_mean: float = Field(..., description="Mean T-wave amplitude (mV)")
    n_beats: float = Field(..., description="Number of detected beats")


class PredictResponse(BaseModel):
    label_id: int
    label: str
    probabilities: dict[str, float]
    risk_score: float
    timestamp: str
    latency_ms: float


class SampleListItem(BaseModel):
    ecg_id: int
    filename_lr: str
    true_label_id: int
    true_label: str


class SampleResponse(BaseModel):
    ecg_id: int
    filename_lr: str
    sampling_rate: int
    leads: list[str]
    signal: list[list[float]]
    true_label_id: int
    true_label: str


app = FastAPI(
    title="Modèle A — ECG 12 leads",
    version="0.1.0",
    description=(
        "Système d'aide à la décision (académique). Classifie l'état "
        "cardiaque sur une fenêtre ECG 12 leads de 10 s."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


predictor: Optional[ECGPredictor] = None
_test_index: Optional[pd.DataFrame] = None


def _load_test_index() -> pd.DataFrame:
    global _test_index
    if _test_index is not None:
        return _test_index
    df = pd.read_csv(PTBXL_CSV)
    df["true_label_id"] = df["scp_codes"].map(assign_class_from_raw)
    df = df[df["strat_fold"].isin(TRAINING.test_folds)].reset_index(drop=True)
    df["available"] = df["filename_lr"].map(
        lambda f: (PTBXL_DIR / f"{f}.dat").exists()
    )
    df = df[df["available"]].reset_index(drop=True)
    _test_index = df
    logger.info("Loaded test index: %d records", len(df))
    return df


@app.on_event("startup")
def _startup() -> None:
    global predictor
    predictor = ECGPredictor(device=INFERENCE.device)
    logger.info(
        "Model loaded=%s on device=%s (checkpoint=%s)",
        predictor.loaded, INFERENCE.device, INFERENCE.checkpoint_path,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": bool(predictor and predictor.loaded),
        "device": INFERENCE.device,
        "checkpoint": str(INFERENCE.checkpoint_path),
        "class_names": list(CLASS_NAMES),
        "lead_names": list(LEAD_NAMES),
        "signal_length": SIGNAL_LENGTH,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if predictor is None:
        raise HTTPException(503, "Predictor not initialized.")
    try:
        sig = np.asarray(req.signal, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(400, f"Invalid signal payload: {exc}") from exc
    if sig.ndim != 2:
        raise HTTPException(400, f"Signal must be 2-D, got shape {sig.shape}.")
    try:
        pred = predictor.predict(sig)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {exc}") from exc
    return PredictResponse(**pred.to_dict())


@app.post("/predict_features", response_model=PredictResponse)
def predict_features(req: PredictFeaturesRequest) -> PredictResponse:
    if predictor is None:
        raise HTTPException(503, "Predictor not initialized.")
    features = np.array([
        req.hr_mean, req.hr_std, req.sdnn, req.rmssd, req.pnn50,
        req.qrs_width_mean, req.qrs_width_std,
        req.pr_interval_mean, req.qt_interval_mean,
        req.st_level_mean, req.t_amplitude_mean, req.n_beats,
    ], dtype=np.float32)
    try:
        pred = predictor.predict_from_features(features)
    except Exception as exc:
        logger.exception("Feature inference failed")
        raise HTTPException(500, f"Inference error: {exc}") from exc
    return PredictResponse(**pred.to_dict())


@app.get("/samples", response_model=list[SampleListItem])
def samples(limit: int = 20) -> list[SampleListItem]:
    df = _load_test_index()
    sample = df.sample(min(limit, len(df)), random_state=42)
    return [
        SampleListItem(
            ecg_id=int(r.ecg_id),
            filename_lr=str(r.filename_lr),
            true_label_id=int(r.true_label_id),
            true_label=CLASS_NAMES[int(r.true_label_id)],
        )
        for r in sample.itertuples()
    ]


@app.get("/samples/{ecg_id}", response_model=SampleResponse)
def get_sample(ecg_id: int) -> SampleResponse:
    df = _load_test_index()
    row = df[df["ecg_id"] == ecg_id]
    if row.empty:
        raise HTTPException(404, f"ecg_id {ecg_id} not in test fold.")
    row = row.iloc[0]

    import wfdb
    record = wfdb.rdrecord(str(PTBXL_DIR / row["filename_lr"]))
    sig = np.asarray(record.p_signal, dtype=np.float32)
    if sig.shape[1] != N_LEADS and sig.shape[0] == N_LEADS:
        sig = sig.T
    sig = sig.T
    return SampleResponse(
        ecg_id=int(row["ecg_id"]),
        filename_lr=str(row["filename_lr"]),
        sampling_rate=SAMPLING_RATE_HZ,
        leads=list(LEAD_NAMES),
        signal=sig.tolist(),
        true_label_id=int(row["true_label_id"]),
        true_label=CLASS_NAMES[int(row["true_label_id"])],
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    html = WEBAPP_DIR / "index.html"
    if not html.exists():
        return JSONResponse({"detail": "Dashboard not built yet."}, status_code=404)
    return FileResponse(html)


if WEBAPP_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(WEBAPP_DIR / "static")),
        name="static",
    )


def main() -> int:
    import uvicorn

    uvicorn.run(
        "scripts.modelA.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
