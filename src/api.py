"""FastAPI app exposing the no-show prediction service.

Run: uvicorn src.api:app --reload
"""
import logging

from fastapi import FastAPI, HTTPException

from src import config, service
from src.service import AppointmentRequest, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="MediTrack AI Appointment Assistant", version=config.MODEL_VERSION)


def require_model():
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.train`.")


@app.get("/health")
def health():
    return {"status": "ok" if service.model is not None else "model_not_loaded",
            "model_version": config.MODEL_VERSION,
            "cache": {**service.cache_stats, "size": len(service.cache)}}


@app.get("/model/info")
def model_info():
    require_model()
    return service.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(request: AppointmentRequest):
    require_model()
    return service.predict(request)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(requests: list[AppointmentRequest]):
    require_model()
    return [service.predict(r) for r in requests]
