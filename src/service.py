"""Prediction service shared by the API and the UI: validate -> cache -> predict -> risk/action."""
import json
import logging
import time
from datetime import date, datetime

import joblib
import pandas as pd
from cachetools import TTLCache
from pydantic import BaseModel, Field, model_validator

from src import config

logger = logging.getLogger(__name__)

try:
    model = joblib.load(config.MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text())
except FileNotFoundError:
    model, metadata = None, {}
    logger.error("Model artifact not found at %s. Run `python -m src.train` first.", config.MODEL_PATH)

cache = TTLCache(maxsize=config.CACHE_MAX_SIZE, ttl=config.CACHE_TTL_SECONDS)
cache_stats = {"hits": 0, "misses": 0}


class AppointmentRequest(BaseModel):
    """Information available *before* the appointment. IDs are never used as model features."""
    patient_id: str | None = Field(None, description="Used for caching/logging only")
    appointment_id: str | None = Field(None, description="Used for caching/logging only")
    age: int = Field(ge=config.AGE_MIN, le=config.AGE_MAX)
    sms_received: bool | None = Field(None, description="None means unknown")
    scheduled_day: datetime = Field(description="When the appointment was booked")
    appointment_day: date
    previous_appointments: int = Field(0, ge=0)
    previous_no_shows: int = Field(0, ge=0)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.previous_no_shows > self.previous_appointments:
            raise ValueError("previous_no_shows cannot exceed previous_appointments")
        if self.appointment_day < self.scheduled_day.date():
            raise ValueError("appointment_day cannot be before scheduled_day")
        if self.appointment_day < date.today():
            raise ValueError("appointment_day is in the past; online scoring is for upcoming appointments")
        return self


class PredictionResponse(BaseModel):
    no_show_probability: float = Field(ge=0, le=1)
    risk_level: str
    recommended_action: str
    factors: list[str]
    model_version: str
    cached: bool


def predict(request: AppointmentRequest) -> PredictionResponse:
    if model is None:
        raise RuntimeError("Model is not loaded")

    # Cache key = every prediction-relevant input + model version (not just patient_id),
    # so changing e.g. sms_received never returns a stale prediction.
    key = (request.model_dump_json(), config.MODEL_VERSION)
    if key in cache:
        cache_stats["hits"] += 1
        return cache[key].model_copy(update={"cached": True})
    cache_stats["misses"] += 1

    start = time.perf_counter()
    row = request.model_dump()
    row["sms_received"] = {True: "yes", False: "no", None: "unknown"}[row["sms_received"]]
    features = pd.DataFrame([row])[config.INPUT_COLUMNS]
    probability = float(model.predict_proba(features)[0, 1])

    if probability >= metadata["high_threshold"]:
        risk = "HIGH"
    elif probability >= metadata["medium_threshold"]:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Lightweight, human-readable drivers (based on patterns seen in the evaluation report).
    lead_days = (request.appointment_day - request.scheduled_day.date()).days
    factors = []
    if request.previous_appointments and request.previous_no_shows / request.previous_appointments >= 0.3:
        factors.append(f"↑ Missed {request.previous_no_shows} of {request.previous_appointments} previous appointments")
    if lead_days >= 14:
        factors.append(f"↑ Booked {lead_days} days in advance (long lead time)")
    if lead_days == 0:
        factors.append("↓ Same-day booking (rarely missed)")
    if request.age >= 60:
        factors.append("↓ Older patients miss fewer appointments")
    if request.previous_appointments == 0:
        factors.append("• No visit history: prediction relies on appointment details only")

    response = PredictionResponse(
        no_show_probability=round(probability, 4),
        risk_level=risk,
        recommended_action=config.RISK_ACTIONS[risk],
        factors=factors,
        model_version=config.MODEL_VERSION,
        cached=False,
    )
    cache[key] = response
    logger.info("appointment=%s risk=%s latency_ms=%.1f cache=miss model=%s", request.appointment_id, risk,
                (time.perf_counter() - start) * 1000, config.MODEL_VERSION)
    return response
