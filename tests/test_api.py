from datetime import date, timedelta

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)
PAYLOAD = {"patient_id": "P1045", "age": 44, "sms_received": True,
           "scheduled_day": f"{date.today() - timedelta(days=10)}T12:00:00",
           "appointment_day": str(date.today() + timedelta(days=8)),
           "previous_appointments": 8, "previous_no_shows": 3}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_model_info():
    assert "high_threshold" in client.get("/model/info").json()


def test_api_prediction():
    body = client.post("/predict", json=PAYLOAD).json()
    assert 0 <= body["no_show_probability"] <= 1
    assert {"risk_level", "recommended_action", "model_version", "factors"} <= body.keys()


def test_batch_prediction():
    response = client.post("/predict/batch", json=[PAYLOAD, {**PAYLOAD, "age": 70}])
    assert response.status_code == 200 and len(response.json()) == 2


def test_invalid_api_input():
    assert client.post("/predict", json={**PAYLOAD, "age": -5}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in PAYLOAD.items() if k != "appointment_day"}).status_code == 422
