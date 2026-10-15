import time
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.metrics import accuracy_score

from src import config, service
from src.preprocessing import clean_data
from src.service import AppointmentRequest

TODAY = date.today()
NEXT_SUNDAY = TODAY + timedelta(days=(6 - TODAY.weekday()) % 7 or 7)  # weekday never seen in training


def make_request(**overrides):
    fields = dict(age=35, sms_received=True,
                  scheduled_day=datetime.combine(TODAY - timedelta(days=5), datetime.min.time()),
                  appointment_day=TODAY + timedelta(days=2), previous_appointments=2, previous_no_shows=0)
    return AppointmentRequest(**{**fields, **overrides})


def test_model_loads():
    assert service.model is not None
    assert {"medium_threshold", "high_threshold"} <= service.metadata.keys()


def test_model_accuracy_on_test_set_at_least_75_percent():
    df = clean_data(pd.read_csv(config.RAW_DATA))
    test = df[df["appointment_day"] >= config.VALID_END]
    proba = service.model.predict_proba(test[config.INPUT_COLUMNS])[:, 1]
    assert accuracy_score(test["no_show"], proba >= service.metadata["high_threshold"]) >= 0.75


@pytest.mark.parametrize("overrides", [{}, {"appointment_day": NEXT_SUNDAY}, {"sms_received": None},
                                       {"previous_appointments": 0, "previous_no_shows": 0}])
def test_probability_between_0_and_1_incl_unknown_category_and_new_patient(overrides):
    result = service.predict(make_request(**overrides))
    assert 0 <= result.no_show_probability <= 1
    assert result.risk_level in config.RISK_ACTIONS


@pytest.mark.parametrize("overrides", [
    {"age": -5}, {"age": 300}, {"previous_appointments": -1},
    {"previous_appointments": 1, "previous_no_shows": 3},
    {"appointment_day": TODAY - timedelta(days=1)},
])
def test_invalid_input_rejected(overrides):
    with pytest.raises(ValidationError):
        make_request(**overrides)


def test_cache_returns_same_prediction_within_1_second():
    request = make_request(patient_id="CACHE-1", age=41)
    first = service.predict(request)
    start = time.perf_counter()
    second = service.predict(request)
    assert time.perf_counter() - start < 1
    assert second.cached and not first.cached
    assert second.no_show_probability == first.no_show_probability


def test_cache_invalidates_changed_input():
    service.predict(make_request(patient_id="CACHE-2", sms_received=False))
    assert service.predict(make_request(patient_id="CACHE-2", sms_received=True)).cached is False


def test_golden_cases_risk_ordering():
    # Low risk: same-day booking, reliable history, older patient.
    low = service.predict(make_request(age=70, scheduled_day=datetime.combine(TODAY, datetime.min.time()),
                                       appointment_day=TODAY, previous_appointments=6, previous_no_shows=0))
    # High risk: long lead time and frequent previous no-shows.
    high = service.predict(make_request(age=25, sms_received=False, scheduled_day=datetime(2026, 1, 1),
                                        appointment_day=TODAY + timedelta(days=30),
                                        previous_appointments=5, previous_no_shows=4))
    assert low.risk_level == "LOW"
    assert high.risk_level == "HIGH"
    assert high.no_show_probability > low.no_show_probability
