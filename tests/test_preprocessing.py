import numpy as np
import pandas as pd
import pytest

from src.preprocessing import add_features, clean_data


@pytest.fixture
def raw():
    row = {"PatientId": 1, "AppointmentID": 10, "Gender": "F", "ScheduledDay": "2016-04-01T10:00:00Z",
           "AppointmentDay": "2016-04-05T00:00:00Z", "Age": 30, "Neighbourhood": "CENTRO", "Scholarship": 0,
           "Hipertension": 0, "Diabetes": 0, "Alcoholism": 0, "Handcap": 0, "SMS_received": 1, "No-show": "No"}
    rows = [dict(row, AppointmentID=10 + i) for i in range(6)]
    rows[1].update(Age=-5, SMS_received=np.nan, Neighbourhood=None, Hipertension=np.nan)  # bad/missing values
    rows[2].update(AppointmentDay="2016-03-01T00:00:00Z")                                # before booking
    rows[3].update(ScheduledDay=None)                                                    # missing critical
    rows[4].update(AppointmentDay="2016-04-10T00:00:00Z", **{"No-show": "Yes"})          # later visit, no-show
    rows[5].update(AppointmentDay="2016-04-20T00:00:00Z")                                # third visit
    return pd.DataFrame(rows + [rows[0]])                                                # + exact duplicate


def test_missing_values_removed(raw):
    assert clean_data(raw).isna().sum().sum() == 0


def test_duplicate_handling(raw):
    clean = clean_data(raw)
    assert clean["appointment_id"].is_unique
    assert clean.attrs["quality_report"]["exact_duplicates"] == 1


def test_invalid_rows_rejected(raw):
    clean = clean_data(raw)
    assert 12 not in clean["appointment_id"].values  # appointment before booking
    assert 13 not in clean["appointment_id"].values  # missing scheduled_day


def test_invalid_age_imputed_and_missing_sms_is_unknown(raw):
    row = clean_data(raw).set_index("appointment_id").loc[11]
    assert 0 <= row["age"] <= 110
    assert row["sms_received"] == "unknown"
    assert row["neighbourhood"] == "UNKNOWN"
    assert row["hypertension"] == 0


def test_history_uses_only_earlier_days(raw):
    clean = clean_data(raw).set_index("appointment_id")
    # Same-day appointments (10, 11) see no history; 14 sees both; 15 also sees the no-show on 14.
    assert clean.loc[10, ["previous_appointments", "previous_no_shows"]].tolist() == [0, 0]
    assert clean.loc[14, ["previous_appointments", "previous_no_shows"]].tolist() == [2, 0]
    assert clean.loc[15, ["previous_appointments", "previous_no_shows"]].tolist() == [3, 1]


def test_features_for_new_patient_avoid_division_by_zero():
    df = pd.DataFrame([{"scheduled_day": "2026-10-01 09:00", "appointment_day": "2026-10-01",
                        "previous_appointments": 0, "previous_no_shows": 0}])
    out = add_features(df).iloc[0]
    assert out["history_no_show_rate"] == 0 and out["history_available"] == 0
    assert out["same_day"] == 1 and out["lead_time_days"] == 0
