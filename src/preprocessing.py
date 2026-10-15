"""Validation, cleaning and feature engineering for appointment data.

clean_data()   -> turns the raw Kaggle export into a validated, gap-free training table.
add_features() -> derives model features; it is embedded in the sklearn Pipeline so
                  training and inference always run exactly the same transformation.
"""
import logging

import pandas as pd

from src.config import AGE_MAX, AGE_MIN

logger = logging.getLogger(__name__)

RENAME = {
    "PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "gender",
    "ScheduledDay": "scheduled_day", "AppointmentDay": "appointment_day", "Age": "age",
    "Neighbourhood": "neighbourhood", "Scholarship": "scholarship", "Hipertension": "hypertension",
    "Diabetes": "diabetes", "Alcoholism": "alcoholism", "Handcap": "handicap",
    "SMS_received": "sms_received", "No-show": "no_show",
}
FLAGS = ["scholarship", "hypertension", "diabetes", "alcoholism"]


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw appointment data. Output has no missing values."""
    df = raw.rename(columns=RENAME).copy()
    report = {"raw_rows": len(df)}

    # Duplicates: exact copies first, then repeated appointment IDs (keep first export).
    report["exact_duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    report["duplicate_appointment_ids"] = int(df["appointment_id"].duplicated().sum())
    df = df.drop_duplicates(subset="appointment_id")

    # Critical fields: rows without them cannot be trusted and are rejected.
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True, errors="coerce").dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True, errors="coerce").dt.tz_localize(None)
    df["no_show"] = df["no_show"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})
    critical = ["patient_id", "appointment_id", "scheduled_day", "appointment_day", "no_show"]
    report["missing_critical"] = int(df[critical].isna().any(axis=1).sum())
    df = df.dropna(subset=critical)

    # Appointment cannot happen before it was booked.
    bad_dates = df["appointment_day"].dt.normalize() < df["scheduled_day"].dt.normalize()
    report["appointment_before_booking"] = int(bad_dates.sum())
    df = df[~bad_dates]

    # Out-of-range ages are treated as unknown and imputed with the median.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    invalid_age = ~df["age"].between(AGE_MIN, AGE_MAX)
    report["invalid_or_missing_age"] = int(invalid_age.sum())
    df.loc[invalid_age, "age"] = df.loc[~invalid_age, "age"].median()

    # Health/benefit flags: a missing flag means "not recorded", i.e. 0. Invalid values -> 0.
    for col in FLAGS:
        df[col] = pd.to_numeric(df[col], errors="coerce").where(lambda s: s.isin([0, 1])).fillna(0).astype(int)
    df["handicap"] = pd.to_numeric(df["handicap"], errors="coerce").clip(0, 4).fillna(0).astype(int)

    # SMS status: missing is NOT the same as "not sent", so keep an explicit "unknown" level.
    df["sms_received"] = pd.to_numeric(df["sms_received"], errors="coerce").map({1: "yes", 0: "no"}).fillna("unknown")
    df["gender"] = df["gender"].fillna("unknown")
    df["neighbourhood"] = df["neighbourhood"].fillna("UNKNOWN").str.strip().str.upper()

    # Leakage-safe patient history: only outcomes from *earlier days* count.
    # Same-day appointments do not see each other's outcomes.
    daily = (df.groupby(["patient_id", "appointment_day"])["no_show"].agg(n="size", ns="sum")
             .reset_index().sort_values(["patient_id", "appointment_day"]))
    grouped = daily.groupby("patient_id")
    daily["previous_appointments"] = grouped["n"].cumsum() - daily["n"]
    daily["previous_no_shows"] = grouped["ns"].cumsum() - daily["ns"]
    df = df.merge(daily[["patient_id", "appointment_day", "previous_appointments", "previous_no_shows"]],
                  on=["patient_id", "appointment_day"], how="left")

    df["no_show"] = df["no_show"].astype(int)
    df = df.sort_values("appointment_day").reset_index(drop=True)

    report["clean_rows"] = len(df)
    report["remaining_missing_values"] = int(df.isna().sum().sum())
    logger.info("Data quality report: %s", report)
    df.attrs["quality_report"] = report
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from pre-appointment information (used inside the Pipeline)."""
    df = df.copy()
    scheduled = pd.to_datetime(df["scheduled_day"], utc=True).dt.tz_localize(None)
    appointment = pd.to_datetime(df["appointment_day"], utc=True).dt.tz_localize(None)

    df["lead_time_days"] = (appointment.dt.normalize() - scheduled.dt.normalize()).dt.days.clip(lower=0)
    df["same_day"] = (df["lead_time_days"] == 0).astype(int)
    df["scheduled_hour"] = scheduled.dt.hour
    df["appointment_dow"] = appointment.dt.dayofweek.astype(str)

    prev = df["previous_appointments"].fillna(0)
    df["history_available"] = (prev > 0).astype(int)
    # Avoid division by zero for new patients: their rate is 0 and history_available flags it.
    df["history_no_show_rate"] = (df["previous_no_shows"].fillna(0) / prev.where(prev > 0)).fillna(0)
    return df
