"""Central configuration: paths, feature lists, split dates and business rules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
REPORT_PATH = ROOT / "reports" / "evaluation.md"

MODEL_VERSION = "v1.1.0"

# Time-based split on appointment_day: train on the past, predict the future.
TRAIN_END = "2016-05-24"  # train: appointment_day <  TRAIN_END
VALID_END = "2016-06-01"  # valid: TRAIN_END <= day < VALID_END, test: day >= VALID_END

AGE_MIN, AGE_MAX = 0, 110

# Columns the model pipeline receives (raw, pre-appointment information only).
# Neighbourhood, scholarship and the health flags are deliberately excluded: they added
# almost no signal (see README) and are sensitive / proxy attributes. Neighbourhood is
# still kept in the cleaned data for the analytics view.
INPUT_COLUMNS = [
    "age", "sms_received", "scheduled_day", "appointment_day",
    "previous_appointments", "previous_no_shows",
]

# Columns the estimator sees after feature engineering.
NUMERIC_FEATURES = [
    "age", "lead_time_days", "same_day", "scheduled_hour",
    "previous_appointments", "previous_no_shows", "history_no_show_rate", "history_available",
]
CATEGORICAL_FEATURES = ["sms_received", "appointment_dow"]

# Business risk mapping (probability -> level -> staff action).
# The MEDIUM/HIGH probability cut-offs are tuned on the validation set by train.py
# and stored in metadata.json, so they move with the model version.
MIN_ACCURACY = 0.75  # HIGH-risk threshold must keep overall accuracy at or above this
RISK_ACTIONS = {
    "LOW": "Standard reminder",
    "MEDIUM": "Send SMS reminder 24h before the appointment",
    "HIGH": "Send SMS reminder and phone the patient to confirm",
}

CACHE_MAX_SIZE = 1024
CACHE_TTL_SECONDS = 3600
