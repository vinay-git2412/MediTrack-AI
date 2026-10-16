# MediTrack AI: Appointment No-Show Assistant

Decision-support tool that helps clinic staff spot appointments likely to become no-shows and act on them early.

It answers **P(no-show | information known before the appointment)** with a 0–1 probability, a risk level, and a recommended staff action:

```json
{
  "no_show_probability": 0.5996,
  "risk_level": "HIGH",
  "recommended_action": "Send SMS reminder and phone the patient to confirm",
  "factors": ["↑ Missed 3 of 5 previous appointments", "↑ Booked 30 days in advance (long lead time)"],
  "model_version": "v1.1.0",
  "cached": false
}
```

The tool supports staff decisions; it never denies or reduces access to care.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python -m src.train               # clean data, compare models, tune thresholds, save model + report
uvicorn src.api:app --reload      # API docs at http://localhost:8000/docs
streamlit run ui/app.py           # staff dashboard at http://localhost:8501
pytest -q                         # 25 tests
```

A trained model is already committed in `models/`, so you can start the API or UI without retraining.

## Architecture

```text
data/raw CSV ─► clean_data() ─► leakage-safe history ─► time-based split
                 (validation,                              │
                  missing data,                            ▼
                  duplicates)            sklearn Pipeline: add_features ─► impute ─► scale / one-hot ─► model
                                                           │
                                     threshold tuning ◄────┤──► reports/evaluation.md
                                                           ▼
                                           models/model.joblib + metadata.json
                                                           │
                                  src/service.py (validate ─► TTL cache ─► predict ─► risk + action)
                                          │                                   │
                                     FastAPI (src/api.py)            Streamlit (ui/app.py)
```

| Path | Purpose |
|---|---|
| `src/config.py` | Paths, feature lists, split dates, risk actions, cache settings |
| `src/preprocessing.py` | `clean_data()` (validation & cleaning) and `add_features()` (feature engineering) |
| `src/train.py` | Model comparison, threshold tuning, evaluation, fairness check, artifacts |
| `src/service.py` | Pydantic schemas, cached prediction, risk/action mapping, explanation factors |
| `src/api.py` | FastAPI: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch` |
| `ui/app.py` | Streamlit: single prediction, batch CSV scoring, historical analytics |
| `reports/evaluation.md` | Full generated evaluation report |

`add_features()` runs as a `FunctionTransformer` inside the serialized sklearn Pipeline. Training and inference therefore always run the same feature code.

## Dataset

[Kaggle: Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) — 110,527 appointments in Vitória, Brazil, for appointment days 2016-04-29 → 2016-06-08 (booking dates go back to 2015-11).

**Data audit findings**

| Check | Result |
|---|---|
| Missing values / duplicate rows / duplicate appointment IDs | 0 / 0 / 0 |
| Target balance | 79.8% show, **20.2% no-show** |
| Age < 0 or > 110 | 6 rows (imputed with median) |
| Appointment day before booking day | 5 rows (rejected) |
| `AppointmentDay` time component | always 00:00, so the appointment hour is unknown |
| Patient history columns | none in the source; derived (see below) |

**Target:** `No-show = "Yes"` → 1 (did not attend), `"No"` → 0. The dataset contains no cancellations or reschedules. If a future export includes them, they should be excluded from training rather than counted as no-shows.

**Leakage review:** all source columns are known before the appointment. `PatientId` and `AppointmentID` are used only for joining and caching and are never model features. `Gender` is excluded from the features (see Fairness).

**Feature selection:** `Neighbourhood`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism` and `Handcap` were used in v1.0.0 but removed in v1.1.0. Together they added less signal than age alone (each cost the model under 0.004 validation PR-AUC). The health flags largely duplicate age, Handcap is 0 in 98% of rows, and these are sensitive medical or proxy attributes that staff often won't have at booking time. Neighbourhood is still cleaned and shown in the analytics view. See [Impact of removing them](#impact-of-removing-low-signal-columns-v100--v110).

## Data cleaning (`clean_data`)

Different fields get different rules; nothing is blindly filled with 0.

| Situation | Rule |
|---|---|
| Exact duplicate rows / repeated appointment IDs | Drop (keep first) |
| Missing patient ID, appointment ID, dates or target | Reject row |
| Appointment before booking | Reject row |
| Age missing or outside 0–110 | Treated as unknown → median |
| Health/benefit flags missing or invalid | 0 (= condition not recorded) |
| SMS status missing | Explicit `"unknown"` category (not "no") |
| Neighbourhood missing (analytics only) | `"UNKNOWN"` |

The output has no missing values (verified by a test). A data quality report with counts is logged and saved to `models/metadata.json`.

## Feature engineering

| Group | Features |
|---|---|
| Scheduling | `lead_time_days`, `same_day`, `scheduled_hour` (booking hour), `appointment_dow` |
| Patient history | `previous_appointments`, `previous_no_shows`, `history_no_show_rate`, `history_available` |
| Context | `age`, `sms_received` |

**Leakage-safe history:** a patient's history counts only appointments on *earlier days*. Same-day appointments never see each other's outcomes.

**New patients:** history counts are 0 and `history_available = 0`. The no-show rate is set to 0 instead of dividing by zero.

**Unknown categories:** the encoder uses `OneHotEncoder(handle_unknown="ignore")`, so an unseen category (e.g. a Sunday appointment, never seen in training) doesn't crash scoring.

## Models and evaluation

The split is **time-based**: train on appointments before 2016-05-24 (67,360), validate on 2016-05-24 to 05-31 (16,711), and test on 2016-06-01 onwards (26,451). This mirrors production, where the model is trained on the past and predicts the future.

**Validation comparison**

| Model | Accuracy@0.5 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| Dummy (always "show") | 0.807 | 0.500 | 0.193 | 0.156 |
| **Logistic Regression** | 0.805 | 0.732 | **0.353** | 0.140 |
| Random Forest | 0.806 | 0.737 | 0.353 | 0.140 |
| Hist Gradient Boosting | 0.800 | 0.733 | 0.340 | 0.142 |

**Logistic Regression** is selected because it has the highest validation PR-AUC (the selection metric, since it focuses on the minority class). The tree models score no better on PR-AUC, so the simpler, faster and more explainable model wins. `train.py` re-runs this selection automatically on every retrain.

**Test-set results (untouched period)**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.722 |
| PR-AUC | 0.330 (vs 0.185 base rate) |
| Brier score | 0.138 (vs 0.151 for the always-show baseline) |

| Decision threshold | Accuracy | Precision | Recall | F1 | % flagged |
|---|---:|---:|---:|---:|---:|
| Default 0.50 | 0.811 | 0.418 | 0.059 | 0.104 | 3% |
| **HIGH risk ≥ 0.35** (phone call) | **0.763** | 0.339 | 0.299 | 0.318 | 16% |
| MEDIUM risk ≥ 0.24 (SMS) | 0.610 | 0.290 | 0.768 | 0.421 | 49% |

✅ **Accuracy requirement:** the model's no-show/show prediction (HIGH threshold) reaches **76.3% accuracy on the test set** (asserted in `tests/test_service.py`).

### Why accuracy alone is misleading here
Only ~18–20% of appointments are no-shows. A model that predicts "everyone shows up" scores **81.5%** accuracy on the test set while catching **zero** no-shows. At the default 0.5 threshold our model is similar: 81% accuracy, but it finds only 6% of no-shows. That's why model selection uses PR-AUC, and why the thresholds are tuned for F1 and recall instead.

### Threshold choice
Both thresholds are tuned on the validation set and stored in `metadata.json`, so they version with the model:
- **MEDIUM (0.24)** maximizes F1 and catches ~77% of no-shows. It suits a cheap action like an automated SMS, where false alarms cost little.
- **HIGH (0.35)** maximizes F1 subject to accuracy ≥ 75%. It flags ~16% of appointments for staff phone calls, a manageable workload that still catches ~30% of no-shows.
- The **staff actions** for each level are configurable in `src/config.py`.

### Calibration
Calibration on test (quantile bins) is close in the low and mid range: predicted 0.22 → observed 0.19. In the top bin the model over-estimates (predicted 0.46 → observed 0.35). Part of the cause is drift: the no-show rate fell from 21.1% (train) to 18.5% (test). Isotonic calibration on the validation set improved the Brier score only marginally (0.1366 → 0.1354 in experiments) and compressed the score range, so it was not applied. Treat the probability as a well-ranked risk score that is approximately calibrated.

### What drives predictions (permutation importance, validation PR-AUC)
1. Booking date / lead time (by far the strongest: same-day bookings are rarely missed; long lead times are missed much more often)
2. Previous no-shows
3. Age (younger adults and children miss more)
4. Previous appointments, SMS: small effects

The UI and API show these as plain-language **factors** next to each prediction.

### Impact of removing low-signal columns (v1.0.0 → v1.1.0)

| Test metric | v1.0.0 (12 inputs) | v1.1.0 (6 inputs) |
|---|---:|---:|
| ROC-AUC | 0.726 | 0.722 |
| PR-AUC | 0.337 | 0.330 |
| Accuracy at HIGH threshold | 76.1% | 76.3% |
| Recall at HIGH threshold | 33.3% | 29.9% |
| No-shows captured by HIGH flag | 1,628 | 1,458 |
| Appointments flagged HIGH | 4,690 | 4,300 |

Ranking quality and accuracy barely moved. The real cost is at the HIGH tier: with slightly less confident scores, the same 0.35 threshold flags ~390 fewer appointments and catches ~170 fewer no-shows (about 10% fewer). We accept that in exchange for a model that uses no medical conditions or location, and a much shorter staff form.

### Fairness check (test set, HIGH threshold)
- **Gender:** recall is similar for women and men (0.30 vs 0.31), and gender is not used as a feature.
- **Age band:** recall varies (0–17: 0.54, 65+: 0.06). This mirrors genuinely different base rates (20% vs 15%), but older patients who do miss appointments are under-flagged. Staff shouldn't treat "LOW" as a guarantee for that group.

See `reports/evaluation.md` for the full tables.

### Business impact (test period simulation)
Over 26,451 appointments with 4,882 no-shows, the HIGH flag selects 4,300 appointments and captures 1,458 actual no-shows. If a call + SMS prevented 25% of those, about **364 appointment slots** would be recovered over 6 working days. The batch tab in the UI includes this simulator with an adjustable effectiveness slider.

## API usage

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "patient_id": "P1045", "age": 44, "sms_received": true,
  "scheduled_day": "2026-10-02T12:00:00", "appointment_day": "2026-10-20",
  "previous_appointments": 8, "previous_no_shows": 3
}'
```

Only `age`, `scheduled_day` and `appointment_day` are required. Optional fields default to safe values: `sms_received` defaults to unknown and history counts default to 0.

| Situation | Response |
|---|---|
| Valid input | 200 with probability in [0, 1] (enforced by the response schema) |
| Age outside 0–110, negative counts, no-shows > appointments, appointment before booking, **appointment already in the past**, missing required field | 422 |
| Model artifact missing | 503 |
| Unknown SMS status, unseen weekday, extra/legacy fields (e.g. `neighbourhood`) | Scored normally (extra fields are ignored) |

`POST /predict/batch` takes a list of the same objects. `GET /health` reports cache hits and misses.

## UI usage (`streamlit run ui/app.py`)
- **Single prediction:** patient form → probability bar, coloured risk level, recommended action, factors, and a cache indicator.
- **Batch scoring:** upload a CSV of tomorrow's appointments (a template download is provided) → ranked call list, expected no-shows, and an intervention simulator.
- **Historical analytics:** a date-range filter, then a daily **summary table (date, appointments, no-shows, no-show rate)**, a trend line, rates by lead time and weekday, and the neighbourhoods with the highest no-show rates.

## Caching
`cachetools.TTLCache` (1,024 entries, 1-hour TTL) sits in the prediction service, so the API and UI both use it.
- **The cache key is every input field plus the model version, not just `patient_id`.** Changing `sms_received` from false to true gives a fresh prediction, and a retrained model never serves stale results.
- **A repeated request returns in about 5 ms (well under 1 s), marked `"cached": true`.** Tests verify both behaviours.
- **Production:** move this to Redis or ElastiCache so the cache is shared across API replicas.

## Assumptions
- Every appointment in the data is a real attendance or a real no-show (no cancellations or reschedules are present).
- Missing health flags mean "not recorded" (0) (cleaning only; they are not model inputs). Missing SMS status means unknown, not "not sent".
- At prediction time, staff (or the scheduling system) supply the patient's previous appointment and no-show counts.
- Online scoring is only for upcoming appointments; past dates are rejected.

## Trade-offs
- **Logistic Regression over boosting:** equal PR-AUC, and it is simpler, faster and easier to explain and audit.
- **Two thresholds instead of one:** the 75% accuracy requirement and high recall pull in opposite directions. Pairing a cheap SMS tier with a precise call tier satisfies both.
- **No class weighting or oversampling:** this keeps probabilities interpretable. The imbalance is handled through thresholds instead.
- **No calibration layer:** the gain was marginal and it would have reused the threshold-tuning data.
- **The final model is trained on the train period only:** this keeps the validation and test sets honest. Refitting on train + validation before deployment is a reasonable next step.
- **In-memory cache and a direct service call from the UI:** simplest to run. Production would use Redis and route the UI through the API.

## Limitations
- **Modest signal:** ROC-AUC is about 0.72. Roughly 1 in 3 HIGH-risk flags is an actual no-show. The dataset lacks the strongest real-world predictors (appointment hour, specialty, doctor, distance, weather, long patient history).
- **Short history window:** the data covers only six weeks of appointments, so most patients look "new". In production, a full history should improve the model considerably.
- **SMS looks counter-intuitive:** in this data SMS is only ever sent for bookings made 3 or more days ahead, so it is confounded with lead time (SMS group: 27.6% no-show, no-SMS group: 16.7%). Its coefficient must not be read as "SMS causes no-shows".
- **Single region, 2016:** a new clinic population needs re-validation before use, and drift is already visible (21% → 18.5% no-show rate).
- **Under-flagging of patients aged 65+:** see the fairness check.

## Edge cases handled
New patients (zero history, no division by zero) · unseen categories · unknown SMS status · invalid or extreme age · negative counts · no-shows exceeding appointments · appointment before booking · past appointment for online scoring · missing critical fields · duplicates in training data · repeated requests (cache) · model artifact missing (503).

## Production roadmap (designed, not built)
- **Redis** cache; **PostgreSQL** to store predictions together with actual outcomes (the feedback loop) for evaluation and retraining.
- A nightly **batch job** that scores the next day's appointments and pushes the call list to staff.
- **Monitoring:**
  - Application: latency, error rate, cache hit rate.
  - ML: prediction distribution, feature drift, observed no-show rate, precision and recall once outcomes arrive.
- A **model registry** with version metadata (already emitted to `metadata.json`), plus scheduled retraining with a validation gate.
- **SHAP** explanations if a tree model is adopted; authentication/RBAC; CI/CD running `pytest` and `python -m src.train`.

## AI usage
AI assistance was used for parts of this project:
- **README** (`README.md`): writing and structuring the documentation.
- **Evaluation report** (`reports/evaluation.md`): writing the report, including the report template in step 10 of `src/train.py` that generates it. The rest of `src/train.py` (training, model selection, threshold tuning) is not AI-written.
- **Streamlit UI** (`ui/app.py`): building the staff dashboard.
- **Test cases** (`tests/`): writing the pytest suite.

All AI-assisted output was reviewed before it was included.
