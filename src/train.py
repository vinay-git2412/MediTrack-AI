"""Train, compare and evaluate no-show models, then save the chosen pipeline.

Run: python -m src.train
"""
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.preprocessing import add_features, clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def metrics_at(y, proba, threshold):
    pred = proba >= threshold
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "flagged_share": pred.mean(),
    }


# 1. Load, validate and clean
df = clean_data(pd.read_csv(config.RAW_DATA))
quality_report = df.attrs["quality_report"]
processed_path = config.ROOT / "data" / "processed" / "appointments_clean.csv"
processed_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_path, index=False)

# 2. Time-based split: past -> train, recent past -> validation, latest -> test
train = df[df["appointment_day"] < config.TRAIN_END]
valid = df[(df["appointment_day"] >= config.TRAIN_END) & (df["appointment_day"] < config.VALID_END)]
test = df[df["appointment_day"] >= config.VALID_END]
X_train, y_train = train[config.INPUT_COLUMNS], train["no_show"]
X_valid, y_valid = valid[config.INPUT_COLUMNS], valid["no_show"]
X_test, y_test = test[config.INPUT_COLUMNS], test["no_show"]
logger.info("Split sizes: train=%d valid=%d test=%d", len(train), len(valid), len(test))

# 3. One shared preprocessing pipeline: features -> impute -> scale / one-hot
preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), config.NUMERIC_FEATURES),
    (make_pipeline(SimpleImputer(strategy="constant", fill_value="unknown"),
                   OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
     config.CATEGORICAL_FEATURES),
)
candidates = {
    "Dummy (always show)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42),
}

# 4. Train and compare on the validation period
comparison, fitted = [], {}
for name, estimator in candidates.items():
    pipeline = make_pipeline(FunctionTransformer(add_features), preprocessor, estimator)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_valid)[:, 1]
    fitted[name] = pipeline
    comparison.append({
        "model": name,
        "accuracy@0.5": accuracy_score(y_valid, proba >= 0.5),
        "roc_auc": roc_auc_score(y_valid, proba),
        "pr_auc": average_precision_score(y_valid, proba),
        "brier": brier_score_loss(y_valid, proba),
    })
    logger.info("%s: %s", name, comparison[-1])
comparison = pd.DataFrame(comparison).set_index("model")

# PR-AUC is the selection metric: it focuses on the minority (no-show) class.
best_name = comparison.drop("Dummy (always show)")["pr_auc"].idxmax()
model = fitted[best_name]
logger.info("Selected model: %s", best_name)

# 5. Threshold tuning on validation
valid_proba = model.predict_proba(X_valid)[:, 1]
sweep = pd.DataFrame([metrics_at(y_valid, valid_proba, t) for t in np.arange(0.05, 0.80, 0.01)])
medium_threshold = sweep.loc[sweep["f1"].idxmax(), "threshold"]  # recall-oriented: cheap SMS
high_ok = sweep[sweep["accuracy"] >= config.MIN_ACCURACY]       # precision-oriented: staff calls
high_threshold = high_ok.loc[high_ok["f1"].idxmax(), "threshold"]
logger.info("Thresholds: medium=%.2f high=%.2f", medium_threshold, high_threshold)

# 6. Final evaluation on the untouched test period
test_proba = model.predict_proba(X_test)[:, 1]
test_metrics = {
    "roc_auc": roc_auc_score(y_test, test_proba),
    "pr_auc": average_precision_score(y_test, test_proba),
    "brier": brier_score_loss(y_test, test_proba),
    "baseline_accuracy_always_show": accuracy_score(y_test, np.zeros(len(y_test))),
}
threshold_table = pd.DataFrame(
    [metrics_at(y_test, test_proba, t) for t in (0.5, high_threshold, medium_threshold)],
    index=["Default 0.5", "HIGH risk (call)", "MEDIUM risk (SMS)"],
)
cm = confusion_matrix(y_test, test_proba >= high_threshold)
frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=8, strategy="quantile")
calibration = pd.DataFrame({"mean_predicted": mean_pred, "observed_no_show_rate": frac_pos})

# 7. Fairness check: performance by subgroup at the HIGH threshold
groups = test.assign(age_band=pd.cut(test["age"], [-1, 17, 39, 64, 120], labels=["0-17", "18-39", "40-64", "65+"]),
                     proba=test_proba, flagged=test_proba >= high_threshold)
fairness = pd.concat([
    groups.groupby(col, observed=True).apply(lambda g: pd.Series({
        "appointments": len(g),
        "no_show_rate": g["no_show"].mean(),
        "mean_predicted": g["proba"].mean(),
        "recall": recall_score(g["no_show"], g["flagged"], zero_division=0),
        "precision": precision_score(g["no_show"], g["flagged"], zero_division=0),
    })).rename(index=lambda v: f"{col}={v}")
    for col in ("gender", "age_band")
])

# 8. Which inputs matter most (permutation importance on raw inputs)
importance = permutation_importance(model, X_valid, y_valid, scoring="average_precision",
                                    n_repeats=3, random_state=42, n_jobs=-1)
importance = pd.Series(importance.importances_mean, index=config.INPUT_COLUMNS).sort_values(ascending=False)

# 9. Business impact on the test period (one row = one appointment)
high = test_proba >= high_threshold
impact = {
    "appointments": len(test),
    "actual_no_shows": int(y_test.sum()),
    "high_risk_flagged": int(high.sum()),
    "no_shows_captured_by_high_flag": int(y_test[high].sum()),
    "no_shows_saved_if_calls_cut_no_shows_by_25pct": round(0.25 * y_test[high].sum()),
}

# 10. Persist model, metadata and report
joblib.dump(model, config.MODEL_PATH)
metadata = {
    "model_version": config.MODEL_VERSION,
    "model_name": best_name,
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "training_data_range": [str(train["appointment_day"].min().date()), str(train["appointment_day"].max().date())],
    "test_data_range": [str(test["appointment_day"].min().date()), str(test["appointment_day"].max().date())],
    "medium_threshold": float(medium_threshold),
    "high_threshold": float(high_threshold),
    "test_accuracy_at_high_threshold": float(threshold_table.loc["HIGH risk (call)", "accuracy"]),
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "data_quality": quality_report,
    "input_columns": config.INPUT_COLUMNS,
}
config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

config.REPORT_PATH.write_text(f"""# Model Evaluation Report ({config.MODEL_VERSION})

Generated by `python -m src.train` on {metadata['trained_at']}.

## Data quality
{pd.Series(quality_report).to_markdown()}

## Split (time-based on appointment day)
| Set | Rows | No-show rate | Period |
|---|---:|---:|---|
| Train | {len(train)} | {y_train.mean():.3f} | < {config.TRAIN_END} |
| Validation | {len(valid)} | {y_valid.mean():.3f} | {config.TRAIN_END} to {config.VALID_END} |
| Test | {len(test)} | {y_test.mean():.3f} | >= {config.VALID_END} |

## Model comparison (validation set)
{comparison.round(4).to_markdown()}

Selected: **{best_name}** (highest validation PR-AUC).

## Test-set performance
{pd.Series(test_metrics).round(4).to_markdown()}

### At each decision threshold
{threshold_table.round(3).to_markdown()}

### Confusion matrix at HIGH threshold ({high_threshold:.2f})
| | Predicted show | Predicted no-show |
|---|---:|---:|
| Actual show | {cm[0, 0]} | {cm[0, 1]} |
| Actual no-show | {cm[1, 0]} | {cm[1, 1]} |

### Calibration (quantile bins)
{calibration.round(3).to_markdown(index=False)}

## Permutation importance (drop in validation PR-AUC)
{importance.round(4).to_markdown()}

## Subgroup check (test set, HIGH threshold)
{fairness.round(3).to_markdown()}

## Business impact (test period)
{pd.Series(impact).to_markdown()}
""")
logger.info("Saved model to %s and report to %s", config.MODEL_PATH, config.REPORT_PATH)
