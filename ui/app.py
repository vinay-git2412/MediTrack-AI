"""Streamlit staff dashboard. Run: streamlit run ui/app.py"""
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src import config, service
from src.preprocessing import clean_data

st.set_page_config(page_title="MediTrack AI", page_icon="🩺", layout="wide")
RISK_COLORS = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}


@st.cache_data
def load_history() -> pd.DataFrame:
    processed = config.ROOT / "data" / "processed" / "appointments_clean.csv"
    if processed.exists():
        return pd.read_csv(processed, parse_dates=["scheduled_day", "appointment_day"])
    return clean_data(pd.read_csv(config.RAW_DATA))


history = load_history()

st.title("🩺 MediTrack AI: Appointment No-Show Assistant")
if service.model is None:
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()
st.caption(f"Model {config.MODEL_VERSION} ({service.metadata['model_name']}) · "
           f"MEDIUM ≥ {service.metadata['medium_threshold']:.2f} · HIGH ≥ {service.metadata['high_threshold']:.2f}")

predict_tab, batch_tab, history_tab = st.tabs(["Single prediction", "Batch scoring", "Historical analytics"])

# --- Single prediction -------------------------------------------------------
with predict_tab:
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        patient_id = c1.text_input("Patient ID (optional)")
        age = c1.number_input("Age", min_value=0, max_value=110, value=35)
        sms = c1.selectbox("SMS reminder received", ["Unknown", "Yes", "No"])
        booking_date = c2.date_input("Booking date", value=date.today())
        booking_time = c2.time_input("Booking time", value=time(9, 0))
        appointment_day = c2.date_input("Appointment date", value=date.today() + timedelta(days=7))
        previous_appointments = c3.number_input("Previous appointments", min_value=0, value=0)
        previous_no_shows = c3.number_input("Previous no-shows", min_value=0, value=0)
        submitted = st.form_submit_button("Predict no-show risk", type="primary")

    if submitted:
        try:
            request = service.AppointmentRequest(
                patient_id=patient_id or None, age=age,
                sms_received={"Yes": True, "No": False, "Unknown": None}[sms],
                scheduled_day=datetime.combine(booking_date, booking_time), appointment_day=appointment_day,
                previous_appointments=previous_appointments, previous_no_shows=previous_no_shows,
            )
            result = service.predict(request)
        except ValidationError as e:
            for err in e.errors():
                st.error(f"{'.'.join(map(str, err['loc'])) or 'input'}: {err['msg']}")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("No-show probability", f"{result.no_show_probability:.2f}")
            m2.markdown(f"**Risk level**\n\n### :{RISK_COLORS[result.risk_level]}[{result.risk_level}]")
            m3.markdown(f"**Recommended action**\n\n{result.recommended_action}")
            st.progress(result.no_show_probability, text=f"{result.no_show_probability:.0%} likelihood of no-show")
            if result.factors:
                st.markdown("**Factors influencing this prediction**\n\n" + "\n".join(f"- {f}" for f in result.factors))
            st.caption(f"{'⚡ Served from cache' if result.cached else 'Fresh prediction'} · "
                       f"cache hits {service.cache_stats['hits']} / misses {service.cache_stats['misses']}")

# --- Batch scoring -----------------------------------------------------------
with batch_tab:
    st.write("Upload tomorrow's appointments as CSV to get a ranked call list. "
             "Columns follow the API fields; only `age`, `scheduled_day` and `appointment_day` are required.")
    template = pd.DataFrame([{
        "patient_id": "P001", "appointment_id": "A001", "age": 34, "sms_received": True, "scheduled_day": f"{date.today() - timedelta(days=20)}T10:00:00",
        "appointment_day": str(date.today() + timedelta(days=1)), "previous_appointments": 4, "previous_no_shows": 2,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False), "appointments_template.csv")
    upload = st.file_uploader("Appointments CSV", type="csv")
    if upload:
        rows, errors = [], []
        for i, record in pd.read_csv(upload, dtype={"patient_id": str, "appointment_id": str}).iterrows():
            record = {k: v for k, v in record.items() if pd.notna(v)}
            try:
                result = service.predict(service.AppointmentRequest(**record))
                rows.append({**record, **result.model_dump(exclude={"factors", "model_version", "cached"})})
            except ValidationError as e:
                errors.append(f"Row {i + 2}: {e.errors()[0]['msg']}")
        for err in errors:
            st.warning(err)
        if rows:
            scored = pd.DataFrame(rows).sort_values("no_show_probability", ascending=False)
            high = scored[scored["risk_level"] == "HIGH"]
            k1, k2, k3 = st.columns(3)
            k1.metric("Appointments", len(scored))
            k2.metric("Expected no-shows", f"{scored['no_show_probability'].sum():.1f}")
            k3.metric("HIGH risk (call list)", len(high))
            effect = st.slider("Assumed reduction in no-shows from a call + SMS", 0, 100, 25, format="%d%%")
            st.info(f"Contacting the {len(high)} HIGH-risk patients could save about "
                    f"**{high['no_show_probability'].sum() * effect / 100:.1f}** appointments.")
            st.dataframe(scored.reindex(columns=["patient_id", "appointment_id", "appointment_day",
                                                 "no_show_probability", "risk_level", "recommended_action"]),
                         hide_index=True, width="stretch")

# --- Historical analytics ----------------------------------------------------
with history_tab:
    first, last = history["appointment_day"].min().date(), history["appointment_day"].max().date()
    period = st.date_input("Appointment period", value=(first, last), min_value=first, max_value=last)
    if len(period) == 2:
        data = history[history["appointment_day"].dt.date.between(*period)]
        k1, k2, k3 = st.columns(3)
        k1.metric("Appointments", f"{len(data):,}")
        k2.metric("No-shows", f"{data['no_show'].sum():,}")
        k3.metric("No-show rate", f"{data['no_show'].mean():.1%}")

        daily = (data.groupby(data["appointment_day"].dt.date)["no_show"].agg(["size", "sum", "mean"])
                 .rename(columns={"size": "Appointments", "sum": "No-Shows", "mean": "No-Show Rate"}))
        daily.index.name = "Date"
        st.subheader("Daily summary")
        st.line_chart(daily["No-Show Rate"], y_label="No-show rate", x_label="Appointment date")
        st.dataframe(daily.style.format({"No-Show Rate": "{:.1%}"}), width="stretch")

        def rate_by(column, label):
            return data.groupby(column, observed=True)["no_show"].mean().rename(label)

        c1, c2 = st.columns(2)
        lead = (data["appointment_day"].dt.normalize() - data["scheduled_day"].dt.normalize()).dt.days
        lead_band = pd.cut(lead, [-1, 0, 3, 7, 14, 30, 365], labels=["0 same day", "1-3", "4-7", "8-14", "15-30", "31+"])
        with c1:
            st.markdown("**No-show rate by booking lead time (days)**")
            st.bar_chart(rate_by(lead_band, "No-show rate"), horizontal=True)
        with c2:
            st.markdown("**No-show rate by weekday**")
            weekday = pd.Categorical(data["appointment_day"].dt.day_name(),
                                     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
            st.bar_chart(rate_by(weekday, "No-show rate"), horizontal=True)
        st.markdown("**Neighbourhoods with the highest no-show rate (≥ 200 appointments)**")
        by_area = data.groupby("neighbourhood")["no_show"].agg(Appointments="size", Rate="mean")
        st.dataframe(by_area[by_area["Appointments"] >= 200].sort_values("Rate", ascending=False).head(10)
                     .style.format({"Rate": "{:.1%}"}), width="stretch")
