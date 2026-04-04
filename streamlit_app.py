from __future__ import annotations

import pandas as pd
import streamlit as st

from fuzzy_aqi.config import INPUT_COLUMNS
from fuzzy_aqi.service import build_model, predict_single


st.set_page_config(page_title="AQI Fuzzy Predictor", layout="wide")


@st.cache_resource
def load_model() -> dict[str, object]:
    return build_model("final.csv")


def bucket_color(bucket: str) -> str:
    colors = {
        "Good": "#2a9d8f",
        "Satisfactory": "#8ab17d",
        "Moderate": "#e9c46a",
        "Poor": "#f4a261",
        "Very Poor": "#e76f51",
        "Severe": "#c1121f",
    }
    return colors.get(bucket, "#264653")


st.title("AQI Fuzzy Logic Predictor")
st.caption("Wang-Mendel + Mamdani baseline using CPCB-style trapezoidal and triangular memberships")

model = load_model()

default_values = {
    "PM2.5": 80.0,
    "PM10": 140.0,
    "NO2": 60.0,
    "CO": 1.5,
    "O3": 70.0,
    "SO2": 20.0,
}
max_values = {
    "PM2.5": 500.0,
    "PM10": 600.0,
    "NO2": 400.0,
    "CO": 50.0,
    "O3": 400.0,
    "SO2": 800.0,
}

st.sidebar.header("Pollutant Inputs")
inputs = {}
for column in INPUT_COLUMNS:
    inputs[column] = st.sidebar.slider(
        label=column,
        min_value=0.0,
        max_value=max_values[column],
        value=default_values[column],
        step=0.1,
    )

result = predict_single(inputs, model)
predicted_aqi = float(result["predicted_aqi"])
predicted_bucket = str(result["predicted_bucket"])
inference_info = result["inference_info"]

left, right = st.columns([1.15, 1.0])

with left:
    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted AQI", f"{predicted_aqi:.2f}")
    c2.markdown(
        f"""
        <div style="padding:1rem;border-radius:14px;background:{bucket_color(predicted_bucket)};color:white;text-align:center;font-weight:700;">
            {predicted_bucket}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if inference_info["used_fallback"]:
        st.warning(
            f"Fallback inference used: {inference_info['fallback_reason']}. "
            f"The model relied on nearest partially matching rules instead of strong aggregated firing."
        )
    else:
        st.success(
            f"Direct Mamdani inference used. Fired rules: {inference_info['fired_rule_count']} | "
            f"Max firing strength: {inference_info['max_firing_strength']:.4f}"
        )

    st.subheader("Input Memberships")
    membership_rows = []
    for column, memberships in result["input_memberships"].items():
        for label, degree in memberships.items():
            membership_rows.append(
                {
                    "Pollutant": column,
                    "Label": label,
                    "Membership": round(float(degree), 4),
                }
            )
    st.dataframe(pd.DataFrame(membership_rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("Top Fired Rules")
    top_rules = result["top_rules"]
    if not top_rules:
        st.info("No strong rules fired for this input combination.")
    else:
        for idx, rule in enumerate(top_rules, start=1):
            antecedent_text = " AND ".join(
                f"{column} is {label}" for column, label in rule["antecedent"].items()
            )
            st.markdown(
                (
                    f"**Rule {idx}**  \n"
                    f"IF {antecedent_text} THEN AQI is **{rule['consequent']}**  \n"
                    f"Firing strength: `{rule['firing_strength']:.4f}` | "
                    f"Support: `{rule['support']}` | Confidence: `{rule['confidence']:.4f}`"
                )
            )

    st.subheader("Rule Coverage")
    st.dataframe(
        pd.DataFrame(result["rule_coverage"]),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Current Inputs")
st.dataframe(
    pd.DataFrame([inputs]),
    use_container_width=True,
    hide_index=True,
)

st.caption("Run with: streamlit run streamlit_app.py")
