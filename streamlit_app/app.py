import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="ClimaSense – Operational Temperature Forecasting",
    page_icon="🌡️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("ClimaSense")
st.markdown(
    """
### AI-powered Short-Term Temperature Forecasting

Supporting Weather-Sensitive Decision Making
"""
)

# ──────────────────────────────────────────────
# Load prediction data
# ──────────────────────────────────────────────

@st.cache_data
def load_predictions():
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "predictions" / "weather_predictions_2022_J1.csv"

    try:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])

        required_columns = [
            "time", "city", "target_temp_max_J1", "prediction",
        ]
        missing = set(required_columns) - set(df.columns)

        if missing:
            st.error(f"Missing columns in predictions file: {missing}")
            st.stop()

        df = df.dropna(subset=required_columns)
        return df

    except FileNotFoundError:
        st.error("Predictions file not found.")
        st.info("Please run notebook 03_model.ipynb first.")
        st.stop()


predictions_df = load_predictions()

# ──────────────────────────────────────────────
# Sidebar – Parameters
# ──────────────────────────────────────────────

st.sidebar.header("Parameters")

cities = sorted(predictions_df["city"].unique())

selected_city = st.sidebar.selectbox(
    "Select a city",
    cities,
    index=cities.index("Paris") if "Paris" in cities else 0,
)

city_df = predictions_df[predictions_df["city"] == selected_city].copy()

# Date filter
st.sidebar.subheader("Analysis period")

date_min = city_df["time"].min().date()
date_max = city_df["time"].max().date()

use_date_filter = st.sidebar.checkbox("Filter by date range", value=False)

if use_date_filter:
    date_range = st.sidebar.date_input(
        "Select period",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    if len(date_range) == 2:
        mask = (city_df["time"].dt.date >= date_range[0]) & (
            city_df["time"].dt.date <= date_range[1]
        )
        city_df = city_df.loc[mask].copy()

        if city_df.empty:
            st.warning("No data available for the selected period.")
            st.stop()

# ──────────────────────────────────────────────
# Forecast indicators
# ──────────────────────────────────────────────

st.header(f"Forecast Performance – {selected_city}")

city_df["error"] = city_df["prediction"] - city_df["target_temp_max_J1"]
city_df["absolute_error"] = city_df["error"].abs()

mae_city = city_df["absolute_error"].mean()
precision_2c = (city_df["absolute_error"] <= 2).mean() * 100
risk_days = (city_df["target_temp_max_J1"] <= 3).sum()

national_mae = (
    predictions_df.groupby("city")
    .apply(
        lambda x: (x["prediction"] - x["target_temp_max_J1"]).abs().mean(),
        include_groups=False,
    )
    .mean()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mae_city:.2f} °C",
        delta=f"{mae_city - national_mae:+.2f} vs 20-city avg",
        delta_color="inverse",
    )

with col2:
    st.metric("Forecast accuracy (±2 °C)", f"{precision_2c:.1f} %")

with col3:
    st.metric("Days at frost risk (≤ 3 °C)", f"{risk_days}")

# ──────────────────────────────────────────────
# Forecast vs actuals (interactive)
# ──────────────────────────────────────────────

st.header("Forecast vs actual temperature")

city_df["prediction_error"] = (city_df["prediction"] - city_df["target_temp_max_J1"]).round(1)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=city_df["time"],
    y=city_df["target_temp_max_J1"],
    name="Actual temperature",
    line=dict(color="#2563EB", width=2),
    hovertemplate="Observed: %{y:.1f} °C<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=city_df["time"],
    y=city_df["prediction"],
    name="Model forecast",
    line=dict(color="#F97316", width=2, dash="dash"),
    customdata=city_df["prediction_error"],
    hovertemplate="Predicted: %{y:.1f} °C<br>Error: %{customdata:+.1f} °C<extra></extra>",
))

fig.add_hline(
    y=3, line_dash="dot", line_color="#A855F7", opacity=0.7,
    annotation_text="Alert threshold (3 °C)",
)
fig.add_hline(
    y=0, line_dash="dot", line_color="#3B82F6", opacity=0.7,
    annotation_text="Frost threshold (0 °C)",
)

fig.update_layout(
    title=f"J+1 forecast vs actual – {selected_city}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=450,
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────
# City performance comparison
# ──────────────────────────────────────────────

st.header("Model Performance across cities")

city_perf = (
    predictions_df.groupby("city")
    .apply(
        lambda x: pd.Series(
            {
                "MAE (°C)": (x["prediction"] - x["target_temp_max_J1"])
                .abs()
                .mean(),
                "Accuracy ±2 °C (%)": (
                    (x["prediction"] - x["target_temp_max_J1"]).abs() <= 2
                ).mean()
                * 100,
                "Days ≤ 3 °C": (x["target_temp_max_J1"] <= 3).sum(),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

st.dataframe(
    city_perf.sort_values("MAE (°C)").style.format(
        {"MAE (°C)": "{:.2f}", "Accuracy ±2 °C (%)": "{:.1f}", "Days ≤ 3 °C": "{:.0f}"}
    ),
    width="stretch",  hide_index=True,
)