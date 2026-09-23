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
    page_icon="🌤️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Design system (CSS)
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    :root{
        --bg:#0D1310; --panel:#141B17; --panel-2:#182019; --border:#26332B;
        --text:#EDF2EE; --text-dim:#94A69A; --text-faint:#5D6E63;
        --green:#4C9A6A; --amber:#D9A441; --rust:#BD5B39; --teal:#3E8F86;
    }

    [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }
    [data-testid="stHeader"] { background-color: transparent !important; }

    [data-testid="stSidebar"] {
        background-color: var(--panel) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.01em; }

    /* KPI cards (st.metric, restyled) */
    div[data-testid="stMetric"] {
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"],
    div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stMetric"] { border-left: 3px solid var(--teal); }
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"],
    div[data-testid="stColumn"]:nth-of-type(2) div[data-testid="stMetric"] { border-left: 3px solid var(--amber); }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"],
    div[data-testid="stColumn"]:nth-of-type(3) div[data-testid="stMetric"] { border-left: 3px solid var(--rust); }
    [data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700; }

    /* App header (logo + title) */
    .app-header{ display:flex; align-items:center; gap:12px; margin-bottom:4px; }
    .app-header .mark{
        width:40px; height:40px; border-radius:11px;
        background:linear-gradient(135deg, var(--green), var(--teal));
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }
    .app-header h1{ font-size:1.5rem; margin:0; }
    .app-header .tagline{ font-size:0.9rem; color:var(--text-dim); margin-top:2px; }

    /* Sidebar help box */
    .help-box{
        margin-top:18px; padding:14px; border-radius:10px;
        background:var(--panel-2); border:1px dashed var(--border);
        font-size:0.82rem; color:var(--text-dim); line-height:1.5;
    }
    .help-box b{ color:var(--text); }
    .legend-chip{ display:flex; align-items:center; gap:6px; margin-top:6px; }
    .dot{ width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }

    /* Dataframe container */
    [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:12px; overflow:hidden; }

    /* keep Plotly's own text color, don't inherit the dark theme */
    [data-testid="stPlotlyChart"] svg text { fill: #1A2620 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
      <div class="mark">
        <svg viewBox="0 0 24 24" fill="none" width="22" height="22">
          <circle cx="8.3" cy="8.3" r="2.5" fill="white"/>
          <path d="M8.3 3.2v1.5M8.3 11.5v1.5M3.2 8.3h1.5M11.5 8.3h1.5M5 5l1.1 1.1M10.5 10.5l1.1 1.1M11.6 5l-1.1 1.1M6.1 10.5L5 11.6"
                stroke="white" stroke-width="1.3" stroke-linecap="round"/>
          <path d="M7 19.3a3.4 3.4 0 0 1 .4-6.8 4.8 4.8 0 0 1 9.2-1.3 3.3 3.3 0 0 1-.4 8.1H7z" fill="white"/>
        </svg>
      </div>
      <div>
        <h1>ClimaSense</h1>
        <div class="tagline">AI-powered Short-Term Temperature Forecasting · Supporting Weather-Sensitive Decision Making</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
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

st.sidebar.markdown(
    """
    <div class="help-box">
      <b>How to read these indicators</b><br>
      MAE = average gap (°C) between forecast and actual temperature. Lower is more reliable.
      <div class="legend-chip"><span class="dot" style="background:var(--rust)"></span> Alert threshold: 3°C</div>
      <div class="legend-chip"><span class="dot" style="background:var(--teal)"></span> Frost threshold: 0°C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    line=dict(color="#4C9A6A", width=2),
    hovertemplate="Observed: %{y:.1f} °C<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=city_df["time"],
    y=city_df["prediction"],
    name="Model forecast",
    line=dict(color="#D9A441", width=2, dash="dash"),
    customdata=city_df["prediction_error"],
    hovertemplate="Predicted: %{y:.1f} °C<br>Error: %{customdata:+.1f} °C<extra></extra>",
))

fig.add_hline(
    y=3, line_dash="dot", line_color="#BD5B39", opacity=0.8,
    annotation_text="Alert threshold (3 °C)",
)
fig.add_hline(
    y=0, line_dash="dot", line_color="#3E8F86", opacity=0.8,
    annotation_text="Frost threshold (0 °C)",
)

fig.update_layout(
    title=dict(text=f"J+1 forecast vs actual – {selected_city}", font=dict(color="#1A2620")),
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#1A2620")),
    height=450,
    hovermode="x unified",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#1A2620"),
)
fig.update_xaxes(
    gridcolor="#E7E9E5", zerolinecolor="#E7E9E5",
    title_font=dict(color="#1A2620"), tickfont=dict(color="#1A2620"),
)
fig.update_yaxes(
    gridcolor="#E7E9E5", zerolinecolor="#E7E9E5",
    title_font=dict(color="#1A2620"), tickfont=dict(color="#1A2620"),
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


def _accuracy_style(value):
    if value >= 60:
        return "background-color: rgba(62,143,134,0.28); color:#c9ece6; font-weight:600;"
    elif value >= 45:
        return "background-color: rgba(217,164,65,0.28); color:#f2dca3; font-weight:600;"
    else:
        return "background-color: rgba(189,91,57,0.28); color:#f0bda3; font-weight:600;"


styled_perf = (
    city_perf.sort_values("MAE (°C)")
    .style.format(
        {"MAE (°C)": "{:.2f}", "Accuracy ±2 °C (%)": "{:.1f}", "Days ≤ 3 °C": "{:.0f}"}
    )
    .bar(subset=["MAE (°C)"], color="#4C9A6A")
    .map(_accuracy_style, subset=["Accuracy ±2 °C (%)"])
)

st.dataframe(
    styled_perf,
    width="stretch",
    hide_index=True,
)
