#!/usr/bin/env python
# coding: utf-8

# # 🌤️ ClimaSense – J+1 Temperature Forecast Dashboard

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

st.set_page_config(
    page_title="ClimaSense - J+1 Temperature Forecast",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ ClimaSense - J+1 Temperature Forecast")


# ## Load prediction data

# In[ ]:


@st.cache_data
def load_predictions():
    project_root = Path(__file__).resolve().parent.parent
    path = project_root /"notebooks" / "data" / "predictions" / "weather_predictions_2022_J1.csv"

    try:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])

        required_columns = [
            "time", "city", "target_temp_max_J1", "prediction"
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
        
predictions_df=load_predictions()

# ## City selection

# In[ ]:


st.sidebar.header("Parameters")

cities = sorted(predictions_df["city"].unique())

selected_city = st.sidebar.selectbox(
    "Select a city",
    cities,
    index=cities.index("Paris") if "Paris" in cities else 0
)

city_df = predictions_df[predictions_df["city"] == selected_city].copy()


# ## Analysis period

# In[ ]:


st.sidebar.subheader("Analysis period")

date_min = city_df["time"].min().date()
date_max = city_df["time"].max().date()

use_date_filter = st.sidebar.checkbox(
    "Filter by date range",
    value=False
)

if use_date_filter:
    date_range = st.sidebar.date_input(
        "Select period",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )

    if len(date_range) == 2:
        mask = (
            (city_df["time"].dt.date >= date_range[0]) &
            (city_df["time"].dt.date <= date_range[1])
        )
        city_df = city_df.loc[mask].copy()

        if city_df.empty:
            st.warning("No data available for the selected period.")
            st.stop()


# ## Key Decision Indicators

# In[ ]:


st.header("Key Decision Indicators")

city_df["error"] = city_df["prediction"] - city_df["target_temp_max_J1"]
city_df["absolute_error"] = city_df["error"].abs()

mae_city = city_df["absolute_error"].mean()
precision_2c = (city_df["absolute_error"] <= 2).mean() * 100
risk_days = (city_df["target_temp_max_J1"] <= 3).sum()

national_mae = predictions_df.groupby("city").apply(
    lambda x: (x["prediction"] - x["target_temp_max_J1"]).abs().mean()
).mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean Absolute Error (°C)",
        f"{mae_city:.2f}",
        delta=f"{mae_city - national_mae:+.2f} vs national",
        delta_color="inverse"
    )

with col2:
    st.metric("Accuracy ±2°C", f"{precision_2c:.1f}%")

with col3:
    st.metric("Days at Frost Risk (≤3°C)", f"{risk_days}")


# ## Decision Support – Tomorrow (J+1)

# In[ ]:


st.header("Decision Support – Tomorrow")

last_day = city_df.iloc[-1]
tomorrow = last_day["time"] + pd.Timedelta(days=1)
predicted_temp = last_day["prediction"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Date", tomorrow.strftime("%d/%m/%Y"))

with col2:
    st.metric("Forecasted Max Temperature", f"{predicted_temp:.1f}°C")

with col3:
    if predicted_temp <= 0:
        st.error("Frost expected – Salting required")
    elif predicted_temp <= 3:
        st.warning("Preventive salting recommended")
    elif predicted_temp <= 5:
        st.info("Increased monitoring advised")
    else:
        st.success("No action required")


# ## Prediction History

# In[ ]:


st.header("Prediction History")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    city_df["time"],
    city_df["target_temp_max_J1"],
    label="Actual temperature",
    linewidth=2
)

ax.plot(
    city_df["time"],
    city_df["prediction"],
    label="Model prediction",
    linestyle="--",
    linewidth=2
)

ax.axhline(3, color="purple", linestyle=":", label="Salting threshold (3°C)")
ax.axhline(0, color="blue", linestyle=":", label="Frost threshold (0°C)")

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"J+1 Forecast vs Actual – {selected_city}")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close()


# ## City Performance Overview

# In[ ]:


st.header("City Performance Overview")

city_perf = predictions_df.groupby("city").apply(
    lambda x: pd.Series({
        "MAE (°C)": (x["prediction"] - x["target_temp_max_J1"]).abs().mean(),
        "Accuracy ±2°C (%)": ((x["prediction"] - x["target_temp_max_J1"]).abs() <= 2).mean() * 100,
        "Days ≤ 3°C": (x["target_temp_max_J1"] <= 3).sum()
    })
).reset_index()

st.dataframe(
    city_perf.sort_values("MAE (°C)"),
    width='stretch'
)

