# 🌤️ ClimaSense

## Weather Intelligence for Operational Decision Support

ClimaSense is a Minimum Viable Product (MVP) demonstrating how historical weather observations and Machine Learning can support short-term operational decision-making.

The project combines data ingestion, exploratory data analysis, predictive modelling and interactive visualisation to transform weather data into practical decision-support information.

---

# Business Context

Weather conditions directly influence many operational activities, including road maintenance, transportation planning and field operations.

Being able to anticipate temperature variations one day ahead helps organisations prepare preventive actions, optimise resource allocation and reduce operational risks.

This project explores how historical weather data can be transformed into actionable operational insights through Machine Learning.

---

# Business Questions

This project addresses the following questions:

- How do weather conditions vary across major French cities?
- Which cities experience the highest exposure to frost conditions?
- Can tomorrow's maximum temperature (J+1) be predicted from historical observations?
- How can these predictions support operational decision-making?

---

# Project Objectives

The objectives of ClimaSense are to:

- analyse historical weather observations collected across France;
- identify regional climatic differences and seasonal patterns;
- build a J+1 temperature forecasting model;
- visualise predictions through an interactive dashboard;
- illustrate how Machine Learning can support operational planning.

---

# Dataset

**Source**

Open-Meteo Archive API

**Coverage**

- France
- 20 cities
- January 2022 – December 2022

The dataset includes daily meteorological observations such as:

- minimum temperature
- maximum temperature
- precipitation
- additional weather indicators used during feature engineering

---

# Solution Overview

```
Open-Meteo Archive API
            │
            ▼
      Data Ingestion
            │
            ▼
 Data Cleaning & Preparation
            │
            ▼
 Exploratory Data Analysis
            │
            ▼
   Feature Engineering
            │
            ▼
 Machine Learning Model
            │
            ▼
  J+1 Temperature Forecast
            │
            ▼
 Decision Support Dashboard
```

---

# Methodology

## Data Preparation

The dataset was cleaned and prepared before modelling.

Feature engineering includes:

- lag feature (J-1 temperature);
- cyclical seasonal encoding (sin / cos);
- temporal variables.

---

## Model Comparison

Several regression models were evaluated.

Models compared:

- Linear Regression ✅
- Polynomial Regression
- Ridge Regression

Evaluation metric:

- Mean Absolute Error (MAE)

Validation strategy:

- chronological train/test split;
- TimeSeriesSplit cross-validation.

Linear Regression provided the best balance between prediction accuracy, simplicity and generalisation.

---

# Results

Main observations include:

- average MAE close to **2.5°C**;
- better prediction accuracy in Mediterranean climates;
- higher variability in continental and mountainous regions;
- satisfactory J+1 forecasting performance for short-term operational support.

---

# Decision Support Dashboard

The Streamlit dashboard transforms model predictions into operational recommendations.

It provides:

- J+1 maximum temperature forecast;
- frost-risk indicators;
- preventive salting recommendation;
- historical prediction analysis;
- city performance comparison.

The objective is not simply to forecast tomorrow's temperature, but to support operational planning through clear and interpretable indicators.

---

# Business Insights

The analysis highlights several practical observations.

- Climate variability differs significantly across French regions.
- Historical weather observations provide useful information for short-term forecasting.
- Simple Machine Learning models can outperform more complex alternatives when data availability is limited.
- Translating predictions into operational recommendations increases the practical value of analytical models.

---

# Project Structure

```
ClimaSense/

├── notebooks/
│   ├── 01_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model.ipynb
│   └── 04_streamlit.ipynb
│
├── notebooks/data/
├── notebooks/models/
├── notebooks/outputs/
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/BadreddinB/ClimaSense.git

cd ClimaSense
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run streamlit_app/app.py
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Open-Meteo Archive API
- Git
- GitHub

---

# Limitations

Current limitations include:

- historical observations limited to 2022;
- J+1 forecasting horizon only;
- 20 French cities;
- offline predictions generated from historical data;
- no automatic model retraining.

---

# Future Improvements

Potential future developments include:

- extending the historical observation period;
- integrating additional meteorological variables;
- evaluating alternative forecasting models;
- using real-time weather observations;
- containerising the application with Docker;
- deploying the dashboard on Hugging Face Spaces.

---

# Operational Perspective

ClimaSense illustrates how Machine Learning can complement operational expertise by transforming historical weather observations into decision-support information.

The objective is not to replace operational judgement, but to provide an additional analytical tool supporting short-term planning and risk anticipation.
 
