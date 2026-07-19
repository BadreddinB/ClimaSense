# 🌤️ ClimaSense

**AI-powered Short-Term Temperature Forecasting**

*Supporting Weather-Sensitive Decision Making*

🔗 **Live Demo:** https://climasense.streamlit.app/

---

## Overview

ClimaSense is an end-to-end data science project demonstrating how historical weather observations and machine learning can support short-term temperature forecasting.

Rather than focusing solely on predictive performance, the project illustrates a complete analytical workflow—from data acquisition and exploratory analysis to model development and interactive visualization. The objective is to transform raw weather observations into interpretable insights that can support weather-sensitive operational decisions.

---

## Business Context

Weather conditions influence thousands of operational decisions every day.

Road maintenance crews anticipate frost events, airport operators prepare ground activities, energy providers adjust production forecasts, and logistics companies adapt transportation planning according to expected weather conditions.

Although operational decisions rely on many sources of information, having a reasonably accurate estimate of tomorrow's maximum temperature can improve planning and reduce uncertainty.

ClimaSense explores this idea using publicly available weather observations and interpretable machine learning techniques.

The project is intentionally designed as a generic decision-support application rather than a domain-specific solution.

Possible application areas include:

- Winter road maintenance

- Airport ground operations

- Energy demand planning

- Logistics planning

- Emergency preparedness

These examples illustrate potential operational contexts only. ClimaSense predicts temperature and does not replace specialized forecasting systems or operational expertise.

---

## Business Problem

The project investigates the following questions:

- How do weather conditions vary across major French cities?

- Which regions experience the greatest exposure to frost or extreme temperatures?

- Can tomorrow's maximum temperature be predicted from historical observations?

- Does a simple machine learning model provide added value compared with a naïve persistence baseline?

---

## Objectives

The objectives of ClimaSense are to:

- collect and validate historical weather observations;

- explore regional climatic differences and seasonal patterns;

- engineer interpretable predictive features;

- develop and evaluate a next-day (J+1) forecasting model;

- compare machine learning against a persistence baseline;

- communicate results through an interactive Streamlit dashboard.

---

## Dataset

| | |

|---|---|

| **Source** | Open-Meteo Archive API |

| **Coverage** | 20 French cities |

| **Period** | January–December 2022 |

| **Granularity** | Daily observations |

| **Observations** | 7,300 |

| **Missing values** | None |


Daily variables include:

- Maximum temperature

- Minimum temperature

- Total precipitation
       
---

## Methodology

The analytical workflow is organized into three Jupyter notebooks.

### Notebook 1 — Data Ingestion

- Retrieve historical observations from the Open-Meteo Archive API.

- Consolidate observations across 20 French cities.

- Validate data completeness and quality.

- Export raw and processed datasets.

### Notebook 2 — Exploratory Data Analysis

The exploratory analysis investigates:

- temperature distributions;

- seasonal patterns;

- regional climatic differences;

- correlations between weather variables;

- frost and heat events;

- statistical anomalies using Z-score detection.

### Notebook 3 — Forecasting Model

Feature engineering generates four interpretable predictors.

| Feature | Purpose |

|---|---|

| Previous day's maximum temperature | Short-term persistence |

| Previous day's minimum temperature | Overnight context |

| Day-of-year (sin) | Seasonal cycle |

| Day-of-year (cos) | Seasonal cycle |

Three regression algorithms are evaluated using **TimeSeriesSplit (5 folds)**.

| Model | Average MAE |

|---|---:|

| Linear Regression | **3.31 °C** |

| Ridge Regression | 12.80 °C |

| Polynomial Regression | 19.48 °C |

Linear Regression is selected because it provides the best balance between prediction accuracy, stability, interpretability, and generalization.

---

## Results

Each city is trained independently using a chronological train/test split:

- **Training:** January–September 2022

- **Testing:** October–December 2022

Overall model performance across the 20 cities:

| Metric | Range |

|---|---|

| MAE | 1.60–3.58 °C |

| R² | 0.63–0.78 |

The model successfully captures seasonal temperature dynamics while maintaining consistent predictive performance across different climatic regions.

### Baseline Comparison

Performance is compared against a naïve persistence baseline using today's temperature as tomorrow's prediction.

The results show that:

- the machine learning model slightly outperforms the baseline in several cities;

- the persistence baseline remains highly competitive for one-day forecasting;

- additional historical data and richer features would likely improve predictive performance.

This comparison highlights an important lesson: more complex models do not automatically outperform simple baselines.

---

## Interactive Dashboard

The Streamlit application transforms model outputs into an accessible visualization interface.

It provides:

- Forecast performance metrics for each city;

- Mean Absolute Error (MAE);

- Accuracy within ±2 °C;

- Frost-risk day count;

- Interactive Plotly visualization comparing predicted and observed temperatures;

- Performance comparison across all 20 cities.

The dashboard is designed to communicate model performance transparently rather than simulate a production forecasting platform.

---

## Repository Structure

```

ClimaSense/

├── data/

│   ├── raw/

│   ├── processed/

│   └── predictions/

│

├── models/

│

├── notebooks/

│   ├── 01_ingestion.ipynb

│   ├── 02_eda.ipynb

│   └── 03_model.ipynb

│

├── outputs/

│   └── figures/

│

├── streamlit_app/

│   └── app.py

│

├── requirements.txt

├── README.md

└── LICENSE

```

---

## Reproducing the Project

Clone the repository:

```bash

git clone https://github.com/BadreddinB/ClimaSense.git

cd ClimaSense

```

Install dependencies:

```bash

pip install -r requirements.txt

```

Run the notebooks sequentially:

```

01_ingestion.ipynb

02_eda.ipynb

03_model.ipynb

```

Launch the dashboard:

```bash

streamlit run streamlit_app/app.py

```

---

## Technologies

- Python

- Pandas

- NumPy

- Scikit-learn

- Plotly

- Matplotlib

- Streamlit

- Open-Meteo Archive API

- Git

- GitHub

---

## Limitations

Current limitations include:

- one year of historical observations;

- 20 French cities;

- four engineered input features;

- offline predictions based on historical data;

- no automatic retraining pipeline.

These constraints are intentional and keep the project focused on demonstrating an interpretable end-to-end analytical workflow.

---

## Future Improvements

Potential extensions include:

- multi-year historical observations;

- additional meteorological variables (humidity, wind speed, pressure);

- multi-day forecasting (J+3, J+7);

- comparison with dedicated time-series forecasting models;

- integration of live weather observations;

- automated retraining and model monitoring;

- Docker containerization.

---

## Key Takeaways

ClimaSense demonstrates how a complete data science workflow can transform a business question into an interpretable machine learning application.

Beyond predictive modeling, the project emphasizes:

- translating an operational problem into an analytical workflow;

- building reproducible data pipelines;

- evaluating models using appropriate validation strategies;

- comparing machine learning against meaningful baselines;

- communicating results through an interactive dashboard.

The project intentionally favors transparency, reproducibility, and simplicity over algorithmic complexity, illustrating how machine learning can support operational decision-making without unnecessary overengineering.
