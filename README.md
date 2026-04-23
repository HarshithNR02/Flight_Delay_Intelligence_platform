# Flight Delay Intelligence Platform

**Predict, explain, and quantify US domestic flight delays using 18.2 million real flights from four federal government datasets.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://your-app.streamlit.app)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi)](http://your-api-url/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## Live Demo

| Resource | URL |
|----------|-----|
| Streamlit Dashboard | https://your-app.streamlit.app |
| FastAPI Docs | http://your-api-url/docs |
| FastAPI Endpoint | `POST http://your-api-url/predict` |

---

## What This Project Does

This system answers three questions for any US domestic flight:

1. **Will it be delayed?** — Binary classification (AUC 0.8629)
2. **By how many minutes?** — Regression (MAE 16.82 min)
3. **Why?** — SHAP TreeExplainer with per-flight explanations

---

## Key Results

| Metric | Value |
|--------|-------|
| Classifier AUC | 0.8629 |
| Precision @ threshold 0.65 | 71.9% |
| Recall @ threshold 0.65 | 61.6% |
| Regression MAE | 16.82 minutes |
| ±15 min accuracy | 69.1% |
| ±30 min accuracy | 88.3% |
| Annualized savings | $2.78 billion |
| Training flights | 13.7M (Jan 2023 – Dec 2024) |
| Test flights | 4.5M (Jan 2025 – Aug 2025) |

---

## The Key Finding

**58% of all flight delays are fundamentally unpredictable from public data.**

I analyzed 602,377 delayed flights with zero prior cascade warning and broke them down by FAA delay cause:

| Cause | % of Surprise Delays |
|-------|----------------------|
| CARRIER_DELAY (maintenance, crew) | 42.9% |
| NAS_DELAY (ATC, ground stops) | 28.3% |
| LATE_AIRCRAFT (outside tracking window) | 20.8% |
| WEATHER_DELAY | 7.9% |

71.2% of unpredictable delays come from carrier maintenance and ATC decisions that happen at the gate, minutes before departure. No public dataset contains pre-departure maintenance inspection results or real-time ATC ground stop decisions.

The model catches **LATE_AIRCRAFT_DELAY at 71.5%** — the only highly predictable delay pattern — because cascade propagation leaves a traceable signal in aircraft tail number history.

---

## SHAP Finding: Split Importance Was Misleading

Standard LightGBM feature importance ranked ORIGIN and DEST as the top 2 predictors. SHAP TreeExplainer told a different story:

| Feature | SHAP Rank | Split Rank | Mean \|SHAP\| |
|---------|-----------|------------|--------------|
| real_time_turn_gap | #1 | #11 | 0.773 |
| scheduled_turnaround_buffer | #2 | #12 | 0.276 |
| prev_tail_arr_delay | #3 | #15 | 0.274 |
| dest_inbound_arr_delay_3h | #4 | #7 | 0.252 |
| airport_fatigue_index | #5 | #33 | 0.156 |
| ORIGIN | #6 | #1 | 0.106 |

`real_time_turn_gap` — the scheduled turnaround buffer minus the actual incoming aircraft delay — is the strongest predictor with nearly 3x the impact of the next feature. Split importance had it at #11.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
│  BTS On-Time  │  NOAA ISD-Lite  │  Form 41  │  T-100   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Data Pipeline (Notebooks 01-05)             │
│  Clean → Merge → EDA → Feature Engineering (61 features)│
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Models                                  │
│  K-Means Clustering │ LightGBM Classifier │ LightGBM Reg│
│  Optuna Tuning      │ SHAP TreeExplainer  │             │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   FastAPI       │         │   Streamlit      │
│   /predict      │         │   7-page app     │
│   (REST API)    │         │   (Dashboard)    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│           Azure Container Instances (Docker)             │
└─────────────────────────────────────────────────────────┘
```

---

## Datasets

| Dataset | Source | Size | Coverage |
|---------|--------|------|----------|
| BTS On-Time Performance | [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ) | 18.2M flights | Jan 2023 – Aug 2025 |
| NOAA ISD-Lite Weather | [ncei.noaa.gov](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | 2.3M hourly obs | 100 airports |
| BTS Form 41 Financials | [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM) | Quarterly | 15 carriers |
| BTS T-100 Airport Data | [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIL) | Monthly | 361 airports |

All datasets are free public federal government sources. No API keys required.

Weather data covers 100 major US airports representing **91.3% of all domestic flights**. Remaining 8.7% of flights at smaller regional airports received NaN weather values handled natively by LightGBM.

---

## Feature Engineering (61 features)

Features are grouped into 8 categories:

**Time (6):** MONTH, DAY_OF_WEEK, DEP_HOUR, ARR_HOUR, IS_HOLIDAY, day_of_year

**Flight & Financial (3):** DISTANCE, profit_margin, origin_monthly_passengers

**Weather Origin (7):** temp, dew_point, pressure, wind_dir, wind_speed, precip_1hr, weather_severity

**Weather Destination (6):** same minus precip_1hr

**Weather Deltas (4):** origin/dest pressure and wind speed change over 3 hours — captures incoming storm fronts

**Rolling Historical (11):** airline/airport/route/flight delay rates at 7d and 30d windows, point-in-time safe

**Cascade Features (3):** cascade_score, cascade_delay_minutes, hours_since_last_delay — 6-hour empirically validated lookback window

**Real-Time Airport State (13):** inbound_arr_delay_3h, dest_inbound_arr_delay_3h, prev_tail_arr_delay, national_hub_delay_2h, real_time_turn_gap, airport_fatigue_index, origin_dep_delay_rate_1h, dest_dep_delay_rate_1h, origin_stress_index, tail_delays_today, tail_active_hours, origin_pressure_drop_stress, hourly_flight_count/dest_hourly_flight_count

**Categoricals (4):** OP_UNIQUE_CARRIER, ORIGIN, DEST, airline_cluster_label — handled via LightGBM native gradient-based categorical splits (no target encoding, no one-hot)

---

## Model Iteration History

| Iteration | Features | AUC | Key Addition |
|-----------|----------|-----|--------------|
| V1 | 16 | 0.6505 | Time, distance, financials |
| V2 | 34 | 0.6819 | NOAA weather |
| V3 | 44 | 0.7888 | Cascade, rolling 30d, clusters |
| V5 | 47 | 0.7980 | ORIGIN, DEST, CARRIER categorical |
| V7 | 53 | 0.8014 | Congestion, turnaround buffer |
| V8 | 48 | 0.8572 | Inbound delays, hub fever, prev tail |
| Optuna | 48 | 0.8578 | 50 trials, TPE sampler |
| V9 | 59 | 0.8623 | Pressure/wind deltas, airport state |
| Final | 61 | 0.8629 | Airport fatigue, pressure stress |

**Temporal split:** No random shuffle. Model trained on 2023-2024, tested on Jan–Aug 2025. Zero data leakage.

---

## Cost-Benefit Analysis

Source: [Airlines for America (A4A), 2024 DOT Form 41 data](https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/) — $100.76/minute total direct operating cost across all US scheduled passenger airlines.

| Metric | Value |
|--------|-------|
| Total delay cost (no model) | $7.75B (8 months) |
| Total cost with model | $5.90B |
| Net savings (8 months) | $1.85B |
| Annualized savings | $2.78B |
| Per caught delay savings | $2,998 |
| ROI | $30.60 per $1 spent on false alarms |

Conservative 35% proactive savings rate. Published research shows 50%+ with optimal control.

---

## FastAPI Endpoint

```bash
POST /predict
Content-Type: application/json

{
  "carrier": "AA",
  "origin": "ORD",
  "dest": "LAX",
  "dep_hour": 14,
  "month": 7,
  "day_of_week": 2
}
```

Response:
```json
{
  "delay_probability": 0.78,
  "estimated_delay_minutes": 42,
  "risk_level": "High",
  "airline_cost": 4232,
  "potential_savings": 1481,
  "top_features": [
    {"feature": "real_time_turn_gap", "shap_value": 4.21, "direction": "delay"},
    {"feature": "dest_inbound_arr_delay_3h", "shap_value": 1.34, "direction": "delay"},
    {"feature": "airport_fatigue_index", "shap_value": 0.53, "direction": "delay"}
  ]
}
```

Full API docs at `http://your-api-url/docs`

---

## Streamlit Dashboard (7 pages)

| Page | Description |
|------|-------------|
| Dashboard | System overview — 18.2M flights, delay rates by month/hour/carrier |
| Flight Predictor | Look up real Jan–Aug 2025 flights, get prediction + SHAP explanation |
| Airline Rankings | Compare all 15 carriers by delay rate, AUC, cascade stats, cluster |
| Route Risk | 5,855 routes scored, filter by tier, lookup any origin-destination |
| Cost Impact | $2.78B savings breakdown, sensitivity analysis at 15%-50% savings rates |
| Cascade Tracker | Trace aircraft tail numbers through cascading delays, N524AE example |
| Model Insights | Global SHAP, dependence plots, threshold analysis, 58% ceiling proof |

---

## How to Run Locally

```bash
# Clone repo
git clone https://github.com/HarshithNR02/Flight_Delay_Intelligence_platform.git
cd Flight_Delay_Intelligence_platform

# Create conda environment
conda create -n flight-delay python=3.11
conda activate flight-delay
pip install -r requirements.txt

# Download data (see Data Sources above)
# Place datasets in dataset/ folder

# Run notebooks in order (01a through 15)

# Run Streamlit app
streamlit run streamlit_app/app.py

# Run FastAPI
uvicorn api.main:app --reload
```

> **Note:** Large files (dataset parquets, model pkl files) are not included in this repo due to GitHub file size limits. Download raw data from the BTS and NOAA sources above and run the pipeline notebooks to reproduce.

---

## Project Structure

```
flight-delay-intelligence-platform/
├── 01a_prep_flights.ipynb
├── 01b_prep_airports.ipynb
├── 01c_prep_weather.ipynb
├── 01d_prep_financials.ipynb
├── 02_clean_flights.ipynb
├── 03_eda.ipynb
├── 04_master_merge.ipynb
├── 05_eda_and_feature_engineering_v1.ipynb
├── 06_clustering.ipynb
├── 07a_feature_engineering_v2.ipynb
├── 07b_feature_engineering_v3.ipynb
├── 08_classifier_optuna_tuning.ipynb
├── 09_feature_engineering_v4_and_classifier_model_training.ipynb
├── 10_regressor_optuna_training.ipynb
├── 11_regressor_model_training.ipynb
├── 12_shap.ipynb
├── 13_cost_benefit.ipynb
├── 14_cascade_delay_analysis.ipynb
├── 15_route_risk_score.ipynb
├── 16_delay_unpredictability_proof.ipynb
├── streamlit_app/
│   ├── app.py
│   └── pages/
│       ├── 1_Flight_Predictor.py
│       ├── 2_Airline_Rankings.py
│       ├── 3_Route_Risk.py
│       ├── 4_Cost_Impact.py
│       ├── 5_Cascade_Tracker.py
│       └── 6_Model_Insights.py
├── api/
│   ├── main.py
│   └── predict.py
├── models/
│   ├── feature_list_final.txt
│   ├── optuna_best_params.json
│   ├── model_final_metadata.json
│   ├── shap_beeswarm_v2.png
│   ├── shap_dependence_plots_v2.png
│   ├── shap_waterfall_high_risk.png
│   └── shap_waterfall_low_risk.png
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | LightGBM, Scikit-learn, Optuna |
| Explainability | SHAP TreeExplainer |
| Data | Pandas, NumPy, PyArrow |
| Visualization | Plotly, Streamlit |
| API | FastAPI, Uvicorn |
| Deployment | Docker, Azure Container Instances |
| Language | Python 3.11 |

---

## About

Built by **Harshith** — Data Science student at UMass Dartmouth.

Targeting roles at Spotify, DoorDash, Shopify, Expedia.

- GitHub: [@HarshithNR02](https://github.com/HarshithNR02)
- LinkedIn: [your linkedin url]

---

*All data from federal government sources (.gov). No proprietary data used.*
