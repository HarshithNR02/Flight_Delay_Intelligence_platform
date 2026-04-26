# Flight Delay Intelligence Platform

**Predict, explain, and quantify US domestic flight delays using 18.2 million real flights from four federal government datasets.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://harshith02-flight-delay-intelligence.hf.space)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## Live Demo

**🔗 [harshith02-flight-delay-intelligence.hf.space](https://harshith02-flight-delay-intelligence.hf.space)**

7-page interactive dashboard: flight predictor with SHAP explanations, airline rankings, route risk scoring, cost-benefit analysis, cascade delay tracker, and model insights.

---

## What This Project Does

This system answers three questions for any US domestic flight:

1. **Will it be delayed?** — Binary classifier (AUC 0.8629)
2. **By how many minutes?** — Regressor (MAE 16.82 min)
3. **Why?** — SHAP TreeExplainer with per-flight explanations

---

## Key Results

| Metric | Value |
|--------|-------|
| Classifier AUC | **0.8629** |
| Precision @ 0.65 | 71.9% |
| Recall @ 0.65 | 61.6% |
| Regressor MAE | 16.82 min |
| ±15 min accuracy | 69.8% |
| ±30 min accuracy | 88.7% |
| Annualized savings | $2.78 billion |
| Training set | 13.7M flights (Jan 2023 – Dec 2024) |
| Test set | 4.5M flights (Jan – Aug 2025) |

---

## The Key Finding

**58% of all flight delays are fundamentally unpredictable from public data.**

602,377 delayed flights with zero prior cascade warning, broken down by FAA cause:

| Cause | % of Surprise Delays |
|-------|----------------------|
| CARRIER_DELAY (maintenance, crew) | 42.9% |
| NAS_DELAY (ATC, ground stops) | 28.3% |
| LATE_AIRCRAFT (outside tracking window) | 20.8% |
| WEATHER_DELAY | 7.9% |

71.2% of unpredictable delays come from carrier maintenance and ATC decisions — events that happen at the gate minutes before departure. No public dataset captures this.

The model catches **LATE_AIRCRAFT_DELAY at 71.5%** because cascade propagation leaves a traceable signal in aircraft tail number history. That's where the model earns its AUC.

---

## SHAP: Why Split Importance Was Misleading

Standard LightGBM feature importance ranked ORIGIN and DEST as top predictors. SHAP told a different story:

| Feature | SHAP Rank | Split Rank | Mean \|SHAP\| |
|---------|-----------|------------|--------------|
| real_time_turn_gap | #1 | #11 | 0.773 |
| scheduled_turnaround_buffer | #2 | #12 | 0.276 |
| prev_tail_arr_delay | #3 | #15 | 0.274 |
| dest_inbound_arr_delay_3h | #4 | #7 | 0.252 |
| airport_fatigue_index | #5 | #33 | 0.156 |
| ORIGIN | #6 | #1 | 0.106 |

`real_time_turn_gap` — scheduled turnaround buffer minus incoming aircraft delay — is the strongest predictor with nearly 3x the impact of the next feature. Split importance had it at #11.

---

## Datasets

| Dataset | Source | Size | Coverage |
|---------|--------|------|----------|
| BTS On-Time Performance | [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ) | 18.2M flights | Jan 2023 – Aug 2025 |
| NOAA ISD-Lite Weather | [ncei.noaa.gov](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | 2.3M hourly obs | 100 airports |
| BTS Form 41 Financials | [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM) | Quarterly | 15 carriers |
| BTS T-100 Airport Data | [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIL) | Monthly | 361 airports |

All free public federal government sources. Weather covers 100 airports representing 91.3% of all domestic flights.

---

## Feature Engineering (61 features)

**Time (6):** MONTH, DAY_OF_WEEK, DEP_HOUR, ARR_HOUR, IS_HOLIDAY, day_of_year

**Flight & Financial (3):** DISTANCE, profit_margin, origin_monthly_passengers

**Weather (13):** Origin and destination temp, dew_point, pressure, wind_dir, wind_speed, precip_1hr, weather_severity

**Weather Deltas (4):** 3-hour pressure and wind speed change at origin/dest — captures incoming storm fronts

**Rolling Historical (11):** Airline, airport, route, and flight-number delay rates at 7d and 30d windows. All point-in-time safe with shift(1) before rolling.

**Cascade (3):** cascade_score, cascade_delay_minutes, hours_since_last_delay — 6-hour empirically validated lookback window per aircraft tail

**Real-Time Airport State (17):** inbound_arr_delay_3h, prev_tail_arr_delay, national_hub_delay_2h, real_time_turn_gap, airport_fatigue_index, origin_stress_index, tail_delays_today, tail_active_hours, and more

**Categorical (4):** OP_UNIQUE_CARRIER, ORIGIN, DEST, airline_cluster_label — LightGBM native gradient-based splits

---

## Model Iteration History

| Iteration | Features | AUC | Key Addition |
|-----------|----------|-----|--------------|
| V1 | 16 | 0.6505 | Time, distance, financials |
| V2 | 34 | 0.6819 | NOAA weather |
| V3 | 44 | 0.7888 | Cascade algorithm, rolling 30d rates, K-Means clusters |
| V5 | 47 | 0.7980 | ORIGIN, DEST, CARRIER as native categoricals |
| V7 | 53 | 0.8014 | Congestion features, turnaround buffer |
| V8 | 48 | 0.8572 | Inbound delays, hub fever, previous tail delay |
| Optuna | 48 | 0.8578 | 50-trial TPE hyperparameter search |
| V9 | 59 | 0.8623 | Pressure/wind deltas, airport state variables |
| Final | 61 | 0.8629 | Airport fatigue index, pressure drop stress |

Temporal split — no random shuffle. Train: 2023–2024, Test: Jan–Aug 2025. Zero data leakage.

---

## Cost-Benefit Analysis

Source: [Airlines for America (A4A)](https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/), 2024 DOT Form 41 — $100.76/minute total operating cost.

| Metric | Value |
|--------|-------|
| Total delay cost (no model) | $7.75B (8 months) |
| Total cost with model | $5.90B |
| Net savings | $1.85B → $2.78B annualized |
| Per caught delay | $2,998 saved |
| ROI | $30.60 per $1 spent on false alarms |

Conservative 35% proactive savings rate. Published research supports 50%+ with optimal control.

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
│              Data Pipeline (Notebooks 01–08)             │
│  Prep → Clean → EDA → Merge → Feature Engineering (61)  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                      Models                              │
│  K-Means Clustering  │  LightGBM Classifier (AUC 0.86)  │
│  Optuna Tuning       │  LightGBM Regressor (MAE 16.8)   │
│  SHAP TreeExplainer  │  Route Risk Scoring               │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   FastAPI        │         │   Streamlit      │
│   /predict       │         │   7-page app     │
└─────────────────┘         └─────────────────┘
          │                           │
          └─────────────┬─────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│      Hugging Face Spaces (Docker) + Azure Blob Storage   │
└─────────────────────────────────────────────────────────┘
```

---

## Streamlit Dashboard

| Page | Description |
|------|-------------|
| Dashboard | Overview — 18.2M flights, delay rates by month/hour/carrier |
| Flight Predictor | Search real Jan–Aug 2025 flights, get prediction + SHAP explanation |
| Airline Rankings | 15 carriers compared by delay rate, AUC, cascade stats, cluster |
| Route Risk | 5,855 routes scored with composite risk index, filterable |
| Cost Impact | $2.78B savings breakdown, sensitivity analysis (15%–50%) |
| Cascade Tracker | Trace aircraft tail numbers through cascading delay chains |
| Model Insights | Global SHAP, dependence plots, threshold curves, 58% ceiling proof |

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

---

## Project Structure

```
├── 01a_prep_flights.ipynb                              # BTS flight data loading
├── 01b_prep_airports.ipynb                             # BTS T-100 airport traffic
├── 01c_prep_weather.ipynb                              # NOAA ISD-Lite weather
├── 01d_prep_financials.ipynb                           # BTS Form 41 financials
├── 02_clean_flights.ipynb                              # Remove cancelled/diverted
├── 03_eda.ipynb                                        # Exploratory data analysis
├── 04_master_merge.ipynb                               # Join all 4 datasets
├── 05a_feature_engineering_v1_cascade_weather.ipynb     # Cascade, weather severity, rolling rates
├── 05b_feature_engineering_v2_rolling_rates_turnaround.ipynb  # Flight-level rates, turnaround
├── 05c_feature_engineering_v3_inbound_tail_hub.ipynb    # Inbound delays, tail history, hub fever
├── 05d_feature_engineering_v4_weather_deltas_stress.ipynb  # Weather deltas, stress indices
├── 06_airline_airport_clustering.ipynb                 # K-Means segmentation
├── 07_classifier_optuna_tuning.ipynb                   # 50-trial Optuna search
├── 08_feature_engineering_final.ipynb                  # Clean 61-feature pipeline
├── 09_classifier_model_training.ipynb                  # Final LightGBM classifier
├── 10_regressor_optuna_training.ipynb                  # Regressor Optuna search
├── 11_regressor_model_training.ipynb                   # Final LightGBM regressor
├── 12_cascade_delay_analysis.ipynb                     # Cascade propagation analysis
├── 13_shap_analysis.ipynb                              # SHAP explanations
├── 14_cost_benefit.ipynb                               # $2.78B savings analysis
├── 15_route_risk_score.ipynb                           # 5,855 routes scored
├── 16_delay_unpredictability_proof.ipynb               # 58% ceiling proof
├── streamlit_app/
│   ├── app.py
│   └── pages/
├── api/
│   ├── main.py
│   └── predict.py
├── Dockerfile
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
| Deployment | Docker, Hugging Face Spaces, Azure Blob Storage |
| Language | Python 3.11 |

---

## How to Run Locally

```bash
git clone https://github.com/HarshithNR02/Flight_Delay_Intelligence_platform.git
cd Flight_Delay_Intelligence_platform

conda create -n flight-delay python=3.11
conda activate flight-delay
pip install -r requirements.txt

# Download raw data from BTS and NOAA (see Datasets above)
# Run notebooks 01a through 16 in order

streamlit run streamlit_app/app.py
uvicorn api.main:app --reload
```

> Large files (datasets, model pickles) are excluded via .gitignore. Download raw data and run the pipeline to reproduce.

---

## About

Built by **Harshith Nerlikere Ramesh** — Data Science, UMass Dartmouth

[![GitHub](https://img.shields.io/badge/GitHub-HarshithNR02-181717?style=flat&logo=github)](https://github.com/HarshithNR02)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshith-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/harshithnr2002)

---

*All data from federal government sources (.gov). No proprietary data used.*
