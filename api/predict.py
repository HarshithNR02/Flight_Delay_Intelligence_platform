import os
import joblib
import shap
import numpy as np
import pandas as pd
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CAT_FEATURES = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'airline_cluster_label']


class FlightRequest(BaseModel):
    OP_UNIQUE_CARRIER: str = Field(..., example="AA")
    ORIGIN: str           = Field(..., example="ORD")
    DEST: str             = Field(..., example="LAX")
    airline_cluster_label: str = Field(..., example="Mainline_Legacy")

    MONTH: int            = Field(..., ge=1, le=12, example=7)
    DAY_OF_WEEK: int      = Field(..., ge=1, le=7, example=2)
    DEP_HOUR: int         = Field(..., ge=0, le=23, example=14)
    ARR_HOUR: int         = Field(..., ge=0, le=23, example=16)
    IS_HOLIDAY: int       = Field(..., ge=0, le=1, example=0)
    day_of_year: int      = Field(..., ge=1, le=366, example=196)

    DISTANCE: float       = Field(..., example=1745.0)
    profit_margin: float  = Field(..., example=0.12)
    origin_monthly_passengers: float = Field(..., example=1500000.0)

    origin_temp: float              = Field(..., example=72.0)
    origin_dew_point: float         = Field(..., example=55.0)
    origin_pressure: float          = Field(..., example=1013.0)
    origin_wind_dir: float          = Field(..., example=270.0)
    origin_wind_speed: float        = Field(..., example=12.0)
    origin_precip_1hr: float        = Field(..., example=0.0)
    origin_weather_severity: float  = Field(..., example=0.1)

    dest_temp: float                = Field(..., example=85.0)
    dest_dew_point: float           = Field(..., example=60.0)
    dest_pressure: float            = Field(..., example=1010.0)
    dest_wind_dir: float            = Field(..., example=180.0)
    dest_wind_speed: float          = Field(..., example=8.0)
    dest_precip_1hr: float          = Field(..., example=0.0)
    dest_weather_severity: float    = Field(..., example=0.0)

    origin_pressure_delta_3h: float     = Field(..., example=-1.2)
    dest_pressure_delta_3h: float       = Field(..., example=0.5)
    origin_wind_speed_delta_3h: float   = Field(..., example=3.0)
    dest_wind_speed_delta_3h: float     = Field(..., example=-1.0)

    airline_delay_rate_30d: float           = Field(..., example=0.22)
    origin_delay_rate_30d: float            = Field(..., example=0.24)
    dest_delay_rate_30d: float              = Field(..., example=0.19)
    route_delay_rate_30d: float             = Field(..., example=0.21)
    origin_avg_taxi_out_30d: float          = Field(..., example=18.5)
    flight_num_delay_rate_30d: float        = Field(..., example=0.25)
    origin_hour_delay_rate_30d: float       = Field(..., example=0.28)
    carrier_origin_delay_rate_30d: float    = Field(..., example=0.23)
    dest_hour_delay_rate_30d: float         = Field(..., example=0.20)
    airline_delay_rate_7d: float            = Field(..., example=0.26)
    origin_delay_rate_7d: float             = Field(..., example=0.27)

    cascade_score: float            = Field(..., example=0.4)
    cascade_delay_minutes: float    = Field(..., example=25.0)
    hours_since_last_delay: float   = Field(..., example=2.5)

    hourly_flight_count: float          = Field(..., example=42.0)
    scheduled_turnaround_buffer: float  = Field(..., example=55.0)
    tail_flight_num_today: float        = Field(..., example=3.0)
    dest_hourly_flight_count: float     = Field(..., example=38.0)

    inbound_arr_delay_3h: float         = Field(..., example=12.0)
    dest_inbound_arr_delay_3h: float    = Field(..., example=8.0)
    prev_tail_arr_delay: float          = Field(..., example=20.0)
    national_hub_delay_2h: float        = Field(..., example=5.0)
    origin_dep_delay_rate_1h: float     = Field(..., example=0.30)
    dest_dep_delay_rate_1h: float       = Field(..., example=0.18)
    origin_stress_index: float          = Field(..., example=0.45)
    real_time_turn_gap: float           = Field(..., example=35.0)
    tail_delays_today: float            = Field(..., example=1.0)
    tail_active_hours: float            = Field(..., example=9.5)
    origin_pressure_drop_stress: float  = Field(..., example=0.2)
    airport_fatigue_index: float        = Field(..., example=0.55)


class SHAPFeature(BaseModel):
    feature: str
    value: float | str
    shap_value: float
    direction: str


class PredictResponse(BaseModel):
    delay_probability: float
    risk_level: str
    estimated_delay_minutes: float
    airline_cost_usd: float
    potential_savings_usd: float
    top_shap_features: list[SHAPFeature]

@lru_cache(maxsize=1)
def get_models():
    clf_path = os.path.join(PROJECT_ROOT, 'models', 'lgbm_delay_classifier_final.pkl')
    reg_path = os.path.join(PROJECT_ROOT, 'models', 'lgbm_delay_regressor_final.pkl')
    feat_path = os.path.join(PROJECT_ROOT, 'models', 'feature_list_final.txt')

    if not os.path.exists(clf_path):
        raise RuntimeError(f"Classifier not found at {clf_path}")
    if not os.path.exists(reg_path):
        raise RuntimeError(f"Regressor not found at {reg_path}")

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    explainer = shap.TreeExplainer(clf)

    with open(feat_path) as f:
        features = f.read().strip().split('\n')

    return clf, reg, explainer, features

@router.post("/predict", response_model=PredictResponse)
def predict(flight: FlightRequest):
    try:
        clf, reg, explainer, FEATURES = get_models()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = flight.model_dump()
    df = pd.DataFrame([data])

    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    try:
        X = df[FEATURES]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    prob = float(clf.predict_proba(X)[:, 1][0])
    delay_min = float(reg.predict(X)[0])

    if prob >= 0.65:
        risk_level = "High"
    elif prob >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    airline_cost = max(0.0, delay_min) * 100.76
    potential_savings = airline_cost * 0.35 if prob >= 0.65 else 0.0

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    sv = shap_vals[0]

    feat_shap = sorted(
        zip(FEATURES, sv, X.iloc[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = []
    for feat, shap_val, feat_val in feat_shap[:10]:
        try:
            val = float(feat_val)
            display_val: float | str = round(val, 2) if not np.isnan(val) else "N/A"
        except (ValueError, TypeError):
            display_val = str(feat_val)

        top_features.append(SHAPFeature(
            feature=feat,
            value=display_val,
            shap_value=round(float(shap_val), 4),
            direction="delay" if shap_val > 0 else "on-time",
        ))

    return PredictResponse(
        delay_probability=round(prob, 4),
        risk_level=risk_level,
        estimated_delay_minutes=round(delay_min, 1),
        airline_cost_usd=round(airline_cost, 2),
        potential_savings_usd=round(potential_savings, 2),
        top_shap_features=top_features,
    )
