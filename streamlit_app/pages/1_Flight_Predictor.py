import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))

st.set_page_config(page_title="Flight Delay Predictor", layout="wide")
st.title("✈️ Flight Delay Predictor")
st.markdown("Search for a real flight from Jan–Aug 2025 and get the model's prediction with explanation.")

@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(PROJECT_ROOT, 'models/lgbm_delay_classifier_final.pkl'))
    reg = joblib.load(os.path.join(PROJECT_ROOT, 'models/lgbm_delay_regressor_final.pkl'))
    explainer = shap.TreeExplainer(clf)
    return clf, reg, explainer

@st.cache_data
def load_test_data():
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'dataset/merged_flights_fe_v2.parquet'))
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    test = df[df['FL_DATE'] >= '2025-01-01'].copy()
    for col in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'airline_cluster_label']:
        test[col] = test[col].astype('category')
    return test

with open(os.path.join(PROJECT_ROOT, 'models/feature_list_final.txt')) as f:
    FEATURES = f.read().strip().split('\n')

clf, reg, explainer = load_models()

with st.spinner("Loading flight data..."):
    test = load_test_data()

st.subheader("Search for a Flight")

carriers = sorted(test['OP_UNIQUE_CARRIER'].cat.categories)
origins  = sorted(test['ORIGIN'].cat.categories)
dests    = sorted(test['DEST'].cat.categories)

col1, col2, col3, col4 = st.columns(4)
with col1:
    carrier = st.selectbox("Airline", carriers, index=carriers.index('AA') if 'AA' in carriers else 0)
with col2:
    origin = st.selectbox("Origin", origins, index=origins.index('ORD') if 'ORD' in origins else 0)
with col3:
    dest = st.selectbox("Destination", dests, index=dests.index('LAX') if 'LAX' in dests else 0)
with col4:
    date = st.date_input("Date", value=pd.to_datetime("2025-07-15"),
        min_value=pd.to_datetime("2025-01-01"), max_value=pd.to_datetime("2025-08-27"))

matches = test[
    (test['OP_UNIQUE_CARRIER'] == carrier) &
    (test['ORIGIN'] == origin) &
    (test['DEST'] == dest) &
    (test['FL_DATE'] == str(date))
].copy()

if len(matches) == 0:
    st.warning(f"No flights found for {carrier} {origin}→{dest} on {date}. Try a different date or route.")
    st.stop()

st.markdown(f"**{len(matches)} flight(s) found**")
display_cols = ['DEP_HOUR', 'ARR_HOUR', 'ARR_DEL15', 'ARR_DELAY',
                'prev_tail_arr_delay', 'real_time_turn_gap', 'inbound_arr_delay_3h']
available_cols = [c for c in display_cols if c in matches.columns]
st.dataframe(matches[available_cols].reset_index(drop=True), use_container_width=True)

flight_idx = st.selectbox("Select flight (row number)", range(len(matches)))
selected = matches.iloc[[flight_idx]]

if st.button("🔍 Predict This Flight", type="primary"):
    flight = selected[FEATURES].copy()

    prob = clf.predict_proba(flight)[:, 1][0]
    delay_min = float(reg.predict(flight)[0])

    shap_vals = explainer.shap_values(flight)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    actual_delayed = int(selected['ARR_DEL15'].iloc[0])
    actual_delay = float(selected['ARR_DELAY'].iloc[0])

    st.markdown("---")
    st.subheader("Prediction Results")

    c1, c2, c3, c4 = st.columns(4)

    if prob >= 0.65:
        c1.error(f"**{prob:.0%}**\n\n⚠️ High Risk")
    elif prob >= 0.40:
        c1.warning(f"**{prob:.0%}**\n\n Medium Risk")
    else:
        c1.success(f"**{prob:.0%}**\n\n✅ Low Risk")

    if delay_min > 15:
        c2.error(f"**+{delay_min:.0f} min**\n\nEstimated delay")
    elif delay_min > 0:
        c2.warning(f"**+{delay_min:.0f} min**\n\nSlight delay")
    else:
        c2.success(f"**{delay_min:.0f} min**\n\nEstimated early")

    if actual_delayed == 1:
        c3.error(f"**+{actual_delay:.0f} min**\n\nActual (delayed)")
    else:
        c3.success(f"**{actual_delay:.0f} min**\n\nActual (on time)")

    cost = max(0, delay_min) * 100.76
    savings = cost * 0.35 if prob >= 0.65 else 0
    c4.metric("Airline Cost", f"${cost:,.0f}",
        delta=f"-${savings:,.0f} early action" if savings > 0 else None)

    model_correct = (prob >= 0.65) == (actual_delayed == 1)
    if model_correct:
        st.success("✅ Model prediction was correct")
    else:
        st.error("❌ Model missed this one")

    st.markdown("---")
    st.subheader("Key Conditions for This Flight")

    key_features = {
        'prev_tail_arr_delay': 'Previous aircraft delay (min)',
        'real_time_turn_gap': 'Turnaround gap (buffer minus prev delay)',
        'inbound_arr_delay_3h': 'Origin avg inbound delay (min)',
        'dest_inbound_arr_delay_3h': 'Destination avg inbound delay (min)',
        'airport_fatigue_index': 'Airport fatigue index (daily stress)',
        'scheduled_turnaround_buffer': 'Scheduled turnaround buffer (min)',
    }

    kf_cols = st.columns(3)
    for i, (feat, label) in enumerate(key_features.items()):
        if feat in selected.columns:
            val = selected[feat].iloc[0]
            val_str = f"{float(val):.1f}" if pd.notna(val) else "N/A"
            kf_cols[i % 3].metric(label, val_str)

    st.markdown("---")
    st.subheader("Why did the model predict this?")

    sv = shap_vals[0]
    feat_shap = sorted(zip(FEATURES, sv, flight.iloc[0]), key=lambda x: abs(x[1]), reverse=True)
    max_shap = abs(feat_shap[0][1]) if feat_shap else 1

    for feat, shap_val, feat_val in feat_shap[:10]:
        direction = "→ Delay" if shap_val > 0 else "→ On-time"
        icon = "🔴" if shap_val > 0 else "🟢"

        try:
            val_float = float(feat_val)
            if np.isnan(val_float):
                val_str = "N/A"
            else:
                val_str = f"{val_float:.1f}"
        except (ValueError, TypeError):
            val_str = str(feat_val)

        impact = abs(shap_val) / max_shap
        st.markdown(f"{icon} **{feat}** = `{val_str}` | SHAP: `{shap_val:+.3f}` {direction}")
        st.progress(float(min(impact, 1.0)))

    st.caption("SHAP: positive = pushes toward delay, negative = pushes toward on-time")