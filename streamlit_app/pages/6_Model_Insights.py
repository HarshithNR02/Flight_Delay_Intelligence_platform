import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))

st.set_page_config(page_title="Model Insights", layout="wide")
st.title("🔬 Model Insights")

tab1, tab2, tab3 = st.tabs(["SHAP Analysis", "Model Performance", "58% Ceiling"])

with tab1:
    st.subheader("SHAP Global Feature Importance")

    shap_values = np.load(os.path.join(PROJECT_ROOT, 'models/shap_values_50k_v2.npy'))
    with open(os.path.join(PROJECT_ROOT, 'models/feature_list_final.txt')) as f:
        features = f.read().strip().split('\n')

    importance = pd.DataFrame({
        'Feature': features,
        'Mean |SHAP|': np.abs(shap_values).mean(axis=0),
    }).sort_values('Mean |SHAP|', ascending=True).tail(20)

    fig1 = px.bar(importance, x='Mean |SHAP|', y='Feature', orientation='h',
        color='Mean |SHAP|', color_continuous_scale='Reds')
    fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=600)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Finding: Split Importance vs SHAP")
    st.markdown("""
| Feature | SHAP Rank | Split Rank | Insight |
|---------|-----------|------------|---------|
| real_time_turn_gap | #1 | #11 | Engineered feature, true top predictor |
| airport_fatigue_index | #5 | #33 | Rare splits, massive impact |
| ORIGIN | #6 | #1 | Many splits, small individual impact |
| DEST | #9 | #2 | Same pattern as ORIGIN |
""")

    beeswarm_path = os.path.join(PROJECT_ROOT, 'models/shap_beeswarm_v2.png')
    if os.path.exists(beeswarm_path):
        st.subheader("SHAP Beeswarm Plot")
        st.image(beeswarm_path)

    dep_path = os.path.join(PROJECT_ROOT, 'models/shap_dependence_plots_v2.png')
    if os.path.exists(dep_path):
        st.subheader("SHAP Dependence Plots")
        st.image(dep_path)

with tab2:
    st.subheader("Classification Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC", "0.8629")
    col2.metric("Best F1", "0.663 (@ 0.65)")
    col3.metric("Features", "61")

    thresh_data = pd.DataFrame({
        'Threshold': [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        'Precision': [0.528, 0.577, 0.625, 0.672, 0.719, 0.764],
        'Recall': [0.763, 0.725, 0.688, 0.652, 0.616, 0.580],
        'F1': [0.624, 0.643, 0.655, 0.662, 0.663, 0.659],
    })
    fig2 = px.line(thresh_data.melt(id_vars='Threshold', var_name='Metric'),
        x='Threshold', y='value', color='Metric', markers=True,
        labels={'value': 'Score'})
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Regression Model")
    col4, col5, col6 = st.columns(3)
    col4.metric("MAE", "16.82 min")
    col5.metric("±15 min Accuracy", "69.1%")
    col6.metric("±30 min Accuracy", "88.3%")

    iter_data = pd.DataFrame({
        'Iteration': ['V1','V2','V3','V5','V7','V8','Optuna','V9','Final'],
        'AUC': [0.6505,0.6819,0.7888,0.7980,0.8014,0.8572,0.8578,0.8623,0.8629],
    })
    fig3 = px.line(iter_data, x='Iteration', y='AUC', markers=True, text='AUC',
        labels={'AUC': 'ROC-AUC'})
    fig3.update_traces(texttemplate='%{text:.4f}', textposition='top center')
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Why 58% of Delays Are Unpredictable")
    st.markdown("Out of **1,037,836** delayed flights, **602,377 (58%)** had zero prior cascade warning.")

    cause_data = pd.DataFrame({
        'Cause': ['CARRIER_DELAY\n(Maintenance, Crew)', 'NAS_DELAY\n(ATC, Ground Stops)',
                  'LATE_AIRCRAFT\n(Outside Window)', 'WEATHER_DELAY'],
        'Percentage': [42.9, 28.3, 20.8, 7.9],
    })
    fig4 = px.pie(cause_data, values='Percentage', names='Cause',
        color_discrete_sequence=['#e74c3c','#e67e22','#f1c40f','#3498db'])
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**71.2%** of surprise delays come from carrier maintenance and ATC — events that happen at the gate, minutes before departure. No public dataset contains this information.")

    catch_data = pd.DataFrame({
        'Delay Type': ['LATE_AIRCRAFT_DELAY','WEATHER_DELAY','CARRIER_DELAY','NAS_DELAY'],
        'Caught %': [71.5, 8.4, 47.1, 46.3],
        'Missed %': [16.8, 5.3, 59.8, 60.4],
    })
    fig5 = px.bar(catch_data, x='Delay Type', y=['Caught %','Missed %'], barmode='group',
        color_discrete_map={'Caught %':'#2ecc71','Missed %':'#e74c3c'})
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)

    st.info("The model catches 71.5% of cascade delays but only 40% of maintenance/ATC delays. That's not a model failure — it's a data ceiling.")