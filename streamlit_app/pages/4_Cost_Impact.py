import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))

st.set_page_config(page_title="Cost Impact", layout="wide")
st.title("💰 Cost-Benefit Analysis")
st.markdown("Source: Airlines for America (A4A), 2024 DOT Form 41 data — $100.76/min")

col1, col2, col3 = st.columns(3)
col1.metric("Total Delay Cost (No Model)", "$7.75B", help="8-month test period")
col2.metric("Total Cost With Model", "$5.90B")
col3.metric("Net Savings", "$1.85B", "$2.78B annualized")

col4, col5, col6 = st.columns(3)
col4.metric("Per Caught Delay", "$2,998 saved")
col5.metric("Per False Alarm", "$250 cost")
col6.metric("ROI", "$30.60 per $1 spent")

st.markdown("---")
st.subheader("Financial Impact by Prediction Category")

fig = go.Figure(data=[go.Bar(
    x=['Caught Delays\n638,975', 'False Alarms\n250,079',
       'Missed Delays\n398,861', 'Correct On-time\n3,231,211'],
    y=[1915517155, -62519750, 0, 0],
    marker_color=['#2ecc71','#e74c3c','#95a5a6','#3498db'],
    text=['$1.92B saved','$62.5M cost','No savings','No cost'],
    textposition='outside',
)])
fig.update_layout(yaxis_title='Dollar Impact', height=450, yaxis_tickformat='$,.0f')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Sensitivity Analysis")

rates = [0.15, 0.20, 0.25, 0.30, 0.35, 0.50]
tp_cost = 5472906156
fp_cost = 62519750
sens = pd.DataFrame({
    'Savings Rate': [f"{r:.0%}" for r in rates],
    'TP Savings': [f"${tp_cost*r:,.0f}" for r in rates],
    'Net Savings (8mo)': [f"${tp_cost*r - fp_cost:,.0f}" for r in rates],
    'Annualized': [f"${(tp_cost*r - fp_cost)*1.5:,.0f}" for r in rates],
})
st.dataframe(sens, use_container_width=True, hide_index=True)
st.caption("**Conservative 35% proactive savings rate used as baseline. Range based on published research (15%-50%).**")