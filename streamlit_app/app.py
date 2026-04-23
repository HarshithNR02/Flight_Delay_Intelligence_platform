import streamlit as st
import pandas as pd
import plotly.express as px
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..'))

st.set_page_config(
    page_title="Flight Delay Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("✈️ Flight Delay Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("**18.2M US Domestic Flights**")
st.sidebar.markdown("AUC 0.8629 | MAE 16.82 min")
st.sidebar.markdown("61 Engineered Features")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Harshith Nerlikere Ramesh](https://github.com/HarshithNR02)")
st.sidebar.markdown("UMass Dartmouth")

st.title("Flight Delay Intelligence Platform")
st.markdown("### Predict, explain, and quantify US domestic flight delays")

@st.cache_data
def load_overview_stats():
    df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, 'dataset/merged_flights_fe_v2.parquet'),
        columns=['FL_DATE','ARR_DEL15','ARR_DELAY','OP_UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','MONTH']
    )
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    return df

df = load_overview_stats()
test = df[df['FL_DATE'] >= '2025-01-01']

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flights", f"{len(df):,}")
col2.metric("Test Set", f"{len(test):,}")
col3.metric("Delay Rate", f"{test['ARR_DEL15'].mean():.1%}")
col4.metric("Model AUC", "0.8629")

col5, col6, col7, col8 = st.columns(4)
col5.metric("MAE", "16.82 min")
col6.metric("Features", "61")
col7.metric("Airlines", f"{df['OP_UNIQUE_CARRIER'].nunique()}")
col8.metric("Annual Savings", "$2.78B")

st.markdown("---")

st.subheader("Delay Rate by Month")
monthly = test.groupby('MONTH').agg(
    delay_rate=('ARR_DEL15','mean'),
    flights=('ARR_DEL15','count'),
).reset_index()
monthly['MONTH'] = monthly['MONTH'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug'})
fig1 = px.bar(monthly, x='MONTH', y='delay_rate', text_auto='.1%',
    labels={'delay_rate':'Delay Rate','MONTH':''},
    color='delay_rate', color_continuous_scale='RdYlGn_r')
fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Delay Rate by Hour of Day")
hourly = test.groupby('DEP_HOUR').agg(
    delay_rate=('ARR_DEL15','mean'),
).reset_index()
fig2 = px.line(hourly, x='DEP_HOUR', y='delay_rate',
    labels={'delay_rate':'Delay Rate','DEP_HOUR':'Departure Hour'}, markers=True)
fig2.update_layout(height=350, yaxis_tickformat='.0%')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Delay Rate by Carrier (Test Set)")
carrier_stats = test.groupby('OP_UNIQUE_CARRIER').agg(
    delay_rate=('ARR_DEL15','mean'),
).reset_index().sort_values('delay_rate', ascending=True)
fig3 = px.bar(carrier_stats, x='delay_rate', y='OP_UNIQUE_CARRIER', orientation='h',
    text_auto='.1%', labels={'delay_rate':'Delay Rate','OP_UNIQUE_CARRIER':''},
    color='delay_rate', color_continuous_scale='RdYlGn_r')
fig3.update_layout(showlegend=False, coloraxis_showscale=False, height=450)
st.plotly_chart(fig3, use_container_width=True)