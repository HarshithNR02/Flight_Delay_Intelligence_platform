import streamlit as st
import pandas as pd
import plotly.express as px
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))

st.set_page_config(page_title="Airline Rankings", layout="wide")
st.title("🏢 Airline Rankings")

@st.cache_data
def load_carrier_data():
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'dataset/merged_flights_fe_v2.parquet'),
        columns=['FL_DATE', 'OP_UNIQUE_CARRIER', 'ARR_DEL15', 'ARR_DELAY',
                 'cascade_score', 'cascade_delay_minutes', 'airline_cluster_label'])
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    return df[df['FL_DATE'] >= '2025-01-01']

test = load_carrier_data()

carrier_stats = test.groupby('OP_UNIQUE_CARRIER').agg(
    flights=('ARR_DEL15', 'count'),
    delay_rate=('ARR_DEL15', 'mean'),
    avg_delay=('ARR_DELAY', 'mean'),
    severe_pct=('ARR_DELAY', lambda x: (x > 60).mean()),
    avg_cascade=('cascade_delay_minutes', 'mean'),
    cluster=('airline_cluster_label', 'first'),
).reset_index().sort_values('delay_rate', ascending=False)

auc_map = {'AA':0.8483,'AS':0.8256,'B6':0.8556,'DL':0.8533,'F9':0.8526,
           'G4':0.8690,'HA':0.8654,'MQ':0.8522,'NK':0.8529,'OH':0.8641,
           'OO':0.8681,'UA':0.8386,'WN':0.8943,'YX':0.8559}
carrier_stats['auc'] = carrier_stats['OP_UNIQUE_CARRIER'].map(auc_map)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Delay Rate by Carrier")
    fig1 = px.bar(carrier_stats.sort_values('delay_rate'),
        x='delay_rate', y='OP_UNIQUE_CARRIER', orientation='h',
        text_auto='.1%', color='delay_rate', color_continuous_scale='RdYlGn_r',
        labels={'delay_rate': 'Delay Rate', 'OP_UNIQUE_CARRIER': ''})
    fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=500)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Model AUC by Carrier")
    fig2 = px.bar(carrier_stats.dropna(subset=['auc']).sort_values('auc'),
        x='auc', y='OP_UNIQUE_CARRIER', orientation='h',
        text_auto='.4f', color='auc', color_continuous_scale='RdYlGn',
        labels={'auc': 'AUC', 'OP_UNIQUE_CARRIER': ''})
    fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=500)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Detailed Carrier Statistics")
display_df = carrier_stats[['OP_UNIQUE_CARRIER','flights','delay_rate','avg_delay',
                             'severe_pct','avg_cascade','auc','cluster']].copy()
display_df.columns = ['Carrier','Flights','Delay Rate','Avg Delay (min)',
                       'Severe %','Avg Cascade Min','Model AUC','Cluster']
display_df['Delay Rate'] = display_df['Delay Rate'].map('{:.1%}'.format)
display_df['Severe %'] = display_df['Severe %'].map('{:.1%}'.format)
display_df['Avg Delay (min)'] = display_df['Avg Delay (min)'].map('{:.1f}'.format)
display_df['Avg Cascade Min'] = display_df['Avg Cascade Min'].map('{:.1f}'.format)
display_df['Flights'] = display_df['Flights'].map('{:,}'.format)
st.dataframe(display_df, use_container_width=True, hide_index=True)