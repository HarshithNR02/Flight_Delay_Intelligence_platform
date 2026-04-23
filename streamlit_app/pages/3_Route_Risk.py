import streamlit as st
import pandas as pd
import plotly.express as px
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))

st.set_page_config(page_title="Route Risk Scoring", layout="wide")
st.title("🗺️ Route Risk Scoring")

@st.cache_data
def load_routes():
    return pd.read_parquet(os.path.join(PROJECT_ROOT, 'models/route_risk_scores.parquet'))

routes = load_routes()

col1, col2, col3 = st.columns(3)
with col1:
    origin_filter = st.selectbox("Filter by Origin", ['All'] + sorted(routes['ORIGIN'].unique()))
with col2:
    dest_filter = st.selectbox("Filter by Destination", ['All'] + sorted(routes['DEST'].unique()))
with col3:
    tier_filter = st.selectbox("Filter by Risk Tier", ['All', 'High', 'Moderate', 'Low', 'Very Low'])

filtered = routes.copy()
if origin_filter != 'All':
    filtered = filtered[filtered['ORIGIN'] == origin_filter]
if dest_filter != 'All':
    filtered = filtered[filtered['DEST'] == dest_filter]
if tier_filter != 'All':
    filtered = filtered[filtered['risk_tier'] == tier_filter]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Routes Shown", f"{len(filtered):,}")
col2.metric("Avg Delay Rate", f"{filtered['delay_rate'].mean():.1%}")
col3.metric("Avg Risk Score", f"{filtered['risk_score'].mean():.3f}")
col4.metric("Total Flights", f"{filtered['flights'].sum():,}")

st.markdown("---")
st.subheader("Routes by Risk Score")
display = filtered.nlargest(50, 'risk_score')[['ORIGIN','DEST','flights','delay_rate',
    'avg_delay_min','severe_delay_pct','risk_score','risk_tier']].copy()
display.columns = ['Origin','Dest','Flights','Delay Rate','Avg Delay','Severe %','Risk Score','Tier']
display['Delay Rate'] = display['Delay Rate'].map('{:.1%}'.format)
display['Severe %'] = display['Severe %'].map('{:.1%}'.format)
display['Avg Delay'] = display['Avg Delay'].map('{:.1f} min'.format)
display['Risk Score'] = display['Risk Score'].map('{:.3f}'.format)
display['Flights'] = display['Flights'].map('{:,}'.format)
st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("Risk Tier Distribution")
tier_counts = routes['risk_tier'].value_counts().reset_index()
tier_counts.columns = ['Tier', 'Count']
fig = px.pie(tier_counts, values='Count', names='Tier',
    color='Tier', color_discrete_map={
        'Very Low':'#2ecc71','Low':'#f1c40f','Moderate':'#e67e22',
        'High':'#e74c3c','Very High':'#8e44ad'})
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Route Lookup")
c1, c2 = st.columns(2)
lookup_origin = c1.selectbox("Origin", sorted(routes['ORIGIN'].unique()), key='lo')
lookup_dest = c2.selectbox("Destination", sorted(routes['DEST'].unique()), key='ld')

match = routes[(routes['ORIGIN'] == lookup_origin) & (routes['DEST'] == lookup_dest)]
if len(match) > 0:
    r = match.iloc[0]
    st.success(f"**{lookup_origin} → {lookup_dest}**: Risk {r['risk_score']:.3f} ({r['risk_tier']})")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Flights", f"{r['flights']:,}")
    mc2.metric("Delay Rate", f"{r['delay_rate']:.1%}")
    mc3.metric("Avg Delay", f"{r['avg_delay_min']:.1f} min")
    mc4.metric("Severe %", f"{r['severe_delay_pct']:.1%}")
else:
    st.warning(f"Route {lookup_origin} → {lookup_dest} not found (fewer than 50 flights)")