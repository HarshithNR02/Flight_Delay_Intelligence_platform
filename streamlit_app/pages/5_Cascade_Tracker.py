import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))

st.set_page_config(page_title="Cascade Tracker", layout="wide")
st.title("🔗 Cascade Delay Tracker")
st.markdown("Trace how delays propagate through aircraft tail numbers throughout a day.")

@st.cache_data
def load_cascade_data():
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'models/cascade_tail_lookup.parquet'))
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    return df

test = load_cascade_data()

col1, col2, col3 = st.columns(3)
col1.metric("Propagation Rate", "48.1%", help="When incoming aircraft is late")
col2.metric("After 4 Cascades", "86.7% delayed", "Avg 116 min")
col3.metric("Top Super-spreader", "DEN", "3.3M cascade minutes")

st.markdown("---")

st.subheader("Propagation Rate by Incoming Delay Severity")
prop_data = pd.DataFrame({
    'Incoming Delay': ['0-15 min','15-30 min','30-60 min','60-120 min','120+ min'],
    'Rate': [22.4, 43.3, 74.1, 89.7, 87.3],
})
fig1 = px.bar(prop_data, x='Incoming Delay', y='Rate', text_auto='.1f',
    color='Rate', color_continuous_scale='RdYlGn_r',
    labels={'Rate': 'Propagation Rate (%)'})
fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Buffer Absorption — Does Turnaround Time Help?")
buf_data = pd.DataFrame({
    'Buffer Range': ['0-30 min','30-60 min','60-90 min','90-120 min','120+ min'],
    'Delayed %': [91.2, 77.6, 61.8, 44.4, 40.1],
})
fig2 = px.bar(buf_data, x='Buffer Range', y='Delayed %', text_auto='.1f',
    color='Delayed %', color_continuous_scale='RdYlGn_r',
    labels={'Delayed %': '% Delayed with Late Incoming Aircraft'})
fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Trace an Aircraft's Day")

col1, col2 = st.columns(2)
tail = col1.text_input("Tail Number", value="N524AE")
date = col2.date_input("Date", value=pd.to_datetime("2025-07-26"),
    min_value=pd.to_datetime("2025-01-01"), max_value=pd.to_datetime("2025-08-27"))

rotation = test[(test['TAIL_NUM'] == tail) &
                (test['FL_DATE'] == pd.Timestamp(date))].sort_values('DEP_HOUR')

if len(rotation) > 0:
    carrier_name = rotation['OP_UNIQUE_CARRIER'].iloc[0]
    total_delay = rotation['ARR_DELAY'].sum()
    delayed_count = (rotation['ARR_DEL15'] == 1).sum()

    st.markdown(f"**{tail}** | {carrier_name} | {date} | {len(rotation)} flights | "
                f"{delayed_count}/{len(rotation)} delayed | "
                f"Total: {total_delay:.0f} min ({total_delay/60:.1f} hrs)")

    colors = ['#e74c3c' if d == 1 else '#2ecc71' for d in rotation['ARR_DEL15']]
    labels = [f"Leg {i+1}: {r['ORIGIN']}→{r['DEST']} | "
              f"Dep {int(r['DEP_HOUR'])}h | "
              f"{'+'if r['ARR_DELAY']>0 else ''}{r['ARR_DELAY']:.0f}m delay | "
              f"Cascade: {r['cascade_score']:.0f}"
              for i, (_, r) in enumerate(rotation.iterrows())]

    fig3 = go.Figure()
    for i, (label, color) in enumerate(zip(labels, colors)):
        fig3.add_trace(go.Bar(
            x=[abs(rotation.iloc[i]['ARR_DELAY']) + 30],
            y=[label],
            orientation='h',
            marker_color=color,
            showlegend=False,
        ))
    fig3.update_layout(height=max(250, len(rotation)*60),
        xaxis_title='Delay Minutes',
        yaxis_autorange='reversed',
        title=f"Total accumulated delay: {total_delay:.0f} min")
    st.plotly_chart(fig3, use_container_width=True)

    cols_to_show = ['ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY',
                    'cascade_score','real_time_turn_gap','prev_tail_arr_delay']
    available = [c for c in cols_to_show if c in rotation.columns]
    st.dataframe(rotation[available].reset_index(drop=True), use_container_width=True)
else:
    st.info(f"No flights found for {tail} on {date}. Try N524AE on 2025-07-26.")
