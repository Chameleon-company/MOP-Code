import streamlit as st
import pandas as pd
import plotly.express as px
from predict_wpf import predict_pipes, prepare_melbourne_for_model
#import openai

st.set_page_config(page_title="Melbourne Water Pipe Failure Prediction", layout="wide")
st.title("🚰 Melbourne Water Pipe Failure Prediction Analysis")

# Load data
@st.cache_data
def load_data():
    df_raw = pd.read_csv("melbourne_prepared.csv")  # or combine file 2 + 3
    return df_raw

df = load_data()
df_pred = predict_pipes(prepare_melbourne_for_model(df))

# Sidebar filters
st.sidebar.header("Filters")
material = st.sidebar.multiselect("Material", df_pred['material'].unique(), default=None)
risk = st.sidebar.multiselect("Risk Level", ['High','Medium','Low'], default=['High'])
age_range = st.sidebar.slider("Pipe Age", int(df_pred['pipe_age'].min()), int(df_pred['pipe_age'].max()), (40, 80))

filtered = df_pred.copy()
if material: filtered = filtered[filtered['material'].isin(material)]
filtered = filtered[(filtered['pipe_age'] >= age_range[0]) & (filtered['pipe_age'] <= age_range[1])]
if risk: filtered = filtered[filtered['risk_level'].isin(risk)]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📍 Interactive Map", "📊 Analytics", "🔍 Top High-Risk Pipes", "🤖 AI Maintenance Advisor"])

with tab1:
    # Folium map (add lat/long if available, otherwise use synthetic or postcode-based)
    st.info("Add latitude/longitude columns to your data for real mapping.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered, x="pipe_age", color="risk_level", title="Risk by Age")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(filtered['material'].value_counts().head(10), title="Risk by Material")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.dataframe(
        filtered.sort_values("failure_probability", ascending=False)
        [['ASSET_ID', 'material', 'pipe_age', 'PIPE_LENGTH', 'failure_probability', 'risk_level']]
        .head(50), use_container_width=True
    )

with tab4:
    st.subheader("AI-Powered Maintenance Recommendations")
    st.info("This section can be enhanced with an actual LLM integration to provide maintenance advice based on pipe features and risk levels.")