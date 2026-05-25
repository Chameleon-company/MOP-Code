import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from cv_module.real_detector import extract_crack_features
from llm_module.report_engine import generate_report
from llm_module.reasoning_engine import generate_reasoning
from llm_module.chatbot import chatbot_response

st.set_page_config(page_title="Crack Detection", layout="wide")

st.title("🏗️ AI-Based Crack Detection & Analysis System")

# ---------- SESSION ----------
if "latest_response" not in st.session_state:
    st.session_state.latest_response = ""

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader("Upload Crack Image", type=["jpg", "png"])

data = None

if uploaded_file:
    uploaded_file.seek(0)
    data = extract_crack_features(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file.seek(0)
        st.image(uploaded_file, caption="Original Image", use_container_width=True)

    with col2:
        st.image(data["processed_image"], caption="Detected Cracks", use_container_width=True)

    # ---------- METRICS ----------
    st.markdown("---")
    st.subheader("📊 Crack Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Length (m)", data["crack_length"])
    col2.metric("Width (mm)", data["crack_width"])
    col3.metric("Severity", data["severity"])

    # ---------- MODERN VISUALS ----------
    st.markdown("---")
    st.subheader("📊 Advanced Visual Analysis")

    col1, col2 = st.columns(2)

    # 🔥 BAR CHART
    with col1:
        fig = px.bar(
            x=["Length (m)", "Width (mm)"],
            y=[data["crack_length"], data["crack_width"]],
            color=["Length", "Width"],
            text_auto=True,
            title="Crack Dimensions"
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 🔥 GAUGE
    with col2:
        severity_map = {"Low": 1, "Medium": 2, "High": 3}
        severity_value = severity_map[data["severity"]]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=severity_value,
            title={'text': "Severity"},
            gauge={
                'axis': {'range': [0, 3]},
                'bar': {'color': "red" if severity_value == 3 else "orange" if severity_value == 2 else "green"},
                'steps': [
                    {'range': [0, 1], 'color': "green"},
                    {'range': [1, 2], 'color': "orange"},
                    {'range': [2, 3], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- DONUT ----------
    st.subheader("📈 Damage Distribution")

    fig = px.pie(
        names=["Crack", "Remaining"],
        values=[data["crack_width"], max(20 - data["crack_width"], 1)],
        hole=0.6
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- CRACK GROWTH ----------
    st.subheader("📈 Crack Growth Prediction")

    time = np.arange(0, 10, 1)
    growth = data["crack_width"] * (1 + 0.1 * time)

    fig = px.line(
        x=time,
        y=growth,
        labels={"x": "Time", "y": "Width (mm)"},
        title="Predicted Growth"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- RADAR ----------
    st.subheader("🕸 Structural Radar")

    categories = ["Length", "Width", "Severity"]
    values = [
        data["crack_length"],
        data["crack_width"],
        {"Low":1,"Medium":2,"High":3}[data["severity"]]
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- HEATMAP ----------
    st.subheader("🔥 Damage Intensity")

    heat_data = np.random.rand(10, 10) * data["crack_width"]

    fig = px.imshow(
        heat_data,
        color_continuous_scale="Reds"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- RISK ----------
    st.subheader("⚠️ Risk Indicator")

    if data["severity"] == "High":
        st.progress(100)
        st.error("Critical Risk")
    elif data["severity"] == "Medium":
        st.progress(60)
        st.warning("Moderate Risk")
    else:
        st.progress(30)
        st.success("Low Risk")

    # ---------- INSIGHTS ----------
    st.subheader("📌 Key Insights")

    col1, col2, col3 = st.columns(3)
    col1.info(f"📏 Length: {data['crack_length']} m")
    col2.warning(f"📐 Width: {data['crack_width']} mm")
    col3.error(f"⚠️ Severity: {data['severity']}")

# ---------- ACTIONS ----------
if data:
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Generate Report"):
            st.markdown(generate_report(data))

    with col2:
        if st.button("🧠 Explain Crack"):
            st.info(generate_reasoning(data))

# ---------- CHATBOT ----------
if data:
    st.markdown("---")
    st.subheader("💬 Ask AI")

    user_input = st.text_input("Ask about the crack:")

    if user_input and user_input != st.session_state.last_input:
        response = chatbot_response(user_input, data)
        st.session_state.latest_response = response
        st.session_state.last_input = user_input

    if st.session_state.latest_response:
        st.markdown("### 🧑 Question")
        st.write(st.session_state.last_input)

        st.markdown("### 🤖 Answer")
        st.write(st.session_state.latest_response)