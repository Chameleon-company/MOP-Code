import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
from predict_wpf import pipes   # already contains predictions

# Streamlit app configuration and layout
st.set_page_config(page_title="Water Pipe Failure Predictor", layout="wide")
st.title("Melbourne Water Pipe Failure Prediction Analysis")

# Sidebar filters
st.sidebar.header("Filters")

# Material filter
material_filter = st.sidebar.multiselect(
    "Material",
    options=pipes["material"].unique()
)

# Pipe age filter (replaces install_year)
age_filter = st.sidebar.slider(
    "Pipe Age",
    int(pipes["pipe_age"].min()),
    int(pipes["pipe_age"].max()),
    (0, 100)
)

# Risk level filter with color legend
st.sidebar.markdown("""
**Risk Levels**  
<span style='color:red;'>● High</span><br>
<span style='color:orange;'>● Medium</span><br>
<span style='color:green;'>● Low</span>
""", unsafe_allow_html=True)
risk_filter = st.sidebar.multiselect(
    "Risk Level",
    ["High", "Medium", "Low"],
    default=["High"]
)
# Apply filters to the dataframe for display and analysis
filtered = pipes.copy()

if material_filter:
    filtered = filtered[filtered["material"].isin(material_filter)]

filtered = filtered[
    (filtered["pipe_age"] >= age_filter[0]) &
    (filtered["pipe_age"] <= age_filter[1])
]

if risk_filter:
    filtered = filtered[filtered["risk_level"].isin(risk_filter)]

# Tabs
tab1, tab2, tab3 = st.tabs(["📍 Risk Map", "📊 Insights & Charts", "🤖 AI Maintenance Reasoning"])
# Tab 1: Risk Map
with tab1:
    st.subheader("Overall Risk Summary")
    # Risk level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pipes", len(filtered))
    col2.metric("High Risk", (filtered["risk_level"] == "High").sum())
    col3.metric("Medium Risk", (filtered["risk_level"] == "Medium").sum())
    col4.metric("Low Risk", (filtered["risk_level"] == "Low").sum())
    
    # Pie chart of risk level distribution
    fig = px.pie(
    filtered,
        names="risk_level",
        title="Risk Level Distribution",
        color="risk_level",
        color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
        )

    fig.update_layout(
        legend_title_text="Risk Level"
    )
    st.plotly_chart(fig)

    # Heatmap of average failure probability by material and risk level
    st.subheader("Risk Heatmap")
    heat = filtered.pivot_table(
        index="material",
        columns="risk_level",
        values="failure_probability",
        aggfunc="mean"
    )
    st.dataframe(heat.style.background_gradient(cmap="Reds"))

    st.subheader("Pipe Network Risk Overview")
    # Sorting options
    sort_col = st.selectbox(
        "Sort pipes by:",
         ["shape__length", "pipe_age", "failure_probability"],
        format_func=lambda x: {
            "shape__length": "Pipe Length (m)",
            "pipe_age": "Pipe Age (years)",
             "failure_probability": "Failure Probability"
        }[x]
    )

    sort_order = st.radio(
        "Sort order:",
        ["Descending", "Ascending"],
        horizontal=True
    )

    top_n = st.selectbox(
        "Show top:",
        [10, 25, 50, 100, "All"]
    )

    # Apply sorting
    df_sorted = filtered.sort_values(
        sort_col,
        ascending=(sort_order == "Ascending")
    )

    if top_n != "All":
        df_sorted = df_sorted.head(top_n)

    # Plot
    st.plotly_chart(
        px.bar(
            df_sorted,
            x="material",
            y="shape__length",
            color="risk_level",
            title=f"Pipe Overview — Sorted by {sort_col}",
            labels={
                "material": "Pipe Material",
                "shape__length": "Pipe Length (m)",
                "risk_level": "Risk Level"
            },
            color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
        )
    )
# Tab 2: Insights & Charts
with tab2:
    col1, col2 = st.columns(2)
    risk_colors = {"High": "red", "Medium": "orange", "Low": "green"}

    with col1:
            fig = px.histogram(
                filtered,
                x="pipe_age",
                color="risk_level",
                title="Failures by Pipe Age",
                labels={
                        "pipe_age": "Pipe Age (years)",
                        "risk_level": "Risk Level",
                },
                color_discrete_map=risk_colors
            )
            fig.update_yaxes(title_text="Pipe Count ")
            st.plotly_chart(fig)

    with col2:
        st.plotly_chart(
            px.bar(
                filtered["material"].value_counts(),
                title="Risk by Material",
                labels={"material": "Material", "value": "Count", "variable": "Variable"},
                color_discrete_map=risk_colors
            )
       )

    display_cols = [
        col for col in ["pipe_id", "material", "pipe_age", "failure_probability", "risk_level"]
        if col in filtered.columns
    ]

    # Rename columns for display
    rename_map = {
        "pipe_id": "Pipe ID",
        "material": "Material",
        "pipe_age": "Pipe Age (years)",
        "failure_probability": "Failure Probability",
        "risk_level": "Risk Level"
    }

    df_display = filtered[display_cols].rename(columns=rename_map)
    st.dataframe(
        df_display.sort_values("Failure Probability", ascending=False),
        use_container_width=True
    )
