import streamlit as st
import pandas as pd
import plotly.express as px
from predict_wpf import pipes
from groq import Groq

st.set_page_config(page_title="Melbourne Water Pipe Risk", layout="wide")
# Groq client setup
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception:
        return None

client = get_groq_client()
st.title("Melbourne Water Pipe Failure Prediction Analysis")

# Debug info - confirm data loaded correctly
st.sidebar.write(f"Total pipes loaded: **{len(pipes)}**")

# Sidebar Filters
st.sidebar.header("Filters")

material_filter = st.sidebar.multiselect(
    "**Material**", 
    options=sorted(pipes["material"].dropna().unique()),
    default=[]
)
# Pipe age filter with dynamic range based on data 
age_filter = st.sidebar.slider(
    "Pipe Age (years)",
    int(pipes["pipe_age"].min()),
    int(pipes["pipe_age"].max()),
    (0, int(pipes["pipe_age"].max()))
)
# Risk level filter with color legend
st.sidebar.markdown("""
**Risk Levels**  
<span style='color:red;'>● High</span><br>
<span style='color:orange;'>● Medium</span><br>
<span style='color:green;'>● Low</span>
""", unsafe_allow_html=True)
risk_filter = st.sidebar.multiselect(
    "**Risk Level Types**", 
    options=sorted(pipes["RISK_LEVEL"].dropna().unique()),
    default=sorted(pipes["RISK_LEVEL"].dropna().unique())  # Show all by default
)

# Applying filters
filtered = pipes.copy()

if material_filter:
    filtered = filtered[filtered["material"].isin(material_filter)]

filtered = filtered[
    (filtered["pipe_age"] >= age_filter[0]) & 
    (filtered["pipe_age"] <= age_filter[1])
]

if risk_filter:
    filtered = filtered[filtered["RISK_LEVEL"].isin(risk_filter)]

# Main display
st.write(f"**Showing {len(filtered)} pipes** out of {len(pipes)} total")

if len(filtered) == 0:
    st.error("No pipes match your current filters. Try broadening the filters (especially Risk Level).")
    st.stop()
    
#Explanation for Fallback
# Pulls basic explanation from selected pipes
def generate_fallback_explanation(pipe_row):
    RISK_LEVEL = pipe_row.get("RISK_LEVEL", "Unknown")
    probability = pipe_row.get("failure_probability", "Unknown")
    material = pipe_row.get("material", "Unknown")
    pipe_age = pipe_row.get("pipe_age", "Unknown")
    pipe_length = pipe_row.get("shape__length", "Unknown")

    return f"""
### Risk Explanation
This pipe has been classified as **{RISK_LEVEL} risk** based on the model's predicted failure probability of **{probability}**.

Key available factors include:
- Material: {material}
- Pipe age: {pipe_age}
- Pipe length: {pipe_length}

### Maintenance Recommendation
For high-risk pipes, prioritise inspection, condition assessment, and possible preventative maintenance. For medium-risk pipes, schedule monitoring and review. For low-risk pipes, continue routine maintenance.

### Priority Reasoning
This recommendation is based on the model output and available pipe attributes. The prediction should support maintenance planning but should not replace engineering judgement.
"""

# Explanation using LLM
def generate_llm_explanation(pipe_row):
    if client is None:      
        return generate_fallback_explanation(pipe_row)  #Uses fallback if no APIKey is detected

#Generates prompt using charateristics of dataset
    prompt = f"""
You are an infrastructure maintenance assistant for a water utility.

The machine learning model has already predicted this pipe's failure risk.
Your task is to clearly explain the model output and provide realistic maintenance recommendations based on the pipe attributes provided.

Pipe information:
- Pipe ID: {pipe_row.get("pipe_id", "Unknown")}
- Risk Level: {pipe_row.get("RISK_LEVEL", "Unknown")}
- Failure Probability: {pipe_row.get("failure_probability", "Unknown")}
- Material: {pipe_row.get("material", "Unknown")}
- Pipe Age: {pipe_row.get("pipe_age", "Unknown")}
- Pipe Length: {pipe_row.get("shape__length", "Unknown")}

Format the response in clean Markdown using these exact headings:

### Risk Explanation
- Explain why the pipe may be considered this risk level.
- Refer to the failure probability and important pipe characteristics.
- Include brief domain reasoning where appropriate (e.g. ageing infrastructure, material deterioration, long pipe sections, maintenance history implications).

### Maintenance Recommendation
- Suggest 2-3 realistic preventative or investigative maintenance actions.
- Explain why those actions may help reduce operational risk.
- Recommendations should sound practical for a water utility environment.

### Priority Reasoning
- Explain whether this pipe should be prioritised compared to lower-risk assets.
- Discuss potential operational consequences of failure where relevant.
- Mention that the recommendation supports, but does not replace, engineering judgement.

Rules:
- Use Markdown formatting.
- Use bullet points where appropriate.
- Be descriptive but still concise and readable.
- Do not invent missing data.
- Do not claim certainty.
"""

#Groq API Call
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq API failed: {e}")
        return generate_fallback_explanation(pipe_row)
    #except Exception:
     #   return generate_fallback_explanation(pipe_row) #additional lines to use fallback if problems arise.


# Tabs for different views - Overview, High Risk Pipes, 
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📍 High Risk Pipes", "🤖 AI Maintenance Reasoning"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pipes", len(filtered))
    col2.metric("High Risk", (filtered["RISK_LEVEL"] == "HIGH").sum())
    col3.metric("Medium Risk", (filtered["RISK_LEVEL"] == "MEDIUM").sum())
    col4.metric("Low Risk", (filtered["RISK_LEVEL"] == "LOW").sum())

    fig = px.pie(
         filtered,
        names="RISK_LEVEL",
        title="Risk Level Distribution",
        color="RISK_LEVEL",
        color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
        width=600,  # increase width of pie chart
        height=600  # increase height of pie chart
    )
    fig.update_layout(
    legend_title="Risk Level"
    )
    st.plotly_chart(fig, use_container_width=False)

    # Bar chart of risk by material - only show top 10 materials for clarity
    material_risk = filtered.groupby("material")["RISK_LEVEL"].value_counts().unstack().fillna(0)
    fig2 = px.bar(material_risk.head(10), 
                      title="Risk Distribution by Material (Top 10)",
                      labels={"material": "Material", "value": "Pipe Count"},
                      color_discrete_map={"HIGH":"red", "MEDIUM":"orange", "LOW":"green"})
    st.plotly_chart(fig2, use_container_width=True)
    col1, col2 = st.columns(2)
    
    # Histogram of pipe age by risk level
    fig3 = px.histogram(filtered, x="pipe_age", color="RISK_LEVEL",
                            title="Pipe Age Distribution by Risk Level",
                            color_discrete_map={"HIGH":"red", "MEDIUM":"orange", "LOW":"green"},
                            nbins=30,
                            labels={"pipe_age": "Pipe Age (years)"})
    fig3.update_yaxes(title="Pipe count")

    st.plotly_chart(fig3, use_container_width=True)

    #Box Plot of pipe age by risk level
    fig4 = px.box(filtered, x="RISK_LEVEL", y="pipe_age", 
                      title="Pipe Age by Risk Level",
                      color="RISK_LEVEL",
                      labels={"RISK_LEVEL": "Risk Level", "pipe_age": "Pipe Age (years)"},
                      color_discrete_map={"HIGH":"red", "MEDIUM":"orange", "LOW":"green"})
    st.plotly_chart(fig4, use_container_width=True)

    # Additional insights - Risk Score distribution if available
    st.subheader("Additional Risk Insights")
    fig5 = px.histogram(filtered, x="RISK_SCORE", 
                        labels={"RISK_SCORE": "Risk Score"}, 
                        title="Risk Score Distribution", 
                        color_discrete_sequence=["steelblue"],
                        nbins=20)
    fig5.update_yaxes(title="Pipe Count")                    
    st.plotly_chart(fig5, use_container_width=True)

    # Average Risk Score by Material - only show top 10 materials for clarity
    avg_risk = (
        filtered.groupby("material")["RISK_SCORE"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig6 = px.bar(
        avg_risk,
        x="material",
        y="RISK_SCORE",
        title="Average Risk Score by Material (Top 10)",
        color_discrete_sequence=["steelblue"],
        labels={"material": "Material"}
    )

    fig6.update_yaxes(title="Average Risk Score")
    st.plotly_chart(fig6, use_container_width=True)

with tab2:
    st.subheader("📈 Detailed Risk Insights & Analysis")

    # charts side by side in a row
    col1, col2 = st.columns(2)

    with col1:
        # 1. Pipe Age Distribution by Risk Level
        fig_age = px.histogram(
            filtered,
            x="pipe_age",
            color="RISK_LEVEL",
            title="Pipe Age Distribution by Risk Level",
            labels={"pipe_age": "Pipe Age (years)"},
            color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
            nbins=40
        )
        fig_age.update_layout(bargap=0.1)
        fig_age.update_yaxes(title="Number of Pipes")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        # Risk by material - showing top 15 materials 
        material_risk = (
            filtered.groupby("material")["RISK_LEVEL"]
            .value_counts()
            .unstack(fill_value=0)
        )
        
        # Safe sorting - handle different possible column names
        sort_col = None
        for col in ["HIGH", "High", "high"]:
            if col in material_risk.columns:
                sort_col = col
                break
        
        if sort_col:
            material_risk = material_risk.sort_values(sort_col, ascending=False).head(15)
        else:
            material_risk = material_risk.head(15)

        fig_material = px.bar(
            material_risk,
            title="Risk Level Breakdown by Material (Top 15)",
            labels={"material": "Material", "value": "Number of Pipes"},
            color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        )
        st.plotly_chart(fig_material, use_container_width=True)

    # Top 100 highest risk pipes table
    st.subheader("Top 100 Highest Risk Pipes")

    display_cols = [col for col in [
        "ASSET_ID", "MAIN_NAME", "material", "pipe_age",
        "failure_probability", "RISK_SCORE", "RISK_LEVEL", "RECOMMENDED_ACTION"
    ] if col in filtered.columns]

    if display_cols:
        df_top = filtered.sort_values("RISK_SCORE", ascending=False)[display_cols].head(100).copy()

        # Renaming columns
        rename_dict = {
            "ASSET_ID": "Asset ID",
            "MAIN_NAME": "Location",
            "material": "Material",
            "pipe_age": "Pipe Age (years)",
            "failure_probability": "Failure Probability",
            "RISK_SCORE": "Risk Score",
            "RISK_LEVEL": "Risk Level",
            "RECOMMENDED_ACTION": "Recommended Action"
        }

        df_top = df_top.rename(columns=rename_dict)

        # Format Failure Probability as percentage
        if "Failure Probability" in df_top.columns:
            df_top["Failure Probability"] = df_top["Failure Probability"].apply(
                lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
            )

        # Risk Level Highlighting
        def highlight_risk(val):
            if val == "HIGH":
                return 'background-color: #ffcccc; color: red; font-weight: bold'
            elif val == "MEDIUM":
                return 'background-color: #ffe6cc; color: #ff8c00; font-weight: bold'
            elif val == "LOW":
                return 'background-color: #ccffcc; color: green'
            return ''

        styled_df = df_top.style.map(highlight_risk, subset=["Risk Level"])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No data available to display.")

    # Additional insights - average risk score by material and age vs length scatter
    st.subheader("Additional Insights")

    col3, col4 = st.columns(2)

    with col3:
        # Average Risk Score by Material
        avg_score = filtered.groupby("material")["RISK_SCORE"].mean().sort_values(ascending=False).head(12)
        fig_avg = px.bar(
            avg_score,
            title="Average Risk Score by Material",
            labels={"value": "Average Risk Score", "material": "Material", "variable": "Variable"},
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_avg, use_container_width=True)

    with col4:
        # Age vs length scatter
        fig_scatter = px.scatter(
            filtered,
            x="pipe_age",
            y="shape__length",
            color="RISK_LEVEL",
            hover_data=["ASSET_ID", "MAIN_NAME"],
            title="Pipe Age vs Length by Risk Level",
            labels={
                "pipe_age": "Pipe Age (years)",
                "shape__length": "Pipe Length (m)",
                "RISK_LEVEL": "Risk Level"
            },
            color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    material_pct = material_risk.div(material_risk.sum(axis=1), axis=0) * 100

    fig_material_pct = px.bar(
        material_pct,
        title="Risk Level Percentage Breakdown by Material (Top 15)",
        labels={"value": "Percentage (%)", "material": "Material", "RISK_LEVEL": "Risk Level"},
        color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
        barmode="stack"
    )

    fig_material_pct.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_material_pct, use_container_width=True)


# Tab 3: LLM-based explanations and maintenance recommendations
with tab3:
    st.subheader("AI Maintenance Reasoning")

    st.write(
        "Select a pipe to generate an explanation of its predicted risk level "
        "and a suggested maintenance action."
    )

    if client is None:
        st.warning(
            "No OpenAI API key found. The app will use fallback rule-based explanations."
        )

    if filtered.empty:
        st.warning("No pipes available based on the current filters.")
    else:
        selected_pipe_id = st.selectbox(
            "Select Pipe ID",
            filtered["pipe_id"].astype(str).unique()
        )

        pipe_row = filtered[
            filtered["pipe_id"].astype(str) == selected_pipe_id
        ].iloc[0]

        st.markdown("### Selected Pipe Details")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", pipe_row.get("RISK_LEVEL", "Unknown"))
        col2.metric(
            "Failure Probability",
            round(float(pipe_row.get("failure_probability", 0)), 4)
        )
        col3.metric("Pipe Age", pipe_row.get("pipe_age", "Unknown"))

        st.write({
            "Pipe ID": pipe_row.get("pipe_id", "Unknown"),
            "Material": pipe_row.get("material", "Unknown"),
            "Pipe Age": pipe_row.get("pipe_age", "Unknown"),
            "Pipe Length": pipe_row.get("shape__length", "Unknown"),
            "Risk Level": pipe_row.get("RISK_LEVEL", "Unknown"),
            "Failure Probability": pipe_row.get("failure_probability", "Unknown")
        })

        if st.button("Generate Maintenance Explanation"):
            with st.spinner("Generating explanation..."):
                explanation = generate_llm_explanation(pipe_row)
                st.markdown(explanation)