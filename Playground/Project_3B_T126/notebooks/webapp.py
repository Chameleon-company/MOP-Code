import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
from predict_wpf import pipes   # already contains predictions
from groq import Groq

# Streamlit app configuration and layout
# Page Config
st.set_page_config(page_title="Water Pipe Failure Predictor", layout="wide")

# Groq client setup
from groq import Groq

def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception:
        return None

client = get_groq_client()

st.write("Client loaded:", client is not None)

#App title
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
    
# Main display - show number of pipes after filterin 

#Explanation for Fallback
# Pulls basic explanation from selected pipes
def generate_fallback_explanation(pipe_row):
    risk_level = pipe_row.get("risk_level", "Unknown")
    probability = pipe_row.get("failure_probability", "Unknown")
    material = pipe_row.get("material", "Unknown")
    pipe_age = pipe_row.get("pipe_age", "Unknown")
    pipe_length = pipe_row.get("shape__length", "Unknown")

    return f"""
### Risk Explanation
This pipe has been classified as **{risk_level} risk** based on the model's predicted failure probability of **{probability}**.

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
- Risk Level: {pipe_row.get("risk_level", "Unknown")}
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
        legend_title_text="Risk Level",
        width=600,     # increasing width of pie chart
        height=600     # increasing height of pie chart
    )
    st.plotly_chart(fig, use_container_width=False)

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

# Tab 3: LLM Suggestion
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
        col1.metric("Risk Level", pipe_row.get("risk_level", "Unknown"))
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
            "Risk Level": pipe_row.get("risk_level", "Unknown"),
            "Failure Probability": pipe_row.get("failure_probability", "Unknown")
        })

        if st.button("Generate Maintenance Explanation"):
            with st.spinner("Generating explanation..."):
                explanation = generate_llm_explanation(pipe_row)
                st.markdown(explanation)