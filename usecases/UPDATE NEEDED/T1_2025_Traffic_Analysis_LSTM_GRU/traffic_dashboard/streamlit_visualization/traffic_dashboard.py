
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#  Streamlit Page Configuration

st.set_page_config(page_title=" Traffic Forecast Dashboard", layout="wide")
st.title(" Traffic Forecasting Dashboard - Melbourne")


#  Load and Cache Data

@st.cache_data
def load_data():
    df_pred = pd.read_csv(
        r"C:\Users\USER\Chameleon AI-IoT\MOP-Code\artificial-intelligence\Traffic Analysis\LSTM-GRU Vehicle Traffic\traffic_dashboard\streamlit_visualization\traffic_dashboard_predictions_filtered.csv"
    )
    df_meta = pd.read_csv(
        r"C:\Users\USER\Chameleon AI-IoT\MOP-Code\artificial-intelligence\Traffic Analysis\LSTM-GRU Vehicle Traffic\traffic_dashboard\streamlit_visualization\lstm_ready_traffic_data.csv"
    )
    df_map = pd.read_csv(
        r"C:\Users\USER\Chameleon AI-IoT\MOP-Code\artificial-intelligence\Traffic Analysis\LSTM-GRU Vehicle Traffic\traffic_dashboard\streamlit_visualization\suburb_location_mapping.csv"
    )

    # Merge metadata with suburb-location mapping
    df_meta = df_meta.merge(df_map, on=["suburb_encoded", "location_encoded"], how="left")

    # Combine prediction and location data
    df_combined = pd.concat(
        [df_pred.reset_index(drop=True), df_meta[["suburb", "location"]].reset_index(drop=True)],
        axis=1
    )
    return df_combined

# Load the combined DataFrame
df = load_data()


#  Suburb and Location Filters

if "suburb" in df.columns and "location" in df.columns:
    all_suburbs = sorted(df["suburb"].dropna().unique())
    selected_suburb = st.sidebar.selectbox("Select Suburb", all_suburbs)

    filtered_df = df[df["suburb"] == selected_suburb]
    all_locations = sorted(filtered_df["location"].dropna().unique())
    selected_location = st.sidebar.selectbox("Select Location", all_locations)

    df = filtered_df[filtered_df["location"] == selected_location]


#  Model Selection Dropdown

model_cols = [col for col in df.columns if col.lower().startswith("predicted")]
model_map = {col: col.replace("Predicted_", "") for col in model_cols}

if not df.empty and model_cols:
    selected_model_col = st.sidebar.selectbox("Select Model", model_map.keys())
    selected_model_label = model_map[selected_model_col]

    
    #  Range Time Step Slider
    
    samples = st.sidebar.slider(
        "Number of Time Steps to Display",
        min_value=100,
        max_value=300,
        value=200,
        step=10
    )

    
    # Model Evaluation Metrics
    
    df_eval = df[["Actual", selected_model_col]].dropna().astype(float).iloc[:samples]
    y_true = df_eval["Actual"].values
    y_pred = df_eval[selected_model_col].values

    if len(y_true) > 0 and len(y_pred) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        st.subheader("📊 Evaluation Metrics")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("R²", f"{r2:.4f}")

        
        #  Actual vs Predicted Plot
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_true, label="Actual", linewidth=2, color='tab:blue')
        ax.plot(y_pred, label=f"Predicted ({selected_model_label})", linestyle="--", linewidth=2, color='tab:orange')
        ax.set_title(f"Actual vs Predicted - {selected_model_label}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Traffic Volume (scaled)", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
    else:
        st.error("No valid data found after removing missing values for the selected model.")
else:
    st.warning("No data available for the selected filters or prediction model.")
