import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Activity Recognition Dashboard", layout="wide")

# Title and info
st.title("MOP AI + IoT – Health Behaviour Monitoring")
st.subheader("Use Case 8: Physical Activity Classification (PAMAP2 Dataset)")
st.markdown("[Click here to view the dataset](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)")
st.markdown("**Developed by:** Anisha Raj")
st.markdown("---")

# Sidebar
st.sidebar.header("Model & Prediction Settings")

model_paths = {
    "Baseline LSTM": "baseline_lstm_model.h5",
    "LSTM + L2": "l2_lstm_model.keras",
    "Stacked LSTM": "l2_stacked_lstm_model.keras",
    "Bidirectional LSTM": "l2_bi_lstm_model.keras",
    "Smaller BiLSTM": "l2_small_bi_lstm_model.keras",
    "GRU": "model_gru.keras",
    "GRU + L2": "l2_gru_model.keras",
    "Bidirectional GRU": "bi_gru_model.keras",
    "Smaller BiGRU(16)": "small_bi_gru_model.keras"
}

model_choice = st.sidebar.selectbox(
    "Choose a Model to Explore",
    ["-- Select a Model --"] + list(model_paths.keys())
)

num_predictions = st.sidebar.slider(
    "Number of Sample Predictions", min_value=1, max_value=10, value=5
)

# Caching model and data
@st.cache_resource
def load_selected_model(path):
    return load_model(path)

@st.cache_data
def load_test_data_and_labels():
    X_test = np.load("X_test_seq_reduced.npy")
    y_test = np.load("y_test_seq_reduced_enc.npy")
    activityIDdict = {
        1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking',
        5: 'running', 6: 'cycling', 7: 'Nordic_walking',
        12: 'ascending_stairs', 13: 'descending_stairs',
        16: 'vacuum_cleaning', 17: 'ironing', 24: 'rope_jumping'
    }
    used_ids = sorted(activityIDdict.keys())
    encoded_to_activity = {i: activityIDdict[aid] for i, aid in enumerate(used_ids)}
    return X_test, y_test, encoded_to_activity

if model_choice != "-- Select a Model --":
    with st.spinner(f"Loading model `{model_choice}` and making predictions..."):
        model = load_selected_model(model_paths[model_choice])
        X_test, y_test, encoded_to_activity = load_test_data_and_labels()
        activity_labels = list(encoded_to_activity.values())
        probs = model.predict(X_test)
        preds = np.argmax(probs, axis=1)
        true = y_test
        accuracy = np.mean(preds == true)

    st.success(f"Model `{model_choice}` loaded and evaluated successfully!")
    st.subheader(f"Selected Model: `{model_choice}`")

    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)
    true = y_test
    accuracy = np.mean(preds == true)

    tab1, tab2, tab3 = st.tabs(["Model Evaluation", "Sample Predictions", "Compare Models"])

    # -------------------- TAB 1: Evaluation --------------------
    with tab1:
        st.markdown(f"### Test Accuracy: **{accuracy * 100:.2f}%**")

        with st.expander("What is a Confusion Matrix?"):
            st.markdown("""
            A **confusion matrix** shows the number of correct and incorrect predictions made by the model compared to the actual values.
            - Rows → Actual classes  
            - Columns → Predicted classes  
            - Diagonal cells = correct predictions
            """)

        # Confusion Matrix
        cm = confusion_matrix(true, preds)
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=activity_labels, yticklabels=activity_labels, cmap='Blues', ax=ax_cm)
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

        with st.expander("What is a Classification Report?"):
            st.markdown("""
            The classification report includes:
            - **Precision**: Correct predictions / All predicted
            - **Recall**: Correct predictions / All actual
            - **F1-score**: Harmonic mean of precision and recall
            - **Support**: Number of samples per class
            """)

        report = classification_report(true, preds, target_names=activity_labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.background_gradient(cmap='YlGnBu'))

        csv_report = df_report.to_csv().encode('utf-8')
        st.download_button(
            label="Download Classification Report as CSV",
            data=csv_report,
            file_name=f"{model_choice}_classification_report.csv",
            mime='text/csv'
        )

    # -------------------- TAB 2: Predictions --------------------
    with tab2:
        st.subheader(f"Sample Predictions (Top {num_predictions})")
        with st.expander("How is prediction done?"):
            st.markdown("""
            - We randomly select `n` test samples.
            - The model predicts the most likely activity class.
            - You can compare this to the true class label.
            """)

        sample_indices = np.random.choice(len(X_test), num_predictions, replace=False)
        sample_preds = np.argmax(probs[sample_indices], axis=1)
        sample_true = true[sample_indices]

        for i in range(num_predictions):
            st.markdown(f"**Sample {i+1}**")
            st.markdown(f"- 🟢 True Label      : `{encoded_to_activity[sample_true[i]]}`")
            st.markdown(f"- 🔵 Predicted Label : `{encoded_to_activity[sample_preds[i]]}`")
            st.markdown("---")

    # -------------------- TAB 3: Comparison --------------------
# -------------------- TAB 3: Comparison --------------------
    with tab3:
        st.subheader("Test Accuracy of All Trained Models")
        
        with st.expander("What are these models?"):
            st.markdown("""
            - **LSTM**: Long Short-Term Memory (good at sequences)  
            - **GRU**: Gated Recurrent Unit (lighter version of LSTM)  
            - **Bidirectional**: Looks at sequence both forward and backward  
            - **+ L2**: Regularization to reduce overfitting  
            - **Smaller**: Reduced parameters to avoid complexity  
            """)

        model_scores = {
            "Baseline LSTM": 39.86,
            "LSTM + L2": 50.34,
            "Stacked LSTM": 57.50,
            "Bidirectional LSTM": 47.81,
            "Smaller BiLSTM": 55.63,
            "GRU": 62.91,
            "GRU + L2": 59.56,
            "Bidirectional GRU": 60.00,
            "Smaller BiGRU(16)": 52.88
        }

        df_scores = pd.DataFrame(model_scores.items(), columns=["Model", "Test Accuracy (%)"])
        df_scores = df_scores.sort_values(by="Test Accuracy (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df_scores)

        # Highlight best model in green
        max_acc = df_scores["Test Accuracy (%)"].max()
        colors = ['seagreen' if acc == max_acc else 'skyblue' for acc in df_scores["Test Accuracy (%)"]]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(df_scores["Model"], df_scores["Test Accuracy (%)"], color=colors)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")

        # Annotate each bar with accuracy value
        for i, (model, acc) in enumerate(zip(df_scores["Model"], df_scores["Test Accuracy (%)"])):
            ax.text(acc + 0.5, i, f"{acc:.2f}%", va='center')

        plt.tight_layout()
        st.pyplot(fig)


else:
    st.info("Select a model from the sidebar to begin.")
