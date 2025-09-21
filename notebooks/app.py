import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class EEG_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(14, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(64, 32)
        self.fc2   = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
class EEG_LSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.fc2  = nn.Linear(32, num_classes)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:,-1,:]))
        return self.fc2(out)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_per_sample(sample):
    mean = sample.mean(axis=1, keepdims=True)
    std  = sample.std(axis=1, keepdims=True)
    std[std==0] = 1
    return (sample - mean)/std

def load_model(model_type, ckpt_path):
    if model_type=="CNN":
        m = EEG_CNN().to(DEVICE)
    else:
        m = EEG_LSTM().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m

def predict(sample, model, model_type):
    x = normalize_per_sample(sample).astype(np.float32)
    if model_type=="CNN":
        xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)      # [1,14,T]
    else:
        xt = torch.from_numpy(x).unsqueeze(0).permute(0,2,1).to(DEVICE)  # [1,T,14]
    with torch.no_grad():
        probs = torch.softmax(model(xt), dim=1).detach().cpu().numpy()[0]
    return int(np.argmax(probs)), float(probs[1])
st.title("🧠 EEG Stress-Level Demo")

model_type = st.radio("Choose Model:", ["CNN","LSTM"])
ckpt = st.text_input(
    "Model checkpoint path",
    value="/content/drive/MyDrive/cnn_model.pth" if model_type=="CNN" else "/content/drive/MyDrive/lstm_model.pth",
    help="Full path to your trained weights (.pth)"
)

uploaded = st.file_uploader("Upload EEG sample (.npy)", type=["npy"])
st.caption("Expected shape: (14, T). We z-score per channel over time.")

if uploaded and ckpt:
    try:
        arr = np.load(uploaded)
        assert arr.ndim==2 and arr.shape[0]==14, "Sample must be (14, T)."
        model = load_model(model_type, ckpt)
        pred, prob = predict(arr, model, model_type)
        label = "High Stress" if pred==1 else "Low Stress"
        st.metric("Prediction", label)
        st.progress(prob)
    except Exception as e:
        st.error(f"Error: {e}")

# -------------------- Week 9: Dashboard --------------------
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("---")
st.header("📊 Model Optimization Dashboard")

# Loads results CSV
results_path = "/content/drive/MyDrive/week9_results_all.csv"
df = pd.read_csv(results_path)

# ---- Plot 1: Accuracy vs Latency ----
fig1, ax1 = plt.subplots(figsize=(6,4))
for model in ["CNN", "LSTM"]:
    subset = df[df["Model"] == model]
    ax1.scatter(subset["Latency_ms"], subset["Accuracy"], label=model)
    for _, row in subset.iterrows():
        ax1.text(row["Latency_ms"], row["Accuracy"], row["Variant"], fontsize=8)
ax1.set_xlabel("Latency (ms)")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy vs Latency")
ax1.legend()
ax1.grid()
st.pyplot(fig1)

# ---- Plot 2: F1 vs Model Size ----
fig2, ax2 = plt.subplots(figsize=(6,4))
for model in ["CNN", "LSTM"]:
    subset = df[df["Model"] == model]
    ax2.scatter(subset["Size_MB"], subset["F1"], label=model)
    for _, row in subset.iterrows():
        ax2.text(row["Size_MB"], row["F1"], row["Variant"], fontsize=8)
ax2.set_xlabel("Model Size (MB)")
ax2.set_ylabel("F1 Score")
ax2.set_title("F1 vs Model Size")
ax2.legend()
ax2.grid()
st.pyplot(fig2)

