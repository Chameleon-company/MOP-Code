# Sleep Stage Prediction & Sleep Disorder Detection

## 📌 Project Overview
This project develops a deep learning-based system to predict sleep stages and detect sleep-related disorders (e.g., insomnia, apnea) using physiological signals like EEG, ECG, and wearable sensor data (heart rate, steps, motion).  
It is part of the **MOP AI+IoT Capstone – Health Behaviour Analysis** initiative, contributing toward **SDG 3 – Good Health and Wellbeing**.

---

## 🔧 Pipeline Overview

### Data Processing
- Imported and cleaned MIT-BIH Polysomnographic `.edf` files using `wfdb` and `mne`.
- Extracted and normalised EEG (F3-Cz) and ECG channels.
- Created 30-second windows (shape: `(3000, 2)`) aligned with sleep stage scoring.
- Preprocessed wearable data (heart rate, steps, motion) from NSRR in `.txt` format.
- Built matching `.npy` tensors for both EEG and wearable inputs.

### Exploratory Analysis
- Visualised EEG & ECG distributions.
- Analysed time-domain and frequency-domain features.
- Evaluated signal variance across sleep stages and demographics.

---

## 🤖 Model Development

### EEG/ECG Model
- Built and trained a **Transformer-based classifier**.
- Used **KMeans** for pseudo-labeling when labels were missing.
- Applied **class weighting** and **early stopping**.
- Exported models in `.keras` and `.tflite` formats.

### Wearable Model
- Developed a **parallel Transformer model** for wearable inputs.
- Evaluated using **accuracy** and **F1-score**.

### Fusion Architecture
- Designed a **late-fusion model** combining EEG and wearable data.
- Architecture is modular and training-ready.

---

## ⚙️ Deployment: GraphQL API
A **FastAPI + Strawberry GraphQL backend** was developed for real-time inference.

**Features:**
- `/predictEEG` and `/predictWearable` endpoints  
- Batch predictions with confidence scores  
- Returns label, probability, metrics (F1, confusion matrix)  
- API key security and inference test scripts included  

---

## 📊 Evaluation Metrics
- **EEG Transformer Accuracy:** ~93%  
- **Wearable Transformer Accuracy:** ~85%  
- **Fusion Model:** Architecture complete, training in progress  

---

## 🚀 How to Run

1. **Clone Repository:**
   ```bash
   git clone https://github.com/yourusername/sleep-stage-prediction.git
   cd sleep-stage-prediction

   pip install -r requirements.txt
   python train_transformer_eeg.py
   python train_transformer_wearable.py
   python run_graphql_api.py


## 📂 Repository Structure

- `data_preprocessing/`: EEG, ECG, and wearable loaders  
- `models/`: Transformer architectures and fusion model  
- `notebooks/`: EDA, clustering, evaluation  
- `api/`: FastAPI + GraphQL code  
- `npy/`: Preprocessed model input tensors  
- `X_eeg.npy, y_eeg.npy`: EEG model inputs/labels  
- `X_wearable.npy`: Wearable model inputs  
- `eeg_transformer.keras`: Exported model  
- `README.md`: Project documentation  

## 👨‍💻 Project Lead
**Leela Venkata Subba Rayudu Gudipalli** – Sub-Project Lead, Health Behaviour  
- Led architecture, preprocessing, and model development  
- Created GraphQL API and transformer pipelines  
- Conducted GitHub reviews, team mentoring, and integration coordination  

---

## 👥 Team Members
- Nattakan  
- Bhavitra  
- Vinita  
- Alen  
- Devin  
- Rishit  

---

## 📌 Future Work
- Dockerise API for production  
- Align more wearable subjects for fusion model  
- Deploy `.tflite` model on mobile or edge devices  
- Build dashboard for real-time predictions  
- Prepare open-source GitHub release  

