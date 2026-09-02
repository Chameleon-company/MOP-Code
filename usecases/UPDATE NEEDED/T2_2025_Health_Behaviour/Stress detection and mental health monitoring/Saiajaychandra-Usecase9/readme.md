# Capstone Project — AI-Based Stress & Mental Health Monitoring  
**Dataset:** DREAMERV  

This repository contains my full capstone project notebook, compiled **sprint by sprint** (Sprints 1–5).  
The work follows a step-by-step structure to show how I experimented, optimized, and finally tested deep learning models for stress and mental health monitoring.

---

##  Repository Contents
- **StressDetection_DREAMERV_CNN_LSTM_Final.ipynb**  
  The main notebook that includes:
  - Sprint 1: Data Collection,Preprocessing
  - Sprint 2: CNN and LSTM baseline Models, Transfer Learning and Hyperparameter Tuning. 
  - Sprint 3: Model evaluations + Transformer exploration  
  - Sprint 4: Optimizations (CNN, LSTM, pruning/quantization, latency profiling) + Streamlit demo + Cloud feasibility  
  - Sprint 5: Final stress testing  
  -**README.md** (this file)

---

##  How to Run
1. Open the notebook in **Google Colab** (recommended).  
2. Set runtime to **GPU** if available.  
3. Run the **Setup** and **Project Config** cells first.  
   - Update `DATASET_ROOT` to point to the DREAMERV dataset location (e.g., Google Drive mount).  
4. Execute the notebook sprint by sprint in sequence.  
5. For the **Streamlit demo**, follow the provided instructions in the notebook (localtunnel link needed in Colab).  

---

## Project Overview
- **Goal:** Use deep learning to monitor stress and mental health signals.  
- **Models tested:** CNNs, LSTMs. 
- **Optimizations applied:** Quantization, pruning, and latency profiling for deployment readiness.  
- **Deployment exploration:** Streamlit demo + cloud feasibility study.  
- **Final evaluation:** Stress testing on held-out DREAMERV data with metrics like accuracy, F1, and confusion matrices.

---

## Reflections
Through this project, I learned:
- CNNs are quick baselines, LSTMs are strong for temporal modeling.  
- Optimizations like quantization and pruning help balance accuracy with efficiency.  
- Latency profiling is essential to prove real-world deployment benefits.  
- Streamlit and cloud experiments make the project more practical, beyond just academic experiments.  

---

##  Next Steps
- Try CNN → LSTM hybrid architectures.  
- Explore subject-wise calibration to handle variability in physiological signals.  
- Investigate lightweight deployment strategies (quantization-aware training, pruning during training, knowledge distillation).  
- Test beyond DREAMERV for broader generalizability.  

---
