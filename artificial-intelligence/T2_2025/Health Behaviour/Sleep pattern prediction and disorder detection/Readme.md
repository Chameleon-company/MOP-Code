Sleep Stage Classification using EEG, ECG, and Wearables
This repository contains the code, models, and documentation for a Capstone Project aimed at developing an AI system to predict sleep patterns and detect sleep disorders using biosignal data (EEG/ECG) and wearable sensor data (heart rate, steps, motion).
📌 Project Objectives
- Predict sleep stages from EEG/ECG biosignal recordings. - Use wearable sensor data (heart rate, motion, steps) for complementary sleep analysis. - Apply clustering and deep learning (CNN, LSTM, Transformer). - Export models to `.keras` and `.tflite` formats for real-time deployment. - Package final models into an API (planned).
📊 Datasets Used

🧪 Project Structure
 📁 /notebooks     └── artificial-intelligence/T2_2025/Health Behaviour/Sleep pattern prediction and disorder detection/Copy_of_ProjectB-2.ipynb    # Full model development notebook 
🧠 Model Pipelines
EEG/ECG Pipeline
- 30-second segmentation from `.csv` files - Feature extraction and clustering (KMeans) - Pseudo-label generation from clusters - Transformer model: 2 attention blocks, global pooling, dropout - GroupKFold evaluation (subject-wise) - Exported model to `.keras` and `.tflite`
Wearable Pipeline *(Planned)*
- Clean and align heart rate, steps, and motion - LSTM/Transformer model (input shape: `(300, 3)`) - Fusion model combining EEG + wearable (postponed)
📦 Model Export
- ✅ `eeg_transformer.keras` — usable in Keras/TensorFlow - ✅ `eeg_transformer.tflite` — deployable on mobile or embedded devices - 🔜 Wearable and Fusion models pending data alignment
🛠️ Setup Instructions
`bash pip install -r requirements.txt jupyter notebook notebooks/Copy_of_ProjectB-2.ipynb  ```
🚀 Future Work
- [ ] Finalize wearable model training and evaluation - [ ] Build and deploy FastAPI REST endpoint for `.keras` inference - [ ] Integrate sleep stage visualizations via API - [ ] Real-time wearable stream support (planned)
 Author
Leela Venkata Subba Rayudu 💻 Master of IT – Deakin University 🌐 https://github.com/rayudu-os956
📄 License
This project uses publicly available open datasets and is intended for educational and research use only.
