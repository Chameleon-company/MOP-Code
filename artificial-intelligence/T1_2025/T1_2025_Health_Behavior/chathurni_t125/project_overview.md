# Project Overview: Human Activity Recognition using Sensor Data

## 1. Project Title
AI-Powered Physical Activity Monitoring with LSTM, GRU, and 3D CNN

## 2. Objective
The main objective of this project is to **classify human physical activities** (such as Walking, Sitting, Jogging, Upstairs, Downstairs) using **time-series data** collected from smartphone **accelerometer** and **gyroscope** sensors.

This system can be used for:
- Fitness tracking,
- Health monitoring,
- Early detection of sedentary behavior risks (obesity, cardiovascular diseases),
- Smart wearable applications.

## 3. Datasets Used
- **WISDM v1.1 Dataset**  
Public dataset containing smartphone accelerometer and gyroscope readings across multiple activities performed by different users.

## 4. Methods and Models Implemented
- **Data Preprocessing**
  - Download and clean raw data.
  - Merge accelerometer and gyroscope data.
  - Handle missing values with interpolation and mean filling.
  - Normalize features using Min-Max Scaling.
  - Label encode activity classes.
  - Create sliding window segments for sequential model input.

- **Models Built**
  - **LSTM (Long Short-Term Memory)**
    - Original LSTM model.
    - Reduced complexity LSTM.
    - Bidirectional LSTM.
  - **GRU (Gated Recurrent Units)**
  - **3D CNN (Convolutional Neural Networks)**

- **Model Improvements**
  - Applied EarlyStopping and ReduceLROnPlateau callbacks.
  - Hyperparameter tuning (epochs, batch size, dropout rates).
  - Balanced training dataset to handle class imbalance.

## 5. Evaluation Metrics
- **Accuracy** (training and testing)
- **Loss** (training and testing)
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix Visualization**
- **True vs Predicted Activity Distribution**

## 6. Key Results
- LSTM models showed strong sequential pattern learning ability.
- GRU models provided good performance with fewer parameters.
- 3D CNN models showed moderate success but were less accurate for sequential dependencies.
- Bidirectional LSTM achieved the best performance in most tests.

## 7. Final Outcome
Successfully created a system capable of recognizing user physical activities with reasonable accuracy across multiple deep learning architectures.  
Documented the full process to support future students continuing this research.

## 8. Future Work
- Explore Transformer-based architectures (like TimeSformer or ViT for Time Series).
- Implement Self-Supervised Learning techniques (e.g., SimCLR) for better feature learning with limited labels.
- Optimize model deployment for mobile and wearable devices.

---

