#  Physical Activity Recognition using Deep Learning (PAMAP2)

This project applies deep learning models to classify physical activities using wearable sensor data from the **PAMAP2** dataset. The trained models are deployed in an interactive **Streamlit dashboard** that allows users to explore, test, and compare model performance in real-time.

---

## Use Case: Health Behaviour Monitoring – UC8  
> Developed as part of the MOP AI + IoT Capstone project

### Objective:
To recognize and classify physical activities such as walking, running, sitting, and climbing stairs based on time-series sensor data from wearable devices.

---

## Features

- Preprocessed and cleaned PAMAP2 sensor data  
- Deep Learning Models: LSTM, GRU, Bidirectional and smaller variants  
- Best test accuracy: **62.91%** (GRU 32 units)  
- Evaluation via confusion matrix and classification report  
- **Streamlit dashboard** for interactive model comparison  
- Modular codebase for scalability and further experimentation  

---

## Folder Structure
```
anisha_t125_UC8/
├── README.md                          # Project-specific readme
├── KT_ani_HandlingGithubLargeFilePushIssue.pdf  # KT documentation
└── UC8_WorkFiles/
    ├── physicalActivityDashboard.py  # Streamlit dashboard app
    ├── codeLogic.ipynb               # Notebook with data prep, training, evaluation
    ├── anisha_projectReport_pdf.pdf        # Project Report
    ├── requirements.txt              # Python dependencies
    ├── *.keras / *.h5                # Trained model files (e.g., GRU, LSTM variants)
    ├── readme.pdf                    # dataset understanding
    ├── DataCollectionProtocol.pdf    # Supporting activity doc
    ├── DescriptionOfActivities.pdf   # Activity description
    ├── PerformedActivitiesSummary.pdf
    ├── subjectInformation.pdf        # Sensor and participant metadata

```
---

## How to Run

## Dataset & Setup Instructions

> **Important:** PAMAP2 `.dat` files are not included in this repository due to GitHub’s 100 MB file size limit.

To run the full preprocessing pipeline:

1. **Manually download** the dataset from the UCI repository:  
   [https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)

2. Place the `.dat` files in the appropriate folder as referenced in the notebook (`codeLogic.ipynb`).


3. **Install dependencies**  
   *(Ensure Python 3.12+ is installed)*

pip install -r requirements.txt

4. **Launch the dashboard**

streamlit run physicalActivityDashboard.py

5. Open your browser and go to the URL displayed in the terminal (typically `http://localhost:8501`)

---

## Dependencies
- Python 3.12+  
- Streamlit  
- TensorFlow 2.16.1  
- NumPy, Pandas, Matplotlib, Seaborn  
- Scikit-learn  

---

## Dataset
[PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) from UCI Machine Learning Repository

---

Developed by **Anisha Raj**  
MOP AI + IoT – Health Behavior Group
