# Urban Parking Demand Forecasting Using Graph Neural Networks

## Project Overview

This project predicts the **next time-step parking occupancy** for on-street parking bays in Melbourne using a **Graph Neural Network (GNN)**.

Each parking bay is treated as a node in a graph:

- **Nodes** = Parking bays
- **Edges** = Spatial relationships between nearby bays
- **Node Features** = Hour, day of week, latitude, longitude
- **Target** = Occupancy at the next time step (`t + 1`)

The system allows users to:

1. Select a geographic area (e.g., Richmond, Docklands)
2. Enter a desired hour and day
3. Predict future occupancy for all bays in that area
4. Visualize results interactively

---

## Problem Statement

Finding available parking in busy urban areas is time-consuming and contributes to congestion.

Traditional machine learning models treat parking bays independently and ignore the influence of nearby bays.

This project addresses that limitation by using a Graph Neural Network that learns both:

- **Temporal patterns** (time-based demand)
- **Spatial patterns** (neighboring bay relationships)

---

## Key Features

- Graph Neural Network using `GCNConv`
- Forecasting logic (`t → t+1`)
- Synthetic area classification for region-based filtering
- User-driven prediction by area and time
- Interactive graph visualization
- Confidence scores and prediction correctness indicators

---

## Project Architecture

```text
Raw Parking Dataset
        ↓
Data Cleaning + Area Assignment
        ↓
Graph Construction
        ↓
Feature Engineering
        ↓
Model Training (GCN)
        ↓
Saved Model (model.pth)
        ↓
User Selects Area + Time
        ↓
Prediction
        ↓
Interactive Visualization
```

---

## Technology Stack

| Component | Technology |
|---------|---------|
| Programming Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Deep Learning | PyTorch |
| Graph Learning | PyTorch Geometric |
| Graph Construction | NetworkX |
| Visualization | Matplotlib |
| Dataset | Melbourne Parking Sensor Data |

---

## Dataset Description

The cleaned dataset contains:

- `bay_id`
- `timestamp`
- `latitude`
- `longitude`
- `occupancy` (0 = free, 1 = occupied)
- `area`

### Areas Included

- Melbourne CBD
- Richmond
- Fitzroy
- Carlton
- Brunswick
- Docklands
- South Yarra
- St Kilda

---

## Forecasting Objective

### Input Features

For each parking bay:

- Hour of day
- Day of week
- Latitude
- Longitude
- Neighbor relationships

### Model Output

Predict:

> **Will this parking bay be occupied at the next time step?** (`t + 1`)

---

## Repository Structure

```text
Urban_Parking_P6/
│
├── data/
│   ├── cleaned_parking_with_area.csv
│   └── cleaned_parking_data.csv
│
├── src/
│   ├── graph.py
│   ├── features.py
│   ├── prepare_data.py
│   ├── model.py
│   ├── train.py
│   ├── visualize_predictions.py
│   ├── select_area.py
│   ├── visualize_area.py
│   └── predict_by_area.py
│
├── model.pth
├── requirements.txt
└── README.md
```

---

# File-by-File Explanation

## `graph.py`

Creates the spatial graph.

### Responsibilities

- Loads the dataset
- Creates one node per parking bay
- Connects nearby bays using Haversine distance
- Stores occupancy as node attributes

### Output

A NetworkX graph object `G`.

---

## `features.py`

Generates model features and labels.

### Responsibilities

- Maps bay IDs to integer node indices
- Creates forecasting labels using `shift(-1)` to represent `t + 1`
- Normalizes feature values
- Creates `edge_index` for PyTorch Geometric

### Output

- `X`: Feature matrix
- `y`: Labels
- `edge_index`: Graph connectivity

---

## `prepare_data.py`

Combines graph and feature engineering.

### Responsibilities

- Imports `G` from `graph.py`
- Calls `prepare_features()` from `features.py`
- Produces tensors used during training

### Output

- `X`
- `y`
- `edge_index`

---

## `model.py`

Defines the Graph Neural Network.

### Architecture

- `GCNConv(input_dim → hidden_dim)`
- ReLU activation
- Dropout
- `GCNConv(hidden_dim → output_dim)`

### Output Classes

- `0` = Free
- `1` = Occupied

---

## `train.py`

Trains the model.

### Responsibilities

- Loads tensors from `prepare_data.py`
- Splits data into 80% training and 20% testing
- Applies class weights for imbalance
- Trains for multiple epochs
- Tracks best test accuracy
- Saves `model.pth`

### Example Output

- Train Accuracy ≈ 0.65
- Test Accuracy ≈ 0.40–0.70

---

## `visualize_predictions.py`

Interactive visualization for the trained model.

### Features

- Real geographic node positions
- Clickable nodes
- Confidence scores
- Correct/incorrect indicators

---

## `select_area.py`

Simple utility script that:

- Lists available areas
- Lets the user choose one
- Displays row and bay counts

---

## `visualize_area.py`

Plots only the parking bays belonging to a selected area.

Useful for validating area-based filtering.

---

## `predict_by_area.py`

Final integrated application.

### Workflow

1. User selects area
2. User enters hour and day
3. Data is filtered to that area
4. Trained model is loaded
5. Predictions are generated
6. Interactive graph is displayed

This is the main script for demonstrations.

---

## `model.pth`

Serialized trained model weights.

Generated by:

```bash
python train.py
```

Loaded by:

```python
model.load_state_dict(torch.load("model.pth"))
```

---

# Model Training Details

## Train/Test Split

- 80% Training
- 20% Testing

## Loss Function

Cross-entropy loss with class weights.

## Optimizer

Adam optimizer.

## Threshold

A probability threshold (e.g., 0.4) is used to classify occupied vs free.

---

# Visualization Legend

| Visual Element | Meaning |
|--------------|---------|
| Red node | Actual occupied bay |
| Green node | Actual free bay |
| Yellow border | Incorrect prediction |
| Black border | Correct prediction |
| Blue node | Selected node |

---

# How to Run the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Train the Model

```bash
cd src
python train.py
```

## 3. Run the Final Application

```bash
cd src
python predict_by_area.py
```

## 4. Enter Inputs

Example:

```text
Select an area by number: 6
Enter hour (0-23): 18
Enter day (0-6): 4
```

This predicts parking occupancy for Richmond at 6 PM on Friday.

---

# Example Use Cases

- Predict Friday evening parking demand in Richmond
- Compare weekend vs weekday occupancy
- Identify likely congested areas
- Explore spatial influence between neighboring bays

---

# Current Limitations

- Areas are synthetic groupings based on clustering/manual assignment
- Accuracy is moderate due to limited features
- Current model uses static snapshots rather than full time-series sequences
- Visualization is intended for demonstration rather than city-scale rendering

---

# Future Improvements

- Use historical sequences (LSTM + GNN)
- Integrate weather and event data
- Build a web dashboard with Streamlit or Flask
- Deploy to cloud for real-time forecasting
- Improve area definitions using real suburbs or GIS boundaries

---


# Sample Results

Typical model output:

- Train Accuracy: ~0.65
- Test Accuracy: ~0.40–0.70

These results demonstrate that the model captures meaningful spatial and temporal occupancy patterns.

--

