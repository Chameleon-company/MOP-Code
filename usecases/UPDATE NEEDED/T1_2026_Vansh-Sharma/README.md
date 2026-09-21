# Spatiotemporal Ride-Hailing Demand Prediction System

## Project Overview

Deep learning system that predicts ride-hailing demand across city zones and time periods, combining spatial information (city regions) with temporal patterns (time of day, day of week).

The system implements two model architectures:

- **LSTM** — captures temporal dependencies (daily cycles, weekly patterns)
- **Temporal Graph Neural Network (T-GNN)** — captures both spatial (zone adjacency) and temporal dependencies using Graph Convolutional Networks + GRU

## Datasets

- **NYC TLC Yellow Taxi** (real data) — 19.5M trips, 6 months (Jan–Jun 2023), 262 zones
- **Melbourne Synthetic** — 2.8M trips, 6 months (Jan–Jun 2025), 40 zones calibrated to realistic Melbourne urban patterns

## Key Results

| City                  | Model | MAE  | RMSE  | MAPE  | Improvement over baseline |
| --------------------- | ----- | ---- | ----- | ----- | ------------------------- |
| NYC (real data)       | LSTM  | 3.08 | 10.41 | 23.9% | 52.9%                     |
| NYC (real data)       | T-GNN | 6.09 | 20.97 | 48.1% | 6.9%                      |
| Melbourne (synthetic) | LSTM  | 2.79 | 4.36  | 20.6% | 66.7%                     |
| Melbourne (synthetic) | T-GNN | 4.54 | 7.49  | 31.9% | 45.7%                     |

- 87%+ predictions within ±5 rides of actual demand
- LSTM consistently outperforms, demonstrating temporal patterns as the dominant signal
- System architecture is city-agnostic — only data changes between cities

## Melbourne Use Case

The system demonstrates applicability to Melbourne for:

- Driver allocation optimization across CBD, entertainment precincts, and transport hubs
- Dynamic pricing — anticipating surge periods before they occur
- Transport infrastructure planning — identifying underserved areas
- Event impact assessment — modeling demand around MCG, Melbourne Park, Flemington

## Tech Stack

- Python 3.11, PyTorch, PyTorch Geometric
- Pandas, NumPy, GeoPandas, Matplotlib
- NYC TLC trip record data + Melbourne synthetic demand generator

## How to Run

Open `spatiotemporal_ridehail_prediction.ipynb` and run all cells sequentially. The notebook:

1. Installs/imports all required libraries
2. Downloads NYC taxi data automatically
3. Trains and evaluates all models
4. Generates Melbourne synthetic data and trains models
5. Produces all comparison tables and visualisation plots

**Requirements:** Python 3.10+, PyTorch, PyTorch Geometric, pandas, numpy, geopandas, matplotlib
