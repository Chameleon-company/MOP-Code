# Tourist Mobility — Luggage Flow Optimization (Melbourne)

This project analyzes how tourists move between **accommodations, attractions, and transport hubs** in Melbourne and optimizes **luggage storage** decisions using data + machine learning. It includes feature engineering from footfall windows, classification (locker usage), regression (travel time), clustering (demand zones), and clear visualizations (graphs and maps).

> Main files  
> - `TouristMobility_usecase.ipynb` — end-to-end notebook  
> - `luggage_flow_simulation_detailed.csv` — input dataset

---

## 1) What this does

- Builds features such as **Footfall_Total** and **Footfall_Peak** from time windows  
- Predicts **Nearby_Locker (Yes/No)** using a classification model  
- Predicts **Estimated_Travel_Time_min** using regression  
- Clusters locations to find **high-demand zones** for lockers  
- Draws a **clean bipartite graph** (Locations → Nearest Lockers) with distance/time labels  
- (Optional) Exports results to a CSV and creates an interactive map

---

## 2) Dataset format

The CSV has these columns (as used in the notebook/code):

- `Location`  
- `Type` (e.g., Attraction, Transport Hub, Market, Museum)  
- `Estimated_Footfall_10_12`  
- `Estimated_Footfall_12_2`  
- `Estimated_Footfall_2_4`  
- `Estimated_Footfall_4_6`  
- `Nearby_Locker` (Yes/No)  
- `Nearest_Locker_Location`  
- `Travel_Distance_km`  
- `Estimated_Travel_Time_min`

> The notebook computes:
> - `Footfall_Total = sum(4 windows)`  
> - `Footfall_Peak = max(4 windows)`  
> - `Nearby_Locker_bin = {Yes:1, No:0}`  
> - `Demand_Score = Footfall_Total / (1 + Travel_Distance_km)`

---

## 3) Getting started

### — Jupyter 
1. Open **Anaconda Navigator** → launch **Jupyter Notebook**.  
2. Navigate to the folder with `TouristMobility_usecase.ipynb` and the CSV.  
3. Open the notebook and run cells top-to-bottom.

---

## 4) Requirements (tested on Windows)

- Python 3.10+  
- pandas, numpy, matplotlib, scikit-learn, networkx, (optional) folium

> If you run into Windows binary/DLL issues, these pinned versions are known-good:  
> `numpy==1.26.4`, `scipy==1.11.4`, `scikit-learn==1.4.2`

---

## 5) Reproduce the analysis (inside the notebook)

1. **Load & feature engineering**  
   - Reads the CSV  
   - Creates `Footfall_Total`, `Footfall_Peak`, `Nearby_Locker_bin`  
   - Prints a small preview to sanity-check columns

2. **Classification (locker presence)**  
   - Features: `Type`, `Travel_Distance_km`, `Footfall_Total`, `Footfall_Peak`  
   - Model: `RandomForestClassifier` (with one-hot for `Type`)  
   - Metrics: accuracy + classification report

3. **Regression (travel time)**  
   - Target: `Estimated_Travel_Time_min`  
   - Features: previous + `Nearby_Locker_bin`  
   - Model: `LinearRegression`  
   - Metrics: MAE, RMSE

4. **Clustering (demand zones)**  
   - Inputs: `Travel_Distance_km`, `Footfall_Total` (standardized)  
   - Model: `KMeans(n_clusters=3)`  
   - Output: `Demand_Cluster` per location + centroids (original units)

5. **Graph visualization (clean, non-scattered)**  
   - Directed bipartite graph: **Locations (left)** → **Nearest Lockers (right)**  
   - Edge labels show **distance (km)** (and **time** if added)  
   - Useful for quick explanation of travel effort

6. **(Optional) Exports**  
   - Writes `luggage_flow_analysis_results.csv` with demand metrics and clusters  
   - Saves an interactive `luggage_flow_map.html` if you enable the map cell

---

## 6) Interpreting results

- **High `Demand_Score` + short distance** → great candidates for **locker placement**.  
- **Classification** helps estimate where locker presence is expected based on current conditions.  
- **Regression** explains how features drive **travel time** (e.g., longer distances, lower locker access).  
- **Clusters** separate **high-demand, moderate, and low-demand** zones.

---

## 7) Project structure (suggested)

```
.
├─ TouristMobility_usecase.ipynb
├─ luggage_flow_simulation_detailed.csv
├─ analysis/                     # optional: scripts if you split notebook cells
│  ├─ features.py
│  ├─ models.py
│  └─ visualize.py
├─ outputs/
│  ├─ luggage_flow_analysis_results.csv
│  └─ luggage_flow_map.html
└─ README.md
```

---

## 8) Troubleshooting

- **KeyError on column name**  
  - Print columns with `print(df.columns.tolist())` and check for typos/casing/extra spaces.  
- **DLL load failed / NumPy/SciPy mismatch on Windows**  
  - Pin compatible versions: `numpy==1.26.4`, `scipy==1.11.4`, `scikit-learn==1.4.2`  
  - Restart the Jupyter kernel after installing.
- **Scattered graph**  
  - Use the **bipartite layout** code provided; it’s deterministic and neat.

---

## 9) How to cite / present

- Open with the **problem** (tourists carrying luggage post-checkout).  
- Show **maps/graphs** and **clusters** to argue where lockers are most needed.  
- Include **model metrics** to demonstrate rigor.  
- End with **recommendations**: priority sites and route guidance.

---

**Happy analyzing!**
