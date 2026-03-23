# Urban Parking Demand Forecasting using Graph Neural Networks (GNN)

## Project Overview
This project focuses on predicting parking availability in urban areas using data-driven techniques. The aim is to analyse parking patterns across different locations and time periods and develop a model that can forecast future parking demand.

---

## Objectives
- Forecast parking demand across different locations and time periods  
- Identify peak parking hours and high-demand areas  
- Reduce traffic congestion caused by searching for parking  
- Support smarter urban planning and decision-making  

---

## Dataset
- **Dataset:** Melbourne Parking Dataset  
- **Source:** (Add link here)  
- **Status:** Cleaned and prepared for analysis  

### Key Features
- Parking Bay ID  
- Timestamp  
- Latitude & Longitude  
- Occupancy Status  

### Dataset Description
The dataset contains both spatial (location-based) and temporal (time-based) information, making it suitable for analysing real-world parking behaviour. It enables the study of demand patterns across different locations and time intervals.

---

## Data Preparation
The dataset was explored and cleaned to ensure consistency and usability.  
- Irrelevant columns were removed  
- Timestamp was converted into proper datetime format  
- Location data was structured into latitude and longitude  
- A cleaned dataset (`cleaned_parking.csv`) was created for further analysis  

---

## Exploratory Data Analysis
Basic analysis was performed to understand parking demand patterns:
- Parking demand varies across different hours of the day  
- Weekdays generally show higher occupancy compared to weekends  
- Certain parking bays consistently have higher demand  
- Parking usage depends on both time and location  

These patterns indicate that parking demand is predictable and suitable for forecasting.

---

## Proposed Approach
This project uses a Graph Neural Network (GNN) approach to model parking demand:

- **Nodes:** Parking bays  
- **Edges:** Connections between nearby parking bays  
- **Features:** Occupancy, time (hour/day), and location  
- **Target:** Predict future parking availability  

This approach allows capturing both spatial and temporal relationships in the data.

---

## Technologies Used
- Python  
- Pandas & NumPy  
- PyTorch / PyTorch Geometric  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## Project Status
- Dataset selected and cleaned  
- Initial exploratory analysis completed  
- Graph-based approach defined  
- Project structure and collaboration setup completed  

---

## Team Members
- Aishwarya  
- Chiranjeevi  
- Aditya  

---

## Future Work
- Build graph structure from the dataset  
- Implement and train the GNN model  
- Evaluate model performance  
- Improve prediction accuracy  