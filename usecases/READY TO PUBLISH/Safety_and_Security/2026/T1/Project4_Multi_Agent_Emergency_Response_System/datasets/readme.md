# Project 4 – Multi-Agent Emergency Response System  
## Dataset Documentation

This folder contains the datasets prepared for Sprint 1 of the project. These datasets support emergency service allocation, routing optimisation, and congestion analysis.

---

## 1. Emergency Crash Dataset
Purpose: Used to simulate emergency incidents and test dispatch decisions.

Key columns:
- ACCIDENT_NO – incident ID
- ACCIDENT_DATE – date of incident
- ACCIDENT_TIME – time of incident
- LATITUDE – incident latitude
- LONGITUDE – incident longitude
- SEVERITY – incident severity

Use in project:
Used as emergency event inputs for testing response allocation.

---

## 2. Hospitals Dataset
Purpose: Identify nearest hospital destinations for ambulance routing.

Key columns:
- name – hospital name
- latitude – location
- longitude – location

Use in project:
Destination points for ambulance routing.

Source:
OpenStreetMap

---

## 3. Fire Stations Dataset
Purpose: Used to allocate nearest fire response units.

Key columns:
- name – fire station name
- latitude – location
- longitude – location

Use in project:
Fire response agent locations.

Source:
OpenStreetMap

---

## 4. Police Stations Dataset
Purpose: Used to allocate police response agents.

Key columns:
- name – police station
- latitude – location
- longitude – location

Use in project:
Police agent allocation for emergency scenarios.

Source:
OpenStreetMap

---

## 5. Road Network Dataset
Files:
- road_nodes.csv
- road_edges.zip

Purpose:
Used to construct routing graph for emergency vehicles.

Key columns:
Nodes:
- node_id
- latitude
- longitude

Edges:
- start node
- end node
- length

Use in project:
Shortest path routing.

Source:
OpenStreetMap

---

## 6. Pedestrian Dataset (City of Melbourne)
Purpose:
Used to model pedestrian congestion affecting emergency response routing.

Key columns:
- sensor_id
- date
- hour
- pedestrian_count
- latitude
- longitude

Processing:
- Removed unnecessary columns
- Extracted coordinates
- Removed missing values
- Created congestion score

Use in project:
Congestion modelling and routing optimisation.

Source:
City of Melbourne Open Data (MOP)

---

## Summary of datasets

| Dataset | Purpose |
|---------|---------|
| Emergency crashes | Incident simulation |
| Hospitals | Ambulance routing |
| Fire stations | Fire dispatch |
| Police stations | Police dispatch |
| Road network | Routing graph |
| Pedestrian data | Congestion modelling |

---

Prepared by: Diyona  
Sprint: Sprint 1  
Project: Multi-Agent Emergency Response System
