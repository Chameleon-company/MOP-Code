# Agent Dispatch Module — Multi-Agent Emergency Response System

**Author:** Manya Mahajan (AI Agent Developer)
**Project:** Project 4 — Multi-Agent Emergency Response System, Victoria, Australia

## Overview

This module implements an intelligent emergency dispatch system that determines which emergency services to send to an incident, resolves the nearest real facility using geographic data, and calculates the fastest route using a road network routing engine.

## Files

| File | Description |
|---|---|
| `Project4_Multi_Agent_Emergency_Response_Dispatch_System.ipynb` | Final notebook — full dispatch pipeline with routing, RL scoring, and visualisations |
| `AgentDispatchModule_Sprint1.ipynb` | Sprint 1 prototype — initial dispatch logic without routing |

## Features

- Dispatch logic covering 10 emergency types across 3 priority levels
- Nearest-facility lookup using Haversine distance for hospitals, fire stations, and police stations
- A* shortest-path routing on the Melbourne road network (228,000+ nodes)
- Pedestrian and transport sensor integration for situational awareness
- Priority escalation based on real-time congestion and peak hour detection
- Agent availability prediction with estimated return times
- RL-inspired dispatch scoring with policy updates for agent selection optimisation
- Validation using real Victorian crash incident data (194,352 records)

## Datasets

All datasets are stored in `datascience/usecases/DEPENDENCIES/Project4_Multi_Agent_Emergency_Response_System_Datasets/` and loaded via GitHub raw URL at runtime.

| Dataset | Records |
|---|---|
| hospitals.csv | 26 emergency-capable facilities |
| fire_stations.csv | 205 stations |
| police_stations.csv | 124 stations |
| emergency_crash.csv | 194,352 incidents |
| road_nodes.csv | 228,213 nodes |
| road_edges.zip | 501,205 edges |
| pedestrian_data.csv | 734,064 records |
| cleaned_transport_2025.csv | 258,573 records |
| melbourne_graph.tar.gz | Melbourne road network graph |
| routing_engine.py | A* routing engine (OSMnx + NetworkX) |

## How to Run

1. Open the final notebook in Jupyter
2. Run all cells from top to bottom
3. The notebook automatically downloads datasets and the routing engine from the DEPENDENCIES folder on first run
4. CUDA is used for routing if available, otherwise falls back to CPU