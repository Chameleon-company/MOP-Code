# Project 4 - Emergency Routing Notebook
Author: Basil Behanan(Routing Optimization Engineer) Project: Project 4 — Multi-Agent Emergency Response System

# Emergency Routing System

## Overview

This project is part of the **Multi-Agent Emergency Response System** developed under the Chameleon company project. The goal of this module is to build a routing and optimization system that can calculate efficient routes between emergency incidents and nearby emergency facilities such as hospitals, fire stations, and police stations.

The routing system uses real-world road network data from **OpenStreetMap (OSM)** through the **OSMnx** library and applies graph-based routing algorithms to compute optimal paths.

---

# Project Structure

```text
EmergencyRoutingSystem/
│
├── README.md
├── benchmark_routing.py
├── melbourne_algorithm_benchmark.csv
├── osmnx_test.ipynb
├── routing_engine.py
````

---

# Files Description

## `osmnx_test.ipynb`

Main development notebook used for:

* Downloading Melbourne road network data
* Graph generation using OSMnx
* Graph visualisation
* Node and edge exploration
* Testing routing algorithms
* Route plotting and experimentation

This notebook was used during the research and prototype stage of the routing system.

---

## `routing_engine.py`

Core routing module containing:

* Route computation functions
* Dijkstra implementation
* A* implementation
* Coordinate-to-node mapping
* Route summary generation

This file is intended to support integration with the dispatch system and future multi-agent coordination modules.

---

## `benchmark_routing.py`

Script used to benchmark routing algorithms on the Melbourne graph.

Current comparisons include:

* Runtime performance
* Route distance
* Route node count

between:

* Dijkstra
* A*

---

## `melbourne_algorithm_benchmark.csv`

Generated benchmark results comparing routing algorithms on the Melbourne road network.

The file stores:

* Algorithm name
* Route distance
* Runtime performance
* Route metrics

Used for analysis and evaluation.

---

# Current Progress

## Melbourne Road Network Integration

The routing system was initially developed using the Geelong road network and later expanded to the Melbourne road network to align with the datasets used in the overall project.

The graph currently includes:

* Real-world road structures
* Nodes and edges
* Distance attributes
* Speed attributes
* Travel-time weights

---

# Routing Features Implemented

## Graph Construction

Implemented using:

* OSMnx
* NetworkX

Features include:

* Road network download
* Graph conversion
* Graph saving/loading
* Travel-time edge weighting

---

## Routing Algorithms

Implemented and tested:

### Dijkstra Algorithm

Used as the baseline shortest-path routing method.

### A* Algorithm

Used as an optimized routing approach with heuristic-based search.

---

# Route Output

The routing system currently returns:

* Distance (km)
* Estimated travel time
* Route node count
* Algorithm used
* Route status

The output structure is designed for future integration with the dispatch module.

---

# Integration Work

The routing module has been structured to support integration with:

* Dispatch agents
* Facility datasets
* Incident datasets
* Future traffic-based routing
* Multi-agent emergency coordination

The routing implementation and outputs were shared with teammates through GitHub to support ongoing integration work.

---

# Shared Outputs

The following outputs and resources have been uploaded and shared:

* Routing implementation scripts
* Graph visualisations
* Route maps
* Benchmark comparison outputs
* CSV benchmark results
* Routing experiment notebook

---

# Important Note About Graph Files

The generated Melbourne `.graphml` graph file is large and therefore is not directly stored in the repository.

Instead:

* The graph can be regenerated directly through code
* Graph generation scripts are included
* Pre-generated graph files can be shared externally when required

This keeps the repository lightweight and reproducible.

---

# Technologies Used

* Python
* OSMnx
* NetworkX
* OpenStreetMap
* Pandas
* Matplotlib
* Jupyter Notebook

---

