# Project 4 - Emergency Routing Notebook

## Notebook Documentation


The notebook builds a road-network routing pipeline for Geelong and compares Dijkstra and A* for emergency route analysis.

## 1. Notebook Purpose

Purpose: Build, test, and benchmark shortest-path routing on a real road network.

Use in project: Baseline routing and performance comparison for emergency response scenarios.

## 2. Input Configuration Used In Code

Place: Geelong, Victoria, Australia

Network type: drive

Start point: latitude -38.1499, longitude 144.3617

End point: latitude -38.1350, longitude 144.3550

Primary edge weight: length

## 3. Processing Steps In The Notebook

1. Import libraries and enable OSMnx cache/logging.
2. Download drivable road graph from OpenStreetMap.
3. Plot full graph and save visualization.
4. Save graph to GraphML and reload from disk.
5. Convert graph to node and edge GeoDataFrames.
6. Inspect node coordinates and edge length attributes.
7. Map start and end coordinates to nearest graph nodes.
8. Plot start and end nodes on the network.
9. Compute shortest route using Dijkstra.
10. Compute shortest route using A*.
11. Calculate route distance using multiedge-safe helper logic.
12. Benchmark runtime for both algorithms.
13. Export algorithm comparison table.

## 4. Routing Methods Implemented

Dijkstra

Purpose: Find shortest path by total road length.

Implementation: NetworkX shortest_path with method set to dijkstra and weight set to length.

A*

Purpose: Find shortest path using a heuristic-guided search.

Implementation: NetworkX astar_path with Euclidean coordinate heuristic and weight set to length.

## 5. Output Files Generated

Graph output:

- [outputs/graphs/geelong.graphml](outputs/graphs/geelong.graphml)

Map outputs:

- [outputs/maps/graph_visualization.png](outputs/maps/graph_visualization.png)
- [outputs/maps/start_end_nodes.png](outputs/maps/start_end_nodes.png)
- [outputs/maps/dijkstra_route.png](outputs/maps/dijkstra_route.png)
- [outputs/maps/astar_route.png](outputs/maps/astar_route.png)

Table output:

- [outputs/tables/algorithm_comparison.csv](outputs/tables/algorithm_comparison.csv)

## 6. Result Summary

The notebook produces:

- A complete drivable road graph for the target area
- Start and destination node mapping on the network
- Dijkstra and A* route visualizations
- Route length values in meters and kilometers
- Runtime comparison in CSV format


