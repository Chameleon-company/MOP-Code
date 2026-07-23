"""Routing engine for the emergency response project.

This module turns the notebook prototype into reusable code for Sprint 3.
Inputs are always expected in ``lat, lon`` order. OSMnx nearest-node lookup
internally uses ``X=lon`` and ``Y=lat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import networkx as nx
import osmnx as ox
import pandas as pd


DEFAULT_PLACE_NAME = "Melbourne, Victoria, Australia"
DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_ALGORITHM = "astar"
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[1] / "outputs" / "graphs" / "melbourne.graphml"


class RoutingEngineError(RuntimeError):
    """Raised when the routing engine cannot complete a request."""


@dataclass(frozen=True)
class CoordinatePair:
    """Simple lat/lon container used for benchmarking and readability."""

    lat: float
    lon: float


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_algorithm(algorithm: str) -> str:
    normalised = algorithm.strip().lower()
    if normalised not in {"astar", "dijkstra"}:
        raise ValueError("algorithm must be 'astar' or 'dijkstra'")
    return normalised


def _edge_weight(edge_data: dict, weight: str) -> float:
    return float(edge_data.get(weight, 0.0))


def _shortest_multiedge_value(graph: nx.MultiDiGraph, u: int, v: int, weight: str) -> float:
    edge_bundle = graph.get_edge_data(u, v)
    if not edge_bundle:
        raise RoutingEngineError(f"Missing edge data between nodes {u} and {v}.")

    best_edge = min(edge_bundle.values(), key=lambda item: _edge_weight(item, weight))
    return _edge_weight(best_edge, weight)


def _euclidean_heuristic(graph: nx.MultiDiGraph, source: int, target: int) -> float:
    source_node = graph.nodes[source]
    target_node = graph.nodes[target]
    dx = float(source_node["x"]) - float(target_node["x"])
    dy = float(source_node["y"]) - float(target_node["y"])
    return (dx * dx + dy * dy) ** 0.5


def add_travel_time_attributes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Attach speed and travel_time attributes to the graph when possible."""

    if not all("travel_time" in edge for _, _, _, edge in graph.edges(keys=True, data=True)):
        graph = ox.routing.add_edge_speeds(graph)
        graph = ox.routing.add_edge_travel_times(graph)
    return graph


def load_or_build_graph(
    place_name: str = DEFAULT_PLACE_NAME,
    network_type: str = DEFAULT_NETWORK_TYPE,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    force_download: bool = False,
) -> nx.MultiDiGraph:
    """Load a cached graph or download and cache a new one.

    The saved graph is enriched with ``travel_time`` so route summaries can
    report both distance and ETA.
    """

    graph_file = Path(graph_path)
    if graph_file.exists() and not force_download:
        graph = ox.load_graphml(graph_file)
        return add_travel_time_attributes(graph)

    graph = ox.graph_from_place(place_name, network_type=network_type)
    graph = add_travel_time_attributes(graph)
    _ensure_parent_dir(graph_file)
    ox.save_graphml(graph, graph_file)
    return graph


def resolve_nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Resolve a lat/lon pair to the nearest graph node."""

    return int(ox.distance.nearest_nodes(graph, X=float(lon), Y=float(lat)))


def compute_route(
    graph: nx.MultiDiGraph,
    origin_node: int,
    destination_node: int,
    algorithm: str = DEFAULT_ALGORITHM,
    weight: str = "length",
) -> list[int]:
    """Compute a route between two graph nodes."""

    selected_algorithm = _normalise_algorithm(algorithm)
    if selected_algorithm == "dijkstra":
        return nx.shortest_path(
            graph,
            source=origin_node,
            target=destination_node,
            weight=weight,
            method="dijkstra",
        )

    return nx.astar_path(
        graph,
        source=origin_node,
        target=destination_node,
        heuristic=lambda source, target: _euclidean_heuristic(graph, source, target),
        weight=weight,
    )


def route_metric(graph: nx.MultiDiGraph, route: Sequence[int], weight: str) -> float:
    """Measure a route across multiedges using the smallest weighted edge."""

    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        total += _shortest_multiedge_value(graph, u, v, weight)
    return total


def summarise_route(
    graph: nx.MultiDiGraph,
    route: Sequence[int],
    *,
    algorithm: str,
    origin_node: int,
    destination_node: int,
    include_route_nodes: bool = False,
) -> dict:
    """Convert a raw route to the stable payload expected by dispatch."""

    distance_m = route_metric(graph, route, "length")
    travel_time_sec = route_metric(graph, route, "travel_time")

    summary = {
        "status": "SUCCESS",
        "algorithm": _normalise_algorithm(algorithm),
        "origin_node": origin_node,
        "destination_node": destination_node,
        "distance_m": round(distance_m, 2),
        "distance_km": round(distance_m / 1000.0, 3),
        "estimated_travel_time_sec": round(travel_time_sec, 2),
        "estimated_travel_time_min": round(travel_time_sec / 60.0, 2),
        "route_nodes": len(route),
    }
    if include_route_nodes:
        summary["route"] = list(route)
    return summary


def connect_routing(
    incident_lat: float,
    incident_lon: float,
    facility_lat: float,
    facility_lon: float,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    graph: nx.MultiDiGraph | None = None,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    place_name: str = DEFAULT_PLACE_NAME,
    include_route_nodes: bool = False,
) -> dict:
    """Route between an incident and facility.

    Parameters use ``lat, lon`` order to match the dispatch notebook.
    """

    selected_algorithm = _normalise_algorithm(algorithm)
    graph = graph or load_or_build_graph(place_name=place_name, graph_path=graph_path)

    try:
        origin_node = resolve_nearest_node(graph, incident_lat, incident_lon)
        destination_node = resolve_nearest_node(graph, facility_lat, facility_lon)
        route = compute_route(
            graph,
            origin_node=origin_node,
            destination_node=destination_node,
            algorithm=selected_algorithm,
            weight="length",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError, TypeError) as exc:
        return {
            "status": "NO_ROUTE",
            "algorithm": selected_algorithm,
            "origin_node": None,
            "destination_node": None,
            "distance_m": None,
            "distance_km": None,
            "estimated_travel_time_sec": None,
            "estimated_travel_time_min": None,
            "route_nodes": 0,
            "error": str(exc),
        }

    return summarise_route(
        graph,
        route,
        algorithm=selected_algorithm,
        origin_node=origin_node,
        destination_node=destination_node,
        include_route_nodes=include_route_nodes,
    )


def benchmark_algorithms(
    scenarios: Iterable[tuple[CoordinatePair, CoordinatePair]],
    *,
    graph: nx.MultiDiGraph | None = None,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    place_name: str = DEFAULT_PLACE_NAME,
) -> pd.DataFrame:
    """Benchmark Dijkstra and A* across multiple incident/facility pairs."""

    graph = graph or load_or_build_graph(place_name=place_name, graph_path=graph_path)
    rows: list[dict] = []

    for scenario_id, (incident, facility) in enumerate(scenarios, start=1):
        for algorithm in ("dijkstra", "astar"):
            started_at = perf_counter()
            result = connect_routing(
                incident.lat,
                incident.lon,
                facility.lat,
                facility.lon,
                algorithm=algorithm,
                graph=graph,
            )
            runtime_sec = perf_counter() - started_at
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "incident_lat": incident.lat,
                    "incident_lon": incident.lon,
                    "facility_lat": facility.lat,
                    "facility_lon": facility.lon,
                    "algorithm": algorithm,
                    "status": result["status"],
                    "distance_km": result["distance_km"],
                    "estimated_travel_time_min": result["estimated_travel_time_min"],
                    "route_nodes": result["route_nodes"],
                    "runtime_sec": round(runtime_sec, 6),
                }
            )

    return pd.DataFrame(rows)


def save_benchmark_results(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist benchmark results to CSV."""

    output_file = Path(output_path)
    _ensure_parent_dir(output_file)
    results.to_csv(output_file, index=False)
    return output_file
