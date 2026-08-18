"""Benchmark helper for the emergency routing engine."""

from __future__ import annotations

from pathlib import Path

from routing_engine import CoordinatePair, benchmark_algorithms, save_benchmark_results


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "tables" / "melbourne_algorithm_benchmark.csv"


def build_default_scenarios() -> list[tuple[CoordinatePair, CoordinatePair]]:
    """Melbourne-focused benchmark scenarios for Sprint 3 evaluation."""

    return [
        (
            CoordinatePair(lat=-37.8136, lon=144.9631),  # Melbourne CBD
            CoordinatePair(lat=-37.8067, lon=144.9767),  # St Vincent's Hospital
        ),
        (
            CoordinatePair(lat=-37.8008, lon=144.9669),  # Carlton Gardens
            CoordinatePair(lat=-37.7989, lon=144.9695),  # Fitzroy area
        ),
        (
            CoordinatePair(lat=-37.8183, lon=144.9540),  # Docklands
            CoordinatePair(lat=-37.8158, lon=144.9461),  # West Melbourne area
        ),
        (
            CoordinatePair(lat=-37.8409, lon=144.9465),  # South Melbourne
            CoordinatePair(lat=-37.8320, lon=144.9600),  # Southbank/CBD edge
        ),
    ]


def main() -> None:
    results = benchmark_algorithms(build_default_scenarios())
    output_path = save_benchmark_results(results, DEFAULT_OUTPUT)
    print(results.to_string(index=False))
    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
