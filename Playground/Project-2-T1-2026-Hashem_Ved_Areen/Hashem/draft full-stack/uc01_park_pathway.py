"""
UC-01: Park Pathway Lighting Design — End-to-End Pipeline

This script demonstrates the full system architecture:
1. Load real Melbourne data (pedestrian counts + streetlight locations)
2. Analyze spatial context for the target location
3. Run deterministic calculations (AS/NZS 1158 standards)
4. Retrieve relevant knowledge base context via RAG
5. Generate an explained design report via LLM (local LM Studio)

Usage:
    python uc01_park_pathway.py              # Full pipeline with default query
    python uc01_park_pathway.py --ingest     # Re-ingest knowledge base first
    python uc01_park_pathway.py --query "your custom query"
    python uc01_park_pathway.py --calc-only  # Skip LLM, show calculations only
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.load_melbourne_data import load_pedestrian_data, load_streetlight_data, get_sensor_summary
from data.spatial_analysis import match_sensors_to_lights, analyze_lighting_efficiency, get_area_lighting_context
from data.osm_loader import resolve_pathway
from llm.calculation_engine import design_lighting, format_design_report
from rag.ingest import ingest_knowledge_base, load_existing_index, CHROMA_DB_DIR
from rag.query_engine import create_query_engine, query_with_context
from core.pipeline import bounds_from_osm_boundary


UC01_DEFAULT_QUERY = (
    "Design lighting for a 200m pathway in Fitzroy Gardens used heavily by "
    "pedestrians in the evening, with a focus on safety and energy efficiency."
)


def analyze_spatial_context(sensor_summary, location_name="Fitzroy Gardens", bounds=None):
    """
    Find relevant pedestrian sensors near the target location and
    estimate activity level from real data.

    Args:
        sensor_summary: DataFrame with per-sensor traffic stats.
        location_name: Human-readable location name.
        bounds: (lat_min, lat_max, lon_min, lon_max) from OSM boundary.
    """
    print(f"\nAnalyzing pedestrian data near '{location_name}'...")

    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        nearby = sensor_summary[
            (sensor_summary["Latitude"] >= lat_min) &
            (sensor_summary["Latitude"] <= lat_max) &
            (sensor_summary["Longitude"] >= lon_min) &
            (sensor_summary["Longitude"] <= lon_max)
        ]
    else:
        nearby = sensor_summary.head(0)

    if not nearby.empty:
        avg_traffic = nearby["avg_hourly_traffic"].mean()
        max_traffic = nearby["max_hourly_traffic"].max()
        sensors = nearby["sensor_name"].tolist()
        print(f"  Found {len(nearby)} sensors: {sensors}")
        print(f"  Average hourly traffic: {avg_traffic:.0f}")
        print(f"  Peak hourly traffic: {max_traffic:.0f}")
        return avg_traffic, max_traffic, sensors
    else:
        cbd_avg = sensor_summary["avg_hourly_traffic"].median()
        print(f"  No sensors found in bounds for '{location_name}'.")
        print(f"  Using city-wide median as proxy: {cbd_avg:.0f} pedestrians/hour")
        return cbd_avg, cbd_avg * 2, []


def main():
    parser = argparse.ArgumentParser(description="UC-01: Park Pathway Lighting Design")
    parser.add_argument("--ingest", action="store_true", help="Re-ingest knowledge base")
    parser.add_argument("--query", type=str, default=UC01_DEFAULT_QUERY)
    parser.add_argument("--calc-only", action="store_true", help="Skip LLM, show calculations only")
    parser.add_argument("--top-k", type=int, default=5, help="RAG retrieval top-k")
    parser.add_argument("--length", type=float, default=200.0, help="Pathway length in metres")
    parser.add_argument("--width", type=float, default=3.0, help="Pathway width in metres")
    parser.add_argument("--location", type=str, default="Fitzroy Gardens")
    args = parser.parse_args()

    # ========================================
    # STEP 1: Load Real Melbourne Data
    # ========================================
    print("=" * 60)
    print("STEP 1: Loading Melbourne Open Data")
    print("=" * 60)
    ped_data = load_pedestrian_data(limit=2000)
    streetlight_data = load_streetlight_data(limit=5000)
    sensor_summary = get_sensor_summary(ped_data)

    print(f"\n  Total pedestrian sensors: {len(sensor_summary)}")
    print(f"  Total streetlights loaded: {len(streetlight_data)}")
    print(f"  Top 5 busiest sensors:")
    for _, row in sensor_summary.head(5).iterrows():
        print(f"    {row['sensor_name']}: avg {row['avg_hourly_traffic']:.0f}/hr")

    # ========================================
    # STEP 2: OSM Resolution + Spatial Analysis
    # ========================================
    print()
    print("=" * 60)
    print("STEP 2: OSM Resolution + Spatial Analysis")
    print("=" * 60)

    # Resolve park boundary from OpenStreetMap
    print(f"\nResolving '{args.location}' from OpenStreetMap...")
    osm_data = resolve_pathway(args.location)
    osm_bounds = None
    if osm_data and osm_data.get("park_boundary"):
        osm_bounds = bounds_from_osm_boundary(osm_data["park_boundary"])
        print(f"  OSM boundary resolved: {osm_bounds}")
    else:
        print(f"  OSM resolution failed for '{args.location}' — spatial context unavailable.")

    # Match sensors to nearest streetlights
    print("\nMatching pedestrian sensors to nearest streetlights...")
    ped_data = match_sensors_to_lights(ped_data, streetlight_data)
    ped_data = analyze_lighting_efficiency(ped_data)

    # Get area-specific context using OSM-derived bounds
    if osm_bounds:
        area_context = get_area_lighting_context(
            ped_data, streetlight_data,
            lat_min=osm_bounds[0], lat_max=osm_bounds[1],
            lon_min=osm_bounds[2], lon_max=osm_bounds[3],
            area_name=args.location,
        )
        print(f"\n  Area context for {args.location}:")
        for k, v in area_context.items():
            print(f"    {k}: {v}")
    else:
        area_context = {
            "area_name": args.location,
            "num_streetlights": 0,
            "avg_lux_level": None,
            "avg_pedestrian_count": 0,
            "efficiency_breakdown": None,
        }
        print(f"\n  No spatial context available for {args.location}.")

    # Determine activity level from real data
    avg_traffic, max_traffic, nearby_sensors = analyze_spatial_context(
        sensor_summary, args.location, bounds=osm_bounds
    )

    # ========================================
    # STEP 3: Deterministic Calculations
    # ========================================
    print()
    print("=" * 60)
    print("STEP 3: Lighting Design Calculations (AS/NZS 1158)")
    print("=" * 60)
    design = design_lighting(
        location_name=f"{args.location} Main Pathway",
        pathway_length_m=args.length,
        pathway_width_m=args.width,
        location_type="park_path",
        avg_pedestrian_count=avg_traffic,
    )

    calc_report = format_design_report(design)
    print(calc_report)

    if args.calc_only:
        # Save and exit
        save_output(args, design, calc_report, llm_report=None)
        return

    # ========================================
    # STEP 4: RAG Retrieval + LLM Report
    # ========================================
    print()
    print("=" * 60)
    print("STEP 4: RAG Knowledge Retrieval + LLM Report Generation")
    print("=" * 60)

    # Load or create RAG index
    if args.ingest or not CHROMA_DB_DIR.exists():
        print("Ingesting knowledge base...")
        index = ingest_knowledge_base()
    else:
        try:
            index = load_existing_index()
        except Exception:
            print("No existing index found. Running ingestion...")
            index = ingest_knowledge_base()

    # Create query engine
    query_engine = create_query_engine(index=index, similarity_top_k=args.top_k)

    # Run query with calculation context + spatial context
    import json
    spatial_summary = (
        f"\nSPATIAL ANALYSIS CONTEXT:\n"
        f"  Nearby pedestrian sensors: {len(nearby_sensors)} ({', '.join(nearby_sensors[:5]) if nearby_sensors else 'none in area'})\n"
        f"  Average hourly pedestrian traffic: {avg_traffic:.0f}\n"
        f"  Peak hourly traffic: {max_traffic:.0f}\n"
        f"  Area streetlights: {area_context.get('num_streetlights', 'N/A')}\n"
        f"  Area avg lux: {area_context.get('avg_lux_level', 'N/A')}\n"
        f"  Efficiency: {json.dumps(area_context.get('efficiency_breakdown', {}))}\n"
    )
    full_context = calc_report + "\n" + spatial_summary

    print(f"\nQuery: {args.query}")
    print("Generating LLM report (local LLM via LM Studio)...")
    print("-" * 60)

    response = query_with_context(
        query_engine=query_engine,
        user_query=args.query,
        calculation_context=full_context,
    )

    llm_report = response.response if hasattr(response, 'response') else str(response)

    print("\n" + "=" * 60)
    print("DESIGN REPORT (LLM-Generated with RAG Context)")
    print("=" * 60)
    print(llm_report)

    # Print retrieved sources
    if hasattr(response, 'source_nodes') and response.source_nodes:
        print("\n" + "-" * 60)
        print("KNOWLEDGE BASE SOURCES RETRIEVED")
        print("-" * 60)
        for i, node in enumerate(response.source_nodes, 1):
            score = f"{node.score:.4f}" if node.score else "N/A"
            source = node.metadata.get("source", "Unknown")
            print(f"  [{i}] Score: {score} | Source: {source}")
            print(f"      {node.text[:150].replace(chr(10), ' ')}...")

    save_output(args, design, calc_report, llm_report)


def save_output(args, design, calc_report, llm_report):
    """Save the full output to a file."""
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "uc01_latest_output.txt"

    with open(output_file, "w") as f:
        f.write(f"UC-01: Park Pathway Lighting Design\n")
        f.write(f"Location: {args.location}\n")
        f.write(f"Query: {args.query}\n")
        f.write(f"{'='*60}\n\n")
        f.write("CALCULATION ENGINE OUTPUT\n")
        f.write(f"{'='*60}\n")
        f.write(calc_report)
        if llm_report:
            f.write(f"\n\n{'='*60}\n")
            f.write("LLM-GENERATED DESIGN REPORT\n")
            f.write(f"{'='*60}\n")
            f.write(llm_report)

    print(f"\nOutput saved to: {output_file}")

    # Also save design summary as JSON
    import json
    json_file = output_dir / "uc01_latest_design.json"
    with open(json_file, "w") as f:
        json.dump(design.summary_dict(), f, indent=2)
    print(f"Design JSON saved to: {json_file}")


if __name__ == "__main__":
    main()
