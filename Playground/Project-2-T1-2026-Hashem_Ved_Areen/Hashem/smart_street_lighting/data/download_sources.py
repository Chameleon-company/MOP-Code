"""
Download publicly available source documents for the RAG knowledge base.

These are real PDFs, guides, and data files from government and industry
sources that the RAG system can ingest alongside our curated markdown docs.

Usage:
    python data/download_sources.py
"""

import os
import requests
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).parent / "downloaded_sources"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Publicly available documents relevant to our project
# Each entry: (filename, url, description)
SOURCES = [
    # Melbourne Open Data — direct CSV exports
    (
        "melbourne_pedestrian_sensors_info.csv",
        "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-sensor-locations/exports/csv?limit=500&format=csv",
        "Pedestrian sensor locations and metadata",
    ),

    # Australian Government — National Greenhouse Accounts Factors
    # This is a public government document
    (
        "nga_factors_summary.txt",
        None,  # Manual — see note below
        "National Greenhouse Accounts Factors — Victoria emission factors. "
        "Download from: https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors",
    ),

    # Bureau of Meteorology — Melbourne solar exposure
    (
        "bom_melbourne_solar_summary.txt",
        None,  # Manual — BOM data requires web access
        "Melbourne solar exposure data. "
        "Download from: http://www.bom.gov.au/climate/averages/tables/cw_086071.shtml",
    ),

    # City of Melbourne — Open Data catalog pages
    (
        "melbourne_streetlights_metadata.csv",
        "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/street-lights-with-emitted-lux-level-council-owned-lights-only/exports/csv?limit=100&format=csv",
        "Sample streetlight data with lux levels (first 100 rows for reference)",
    ),
]


def download_file(filename: str, url: str, description: str) -> bool:
    """Download a single file."""
    filepath = DOWNLOAD_DIR / filename
    if filepath.exists():
        print(f"  Already exists: {filename}")
        return True

    if url is None:
        print(f"  MANUAL: {filename} — {description}")
        # Create a placeholder with download instructions
        with open(filepath, "w") as f:
            f.write(f"# {filename}\n\n")
            f.write(f"This file must be downloaded manually.\n\n")
            f.write(f"{description}\n")
        return False

    try:
        print(f"  Downloading: {filename}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"  Saved: {filepath} ({len(resp.content):,} bytes)")
        return True
    except Exception as e:
        print(f"  FAILED: {filename} — {e}")
        return False


def download_all():
    """Download all available sources."""
    print("=" * 60)
    print("Downloading source documents for RAG knowledge base")
    print("=" * 60)

    success = 0
    manual = 0
    failed = 0

    for filename, url, description in SOURCES:
        result = download_file(filename, url, description)
        if url is None:
            manual += 1
        elif result:
            success += 1
        else:
            failed += 1

    print(f"\nResults: {success} downloaded, {manual} require manual download, {failed} failed")
    print(f"Files saved to: {DOWNLOAD_DIR}")

    # Update ingestion to include downloaded sources
    print(f"\nTo ingest these into the RAG system:")
    print(f"  1. Place any manually downloaded PDFs in {DOWNLOAD_DIR}")
    print(f"  2. Run: python uc01_park_pathway.py --ingest")
    print(f"  (The ingester reads .md, .txt, and .pdf files from data/knowledge_base/")
    print(f"   and data/downloaded_sources/)")


if __name__ == "__main__":
    download_all()
