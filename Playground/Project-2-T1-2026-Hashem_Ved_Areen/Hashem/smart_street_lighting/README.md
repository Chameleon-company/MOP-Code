# Smart Street Lighting Design System

AI-powered street lighting design for Melbourne parks, pathways, and streets.

**Capstone Project 2 — MOP (Melbourne Open Playground)**
Deakin University | SIT764 | T1 2026

**Team:** Hashem Mahmoud (Lead), Ved Sharma, Areen Yapa
**Mentor:** Ella Zarandi

---

## Overview

This system proposes optimal streetlight placement for urban areas using real Melbourne data, Australian lighting standards (AS/NZS 1158), and AI-generated design reports.

Given a natural language query like:
> "Design lighting for a 200m pathway in Fitzroy Gardens with high evening traffic"

The system produces:
- **Lighting layout:** 13 lights, 16m spacing, P9 category
- **Technology:** 30W LED, 3000K warm white
- **Energy estimate:** $328/year, 57% saving vs HPS
- **Justified design report** citing AS/NZS 1158 standards

## Architecture

```
Melbourne Open Data API    →  Spatial Analysis (k-NN haversine)
OpenStreetMap Overpass API →  Park Boundary + Pathway Geometry
Crime Statistics Agency    →  Safety Risk Assessment (CPTED)
                           →  Calculation Engine (AS/NZS 1158)
                           →  Output Validator (LLM cross-check)
                           →  RAG Knowledge Base (ChromaDB + nomic embeddings)
                           →  LLM Report Generation (local LLM via LM Studio)
                           →  Web UI (React + Tailwind + Leaflet + FastAPI)
```

| Layer | Component | Purpose |
|-------|-----------|---------|
| Data | Melbourne Open Data API | Real pedestrian counts + streetlight positions |
| Data | OpenStreetMap (Overpass) | Park boundaries, pathway geometries, intersections |
| Data | Crime Statistics Agency | Victoria offence rates per LGA for safety scoring |
| Spatial | k-NN (haversine) | Match sensors to lights, classify efficiency |
| Spatial | Geometry engine | Polyline light placement with intersection/entry awareness |
| Safety | CPTED risk analysis | Composite safety score → P-category upgrade |
| Calculation | AS/NZS 1158 engine | Deterministic: light count, spacing, energy, cost, budget |
| Validation | Output cross-check | Verify LLM report numbers match calc engine |
| RAG | ChromaDB + nomic embeddings | Retrieve standards, guidelines, CPTED, Melbourne context |
| LLM | Local LLM (LM Studio) | Generate structured design reports |
| UI | React + Tailwind + Leaflet | Interactive chat + map + data dashboards |

## Quick Start

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) with an LLM (e.g. `qwen2.5-7b-instruct`) and `text-embedding-nomic-embed-text-v1.5` loaded
- Node.js 18+ (for React dashboard)

### Installation

```bash
cd smart_street_lighting
pip install -r requirements.txt
cd frontend && npm install
```

### Configure LM Studio

1. Start LM Studio and load both models
2. Start the local server (default: `localhost:1234`)
3. Copy `.env.example` to `.env` and set the URL if not localhost:

```bash
cp .env.example .env
# Edit .env if LM Studio runs on a different machine
```

### Run the System

**Build the knowledge base (first time, or after editing docs):**
```bash
python ingest.py
```

**Web Dashboard**
```bash
# Terminal 1: API backend
uvicorn api:app --port 8000

# Terminal 2: React frontend
cd frontend && npm run dev
# Open http://localhost:3000
```

### Run Tests

```bash
python -m pytest tests/ -v                    # Unit tests (113 tests)
python evaluation/eval_calculations.py        # Calculation verification
python evaluation/eval_rag_retrieval.py       # RAG retrieval quality
python evaluation/eval_prompts.py             # Prompt strategy comparison
python evaluation/eval_rag_vs_norag.py        # RAG vs no-RAG baseline

# CLI smoke test — runs the full pipeline for a single hardcoded query
python uc01_park_pathway.py                   # Full pipeline (data + calc + RAG + LLM)
python uc01_park_pathway.py --calc-only       # Calculations only (no LLM needed)
```

## Project Structure

```
smart_street_lighting/
├── data/
│   ├── knowledge_base/          # RAG source documents (11 files)
│   │   ├── as_nzs_1158_p_categories.md
│   │   ├── as_nzs_1158_v_categories.md
│   │   ├── energy_efficiency_benchmarks.md
│   │   ├── energy_efficiency_detailed.md     # NEW: LED/HPS/Solar comparison
│   │   ├── cpted_principles.md               # NEW: Crime prevention lighting
│   │   ├── victorian_lighting_guidelines.md  # NEW: VicRoads, Parks Vic
│   │   ├── melbourne_parks_context.md
│   │   ├── pedestrian_crossings_school_zones.md
│   │   └── adaptive_dimming_smart_lighting.md
│   ├── cache/crime/             # Cached crime statistics CSV
│   ├── load_melbourne_data.py   # Melbourne Open Data API loader
│   ├── spatial_analysis.py      # k-NN matching, efficiency classification
│   ├── temporal_analysis.py     # Hourly traffic, dimming schedules
│   ├── osm_loader.py            # NEW: OpenStreetMap park/pathway resolution
│   ├── geometry.py              # NEW: Haversine, polyline placement, intersections
│   ├── safety_analysis.py       # NEW: Crime risk scoring per LGA
│   └── visualize.py             # Folium maps
├── llm/
│   └── calculation_engine.py    # AS/NZS 1158 calculations (+ geometry, budget, safety)
├── rag/
│   ├── lm_studio.py             # Custom LlamaIndex LLM + Embedding for LM Studio
│   ├── ingest.py                # Document ingestion to ChromaDB
│   └── query_engine.py          # RAG query engine with structured report prompt
├── core/
│   ├── pipeline.py              # Shared pipeline (data + OSM + safety + calc + dimming)
│   ├── database.py              # PostgreSQL conversation persistence
│   ├── validator.py             # NEW: LLM report cross-check vs calc engine
│   └── intent.py                # NEW: Multi-turn intent classification
├── evaluation/
│   ├── test_set.py              # 17 known-answer test cases
│   └── ...                      # Eval scripts
├── tests/
│   ├── fixtures/                # Cached OSM test data (no network needed)
│   ├── test_calculation_engine.py  # 26 unit tests
│   ├── test_data_loading.py        # 10 data/spatial/temporal tests
│   ├── test_osm_loader.py          # 13 geometry + OSM tests
│   ├── test_safety_analysis.py     # 18 crime/safety tests
│   ├── test_validator.py           # 13 validation tests
│   ├── test_intent.py              # 14 intent classification tests
│   ├── test_enhanced_engine.py     # 10 geometry/budget/safety engine tests
│   └── test_e2e.py                 # 9 end-to-end pipeline tests
├── frontend/                    # React + Tailwind + Leaflet dashboard
│   └── src/components/          # ChatPanel, DesignMap, TrafficPanel, EfficiencyPanel
├── uc01_park_pathway.py         # UC-01 CLI entry point
├── api.py                       # FastAPI backend for React frontend
├── UC01_Park_Pathway_Demo.ipynb # Demo notebook
└── requirements.txt
```

## Evaluation Results

| Evaluation | Result |
|---|---|
| RAG vs no-RAG | **275% improvement** in factual accuracy (0.75 vs 0.20) |
| RAG retrieval | **90% source accuracy**, 80% fact coverage |
| Prompt strategies | CoT best quality (0.43), few-shot best efficiency (60% pass rate) |
| Calculation engine | **7/7 verification tests pass** |
| Unit tests | **113 tests**, all passing |

See [evaluation/EVALUATION_REPORT.md](evaluation/EVALUATION_REPORT.md) for full details.

## Key Findings from Melbourne Data

- **8.1% of sensor locations are underlit** — high traffic but insufficient lighting
- **LED saves 57-66% energy** vs HPS with 9.6-year retrofit payback
- **36% additional saving** from adaptive dimming based on real hourly traffic patterns
- **Solar-only is marginal** for Melbourne (1.6 peak sun hours in winter)
- **3000K warm white** recommended per City of Melbourne ecological guidelines

## Data Sources

- [Melbourne Pedestrian Counting System](https://data.melbourne.vic.gov.au/) — hourly counts from 98 sensors
- [Melbourne Council Streetlights](https://data.melbourne.vic.gov.au/) — 9,500 lights with lux levels
- [OpenStreetMap](https://www.openstreetmap.org/) — park boundaries and pathway geometries (via Overpass API)
- [Crime Statistics Agency Victoria](https://www.crimestatistics.vic.gov.au/) — offence rates per LGA
- AS/NZS 1158 — Australian Standard for road and public space lighting
- Bureau of Meteorology — Melbourne solar irradiance data
- National Greenhouse Accounts — Victorian carbon emission factors

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| RAG Framework | LlamaIndex |
| Vector Store | ChromaDB (local, persistent) |
| Embeddings | nomic-embed-text-v1.5 (local via LM Studio) |
| LLM | Local via LM Studio (configurable in .env) |
| Spatial Analysis | scikit-learn (k-NN with haversine) |
| Maps | Leaflet.js + Folium |
| Charts | Recharts |
| Web UI | React + Tailwind |
| API | FastAPI |
| Testing | pytest |

## License

Capstone project — Deakin University / Melbourne Open Playground (Chameleon)
