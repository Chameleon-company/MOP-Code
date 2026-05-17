# RAG Knowledge Base

These documents are ingested into ChromaDB as the RAG knowledge base.
The LLM retrieves relevant chunks from these files to ground its design reports.

## Documents

| File | Source | Content |
|------|--------|---------|
| `as_nzs_1158_p_categories.md` | AS/NZS 1158.3.1 | P1-P12 pedestrian lighting categories with lux, uniformity, spacing rules |
| `as_nzs_1158_v_categories.md` | AS/NZS 1158.1.1 | V1-V5 vehicular road lighting + intersection/crossing requirements |
| `energy_efficiency_benchmarks.md` | Multiple sources | LED vs HPS comparison, Melbourne solar data, carbon factors, payback periods |
| `melbourne_parks_context.md` | Melbourne Open Data | Park-specific context: Fitzroy Gardens, Royal Park, Princes Park, etc. |
| `pedestrian_crossings_school_zones.md` | AS/NZS 1158.4 | Crossing lighting, school zones, residential street categories |
| `adaptive_dimming_smart_lighting.md` | Industry best practice | Dimming schedules, smart controls, dark sky/ecological compliance |

## Ingestion

Documents are chunked at 1024 tokens with 100-token overlap, embedded using
nomic-embed-text-v1.5 (local via LM Studio), and stored in ChromaDB.

To re-ingest after adding/editing documents:
```bash
python uc01_park_pathway.py --ingest
```

## Adding New Documents

1. Add a `.md` or `.txt` file to this directory
2. Use clear section headers (the chunker splits on markdown headings)
3. Include source attribution at the top of the file
4. Re-run ingestion
