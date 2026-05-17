"""
FastAPI backend for the Smart Street Lighting Design System.

Exposes the pipeline as REST endpoints for the React frontend.

Usage:
    uvicorn api:app --reload --port 8000
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import AppData, run_design, parse_query
from core.database import (
    create_conversation, list_conversations, get_conversation,
    update_conversation_title, delete_conversation as db_delete_conversation,
    add_message, get_messages,
)
from data.temporal_analysis import get_weekday_weekend_profile
from llm.calculation_engine import P_CATEGORIES

# Lazy-load RAG components (only when LLM endpoint is called)
RAG_LOADED = False
QUERY_ENGINE = None


def load_rag():
    global RAG_LOADED, QUERY_ENGINE
    if RAG_LOADED:
        return
    try:
        from rag.ingest import load_existing_index, ingest_knowledge_base
        from rag.query_engine import create_query_engine
        try:
            index = load_existing_index()
        except Exception:
            index = ingest_knowledge_base()
        QUERY_ENGINE = create_query_engine(index=index)
        RAG_LOADED = True
        print("RAG + LLM loaded.")
    except Exception as e:
        print(f"RAG load failed: {e}")


# ============================================================
# Load data at startup
# ============================================================
DATA = AppData()

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title="Smart Street Lighting API",
    description="AI-powered street lighting design for Melbourne",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response models
# ============================================================
class DesignRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    location: Optional[str] = None
    pathway_length_m: Optional[float] = None
    pathway_width_m: Optional[float] = 3.0
    activity_level: Optional[str] = None
    location_type: Optional[str] = "park_path"
    use_llm: bool = True


class DesignResponse(BaseModel):
    design: dict
    calculation_report: str
    llm_report: Optional[str] = None
    sources: list = []
    spatial_context: dict = {}
    dimming_savings: dict = {}
    map_data: Optional[dict] = None
    safety_context: Optional[dict] = None
    conversation_id: Optional[str] = None


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/data/summary")
def data_summary():
    """Summary of loaded Melbourne data."""
    eff = DATA.ped_matched["efficiency"].value_counts().to_dict()
    top_sensors = []
    for _, row in DATA.sensor_summary.head(10).iterrows():
        top_sensors.append({
            "name": row["sensor_name"],
            "lat": round(row["Latitude"], 6),
            "lon": round(row["Longitude"], 6),
            "avg_traffic": round(row["avg_hourly_traffic"], 1),
            "max_traffic": int(row["max_hourly_traffic"]),
        })

    return {
        "pedestrian_sensors": int(DATA.ped_data["sensor_name"].nunique()),
        "pedestrian_records": len(DATA.ped_data),
        "streetlights": len(DATA.streetlight_data),
        "avg_lux": round(DATA.streetlight_data["lux_level"].mean(), 1),
        "efficiency_breakdown": eff,
        "top_sensors": top_sensors,
    }


@app.get("/api/data/categories")
def get_categories():
    """AS/NZS 1158 P-category reference data."""
    return {cat_id: cat_data for cat_id, cat_data in P_CATEGORIES.items()}


@app.get("/api/data/hourly-traffic")
def hourly_traffic():
    """Hourly pedestrian traffic profile."""
    profile = []
    for _, row in DATA.hourly_profile.iterrows():
        profile.append({
            "hour": int(row["hour"]),
            "avg_traffic": round(row["avg_traffic"], 1),
            "median_traffic": round(row["median_traffic"], 1),
            "peak_traffic": int(row["peak_traffic"]),
        })
    return {"profile": profile}


@app.get("/api/data/dimming-schedule")
def dimming_schedule():
    """Adaptive dimming schedule based on traffic patterns."""
    return {"schedule": DATA.dimming_schedule}


@app.get("/api/data/weekday-weekend")
def weekday_weekend():
    """Weekday vs weekend traffic profiles."""
    ww = get_weekday_weekend_profile(DATA.ped_temporal)
    result = {}
    for _, row in ww.iterrows():
        day_type = row["day_type"]
        if day_type not in result:
            result[day_type] = []
        result[day_type].append({
            "hour": int(row["hour"]),
            "avg_traffic": round(row["avg_traffic"], 1),
        })
    return result


@app.get("/api/data/streetlights")
def streetlights(limit: int = 500):
    """Streetlight positions and lux levels (for map)."""
    sample = DATA.streetlight_data.dropna(subset=["Latitude", "Longitude"]).head(limit)
    lights = []
    for _, row in sample.iterrows():
        lights.append({
            "lat": round(row["Latitude"], 6),
            "lon": round(row["Longitude"], 6),
            "lux": round(row.get("lux_level", 0) or 0, 1),
        })
    return {"lights": lights, "total": len(DATA.streetlight_data)}


@app.get("/api/data/sensors")
def sensors():
    """Pedestrian sensor positions and traffic summary (for map)."""
    result = []
    for _, row in DATA.sensor_summary.iterrows():
        if row["Latitude"] and row["Longitude"]:
            result.append({
                "name": row["sensor_name"],
                "lat": round(row["Latitude"], 6),
                "lon": round(row["Longitude"], 6),
                "avg_traffic": round(row["avg_hourly_traffic"], 1),
            })
    return {"sensors": result}


@app.post("/api/design", response_model=DesignResponse)
def create_design(req: DesignRequest):
    """Main design endpoint — runs the full pipeline and persists chat."""
    from core.intent import classify_intent, apply_modification

    # Auto-create conversation if not provided
    conv_id = req.conversation_id
    if not conv_id:
        title = req.query[:80] + ("..." if len(req.query) > 80 else "")
        conv = create_conversation(title)
        conv_id = str(conv["id"])

    # Save user message
    add_message(UUID(conv_id), "user", req.query)

    # Check for existing design in conversation (for multi-turn support)
    existing_design = None
    try:
        msgs = get_messages(UUID(conv_id))
        for msg in reversed(msgs):
            if msg.get("design_data") and msg["design_data"].get("design"):
                existing_design = msg["design_data"]["design"]
                break
    except Exception:
        pass

    # Classify intent
    intent_result = classify_intent(req.query, has_existing_design=existing_design is not None)
    intent = intent_result["intent"]

    # For questions about existing design, use RAG only (no recalculation)
    if intent == "question" and existing_design:
        load_rag()
        llm_report = None
        if QUERY_ENGINE:
            try:
                from rag.query_engine import query_with_context
                context = f"EXISTING DESIGN: {json.dumps(existing_design, default=str)}"
                response = query_with_context(QUERY_ENGINE, req.query, context)
                llm_report = response.response if hasattr(response, 'response') else str(response)
            except Exception as e:
                llm_report = f"Error answering question: {str(e)[:200]}"

        add_message(UUID(conv_id), "assistant", llm_report or "Unable to answer.", design_data={
            "design": existing_design,
        })
        return DesignResponse(
            design=existing_design,
            calculation_report="(Previous design — question only)",
            llm_report=llm_report,
            conversation_id=conv_id,
        )

    # For modifications, apply overrides from the modification intent
    extra_kwargs = {}
    if intent == "modify" and existing_design:
        modification = {
            "modify_target": intent_result.get("modify_target"),
            "message": req.query,
        }
        overrides = apply_modification(existing_design, modification)
        # Carry forward location from existing design
        if not req.location and existing_design.get("location"):
            extra_kwargs["location"] = existing_design["location"]
        if not req.pathway_length_m and existing_design.get("pathway_length_m"):
            extra_kwargs["pathway_length_m"] = existing_design["pathway_length_m"]
        if "activity_level" in overrides:
            extra_kwargs["activity_level"] = overrides["activity_level"]

    design, calc_report, spatial, dimming, full_context = run_design(
        DATA, req.query,
        location=req.location or extra_kwargs.get("location"),
        pathway_length_m=req.pathway_length_m or extra_kwargs.get("pathway_length_m"),
        pathway_width_m=req.pathway_width_m,
        activity_level=req.activity_level or extra_kwargs.get("activity_level"),
        location_type=req.location_type,
    )

    # LLM report (optional)
    llm_report = None
    sources = []
    if req.use_llm:
        load_rag()
        if QUERY_ENGINE:
            try:
                from rag.query_engine import query_with_context
                # Use the design-specific retrieval query for better RAG results
                retrieval_q = spatial.get("retrieval_query", req.query)
                response = query_with_context(QUERY_ENGINE, retrieval_q, full_context)
                llm_report = response.response if hasattr(response, 'response') else str(response)
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for node in response.source_nodes[:3]:
                        sources.append({
                            "source": node.metadata.get("source", "Unknown"),
                            "score": round(node.score, 3) if node.score else 0,
                            "text": node.text[:200],
                        })
            except Exception as e:
                llm_report = f"LLM error: {str(e)[:200]}"

    # Validate LLM report against calculation engine
    if llm_report and not llm_report.startswith("LLM error"):
        from core.validator import validate_report, append_verification_footer
        validation = validate_report(llm_report, design.summary_dict())
        if validation["corrected_report"]:
            llm_report = validation["corrected_report"]
        if validation["needs_regeneration"]:
            # Try once more with stricter prompt — fall back to calc report on second failure
            try:
                strict_query = (
                    f"CRITICAL: Use these EXACT numbers — "
                    f"{design.num_lights} lights, {design.spacing_m}m spacing, "
                    f"P-category {design.p_category}, ${design.annual_energy_cost_aud:.2f}/year. "
                    f"\n\n{full_context}\n\nUSER QUERY: {req.query}"
                )
                response2 = QUERY_ENGINE.query(strict_query)
                llm_report2 = response2.response if hasattr(response2, 'response') else str(response2)
                v2 = validate_report(llm_report2, design.summary_dict())
                if not v2["needs_regeneration"]:
                    llm_report = v2.get("corrected_report") or llm_report2
            except Exception:
                pass  # Keep the first (corrected) report
        llm_report = append_verification_footer(llm_report, design.summary_dict())

    # Save assistant message with design data
    assistant_content = llm_report or calc_report
    add_message(UUID(conv_id), "assistant", assistant_content, design_data={
        "design": design.summary_dict(),
        "sources": sources,
        "spatial_context": spatial,
        "dimming_savings": dimming,
    })

    return DesignResponse(
        design=design.summary_dict(),
        calculation_report=calc_report,
        llm_report=llm_report,
        sources=sources,
        spatial_context=spatial,
        dimming_savings=dimming,
        map_data=spatial.get("map_data"),
        safety_context=spatial.get("safety_context"),
        conversation_id=conv_id,
    )


# ============================================================
# Conversation endpoints
# ============================================================

class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationUpdate(BaseModel):
    title: str


@app.get("/api/conversations")
def api_list_conversations():
    """List all conversations, newest first."""
    convs = list_conversations()
    return [
        {
            "id": str(c["id"]),
            "title": c["title"],
            "created_at": c["created_at"].isoformat(),
            "updated_at": c["updated_at"].isoformat(),
        }
        for c in convs
    ]


@app.post("/api/conversations")
def api_create_conversation(req: ConversationCreate):
    """Create a new conversation."""
    conv = create_conversation(req.title)
    return {
        "id": str(conv["id"]),
        "title": conv["title"],
        "created_at": conv["created_at"].isoformat(),
        "updated_at": conv["updated_at"].isoformat(),
    }


@app.get("/api/conversations/{conversation_id}")
def api_get_conversation(conversation_id: str):
    """Get a conversation with all its messages."""
    conv = get_conversation(UUID(conversation_id))
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = get_messages(UUID(conversation_id))
    return {
        "id": str(conv["id"]),
        "title": conv["title"],
        "created_at": conv["created_at"].isoformat(),
        "updated_at": conv["updated_at"].isoformat(),
        "messages": [
            {
                "id": str(m["id"]),
                "role": m["role"],
                "content": m["content"],
                "design_data": m["design_data"],
                "created_at": m["created_at"].isoformat(),
            }
            for m in msgs
        ],
    }


@app.patch("/api/conversations/{conversation_id}")
def api_update_conversation(conversation_id: str, req: ConversationUpdate):
    """Update conversation title."""
    conv = update_conversation_title(UUID(conversation_id), req.title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": str(conv["id"]),
        "title": conv["title"],
        "updated_at": conv["updated_at"].isoformat(),
    }


@app.delete("/api/conversations/{conversation_id}")
def api_delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    if not db_delete_conversation(UUID(conversation_id)):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}
