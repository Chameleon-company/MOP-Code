import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
import io
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from supabase import create_client, Client
import requests

sys.path.insert(0, str(Path(__file__).parent / "LLM_pipeline"))

from LLM_pipeline.src.indexing import *
from LLM_pipeline.src.evaluating import *
from LLM_pipeline.src.chunking import *
from LLM_pipeline.src.answering import *

from crack_detection.pipeline import run_pipeline
from Metric_Generator.crackAnalyser import generateMetricReport
from reportGenerationHelpers import uploadReport

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Crack Detection RAG", layout="wide")

if __name__ == "__main__":
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    store = ChromaRAGStore(dir=str(Path(__file__).parent / "LLM_pipeline" / "src" / "chroma_db"), collection_name="crack_detection")
    reranker = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
    reranker.model = CrossEncoder(DEFAULT_RERANKER_MODEL, device='cpu')
    generator = AnswerGenerator(model=DEFAULT_GENERATION_MODEL)
    
    st.title("Bridge & Road Crack Detection System")
    st.markdown("_RAG-powered maintenance recommendations and analysis_")
    
    # Session states
    if "image_url" not in st.session_state:
        st.session_state.image_url = None
    if "mask_url" not in st.session_state:
        st.session_state.mask_url = None
    if "overlay_url" not in st.session_state:                          # NEW
        st.session_state.overlay_url = None                            # NEW
    if "report" not in st.session_state:
        st.session_state.report = None
    if "llm_results" not in st.session_state:
        st.session_state.llm_results = None
    if "metric_report" not in st.session_state:
        st.session_state.metric_report = None
    
    # Sidebar for input
    st.sidebar.header("1. Upload Image")
    uploaded_file = st.sidebar.file_uploader("Upload crack image", type=["jpg", "jpeg", "png"])
    auto_metrics = {}
    if uploaded_file is not None:
        with st.sidebar:
            with st.spinner("Generating crack metrics"):
                # Save to temp file
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # Upload images to Supastorage and return URLs
                result = run_pipeline(temp_path)
                st.session_state.image_url = result["original_url"]
                st.session_state.mask_url = result["mask_url"]
                st.session_state.overlay_url = result["overlay_url"]   # NEW
                
                # Download mask and generate metrics
                mask_response = requests.get(st.session_state.mask_url)
                mask_img = Image.open(io.BytesIO(mask_response.content))
                auto_metrics = generateMetricReport(mask_img, uploaded_file.name)
            st.success("Done generating metrics")
            
    st.sidebar.header("2. Inspection metrics")
    with st.sidebar.form("inspection_form"):
        image_id = st.text_input("Image ID", value=uploaded_file.name if uploaded_file else "bridge_01.jpg")
        crack_detected = st.checkbox("Crack Detected", value=auto_metrics.get("crack_detected", True))
        largest_crack_area_ratio = st.slider(
            "Estimated Crack Area Ratio", 0.0, 100.0,
            float(auto_metrics.get("largest_crack_area_ratio", 0.05)), step=0.01
        )
        largest_crack_length = st.number_input(
            "Estimated Crack Length (px)",
            value=float(auto_metrics.get("largest_crack_length", 438.0)), step=1.0, min_value=0.0
        )
        num_regions = st.number_input(
            "Number of Crack Regions",
            value=int(auto_metrics.get("num_crack_regions", 3)), min_value=1, step=1
        )
        damage_level = st.slider(
            "Damage Level", 0.0, 10.0,
            float(auto_metrics.get("damage_level", 2.5)), step=0.01
        )
        severity = st.selectbox(
            "Severity Level",
            ["Nan", "Superficial", "Minor", "Medium", "High", "Severe", "Catastrophic"],
            index=["Nan", "Superficial", "Minor", "Medium", "High", "Severe", "Catastrophic"].index(
                auto_metrics.get("severity", "Minor")
            ) if auto_metrics.get("severity") else 2
        )
        submitted = st.form_submit_button("Analyze Inspection")
    
    # Display images
    if st.session_state.image_url and st.session_state.mask_url:
        st.subheader("Uploaded Images")
        col_img, col_mask, col_overlay = st.columns(3)              # CHANGED: 2 → 3 columns
        with col_img:
            st.markdown("**Original Image**")
            st.image(st.session_state.image_url)
        with col_mask:
            st.markdown("**Binary Mask**")
            st.image(st.session_state.mask_url)
        with col_overlay:                                            # NEW column
            st.markdown("**Crack Overlay**")
            if st.session_state.overlay_url:
                st.image(st.session_state.overlay_url)
            else:
                st.info("Overlay not available")
        st.markdown("---")
        
    # Main analysis
    if submitted:
        # Create payload
        payload = InspectionPayload.from_dict({
            "image_id": image_id,
            "crack_detected": crack_detected,
            "num_crack_regions": num_regions,
            "largest_crack_area_ratio": largest_crack_area_ratio,
            "largest_crack_length": largest_crack_length,
            "severity": severity,
            "damage_level": damage_level,
        })
        
        # Semantic query
        prompt = payload.to_semantic_query()
        with st.expander("Semantic Query", expanded=True):
            st.info(prompt)
    
        # Generate report
        with st.spinner("Generating maintenance report..."):
            chroma_results = query_by_inspection(
                payload=payload,
                embedder=embedder,
                store=store,
                top_k=DEFAULT_RETRIEVAL_TOP_K,
            )
            
            report = generate_maintenance_report(
                payload=payload,
                embedder=embedder,
                store=store,
                chroma_results=chroma_results,
                reranker=reranker,
                generator=generator,
            )
            
        context_texts = [source.chunk_text for source in report.sources]
         
        with st.spinner("Evaluating report grounding..."):
            evaluator = LLMGroundingEvaluator()
            llm_results = evaluator.evaluate_report(report, context_texts)
        
        # Save to session state for the Save button
        st.session_state.report = report
        st.session_state.llm_results = llm_results
        st.session_state.metric_report = {
            "image_id": image_id,
            "num_crack_regions": num_regions,
            "largest_crack_area_ratio": largest_crack_area_ratio,
            "largest_crack_length": largest_crack_length,
            "severity": severity,
            "damage_level": damage_level,
            "image_url": st.session_state.image_url or "",
            "mask_url": st.session_state.mask_url or "",
            "overlay_url": st.session_state.overlay_url or "",      # NEW
        }
        
        # Three-column layout for results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Maintenance Report")
            
            sections = [
                ("1. FINDINGS SUMMARY", report.findings_summary),
                ("2. RISK ASSESSMENT", report.risk_assessment),
                ("3. RECOMMENDED REPAIR ACTIONS", report.repair_actions),
                ("4. INSPECTION SCHEDULE / NEXT STEPS", report.inspection_schedule),
            ]
            
            for title, content in sections:
                with st.expander(title, expanded=True):
                    st.write(content)
        with col2:
            st.subheader("Sources & Retrieved Context")
            with st.expander("Sources", expanded=True):
                for line in report.footnotes.splitlines()[1:]:
                    st.write(line)
                    
            with st.expander("Retrieved Context", expanded=True):
                context_texts = [source.chunk_text for source in report.sources]
                for i, text in enumerate(context_texts, 1):
                    with st.expander(f"Source {i}", expanded=True):
                        st.write(text)
        
        with col3:
            st.subheader("Hallucination Evaluation")
            with st.spinner("Evaluating report grounding..."):
                evaluator = LLMGroundingEvaluator()
                llm_results = evaluator.evaluate_report(report, context_texts)
            
            risk = llm_results['overall_hallucination_risk']
            if risk == "Low":
                st.success(f"Hallucination Risk: {risk}")
            elif risk == "Medium":
                st.warning(f"Hallucination Risk: {risk}")
            else:
                st.error(f"Hallucination Risk: {risk}")
            
            for section, result in llm_results['sections'].items():
                if section == "risk_assessment":
                    section = "RISK ASSESSMENT"
                elif section == "repair_actions":
                    section = "RECOMMENDED REPAIR ACTIONS"
                else:
                    section = "INSPECTION SCHEDULE / NEXT STEPS"
                
                with st.expander(section, expanded=True):
                    st.write(f"Hallucination risk: {result['hallucination_risk']}")
                    st.write(result['reasoning'])
                    
        # Save to database
        sections = st.session_state.llm_results['sections']
        db_record = {
            **st.session_state.metric_report,
            "report_status": "Generated",
            "risk_assessment": sections['risk_assessment']['reasoning'],
            "repair_actions": sections['repair_actions']['reasoning'],
            "inspection_schedule": sections['inspection_schedule']['reasoning'],
        }
        uploadReport(supabase, db_record)
                
    else:
        st.info("Upload the crack image in the sidebar and click 'Analyze Inspection' to begin.")