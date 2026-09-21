import streamlit as st
import pandas as pd
from src.indexing import *
from src.evaluating import *
from src.chunking import *
from src.answering import *

st.set_page_config(page_title="Crack Detection RAG", layout="wide")

if __name__ == "__main__":
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    store = ChromaRAGStore(dir="src/chroma_db/", collection_name="crack_detection")
    reranker = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
    generator = AnswerGenerator(model=DEFAULT_GENERATION_MODEL)
    
    st.title("Bridge & Road Crack Detection System")
    st.markdown("_RAG-powered maintenance recommendations and analysis_")
    
    # Sidebar for input
    st.sidebar.header("Inspection Metrics")
    with st.sidebar.form("inspection_form"):
        image_id = st.text_input("Image ID", value="bridge_01.jpg")
        crack_detected = st.checkbox("Crack Detected", value=True)
        largest_crack_area_ratio = st.slider(
            "Largest Crack Area Ratio", 0.0, 1.0, 0.05, step=0.01
        )
        largest_crack_length = st.number_input(
            "Largest Crack Length (px)", value=438.0, step=1.0, min_value=0.0
        )
        num_regions = st.number_input(
            "Number of Crack Regions", value=3, min_value=1, step=1
        )
        damage_level = st.slider(
            "Damage Level", 0.0, 1.0, 0.25, step=0.01
        )
        severity = st.selectbox("Severity Level", ["Nan", "Superficial", "Minor", "Medium", "High", "Severe", "Catastrophic"], index=2)
        submitted = st.form_submit_button("Analyze Inspection")
        
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
        
        # Display input summary
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Severity", severity)
        with col2:
            st.metric("Damage Level", f"{damage_level:.0%}")
        with col3:
            st.metric("Crack Regions", num_regions)
        with col4:
            st.metric("Largest Crack Area Ratio", f"{largest_crack_area_ratio:.0%}")
        with col5:
            st.metric("Largest Crack Length", f"{largest_crack_length:.1f}px")
        
        st.markdown("---")
        
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
                
    else:
        st.info("Fill out the inspection metrics in the sidebar and click 'Analyze Inspection' to begin.")