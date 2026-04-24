from .src.indexing import *
from .src.evaluating import *
from .src.chunking import *
from .src.answering import *
import streamlit as st


def report_generation(metric_report: dict):
    embedder  = OpenAIEmbedder(model="text-embedding-3-small")
    store     = ChromaRAGStore(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)) , "src/chroma_db/"), collection_name="crack_detection")
    reranker  = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
    generator = AnswerGenerator(model=DEFAULT_GENERATION_MODEL)

    #! Example Input
    width = 62
    payload = InspectionPayload.from_dict(metric_report)
    print(f"\n{'='*width}")
    print(f" CRACK METRICS IN JSON FORMAT")
    print(f"{'='*width}")
    print(f"  {payload}")
    
    prompt = payload.to_semantic_query()
    print(f"\n{'='*width}")
    print(f"  TRANSFORM CRACK METRICS INTO NATURAL LANGUAGE QUERY")
    print(f"{'='*width}")
    print(f"  {prompt}")
    
    #! Report output
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
    #print_maintenance_report(report)

    #print(f"Finding Summary: {report.findings_summary}")
    #print(f"Risk Assessment: {report.risk_assessment}")
    #print(f"Recommended Repair Action: {report.repair_actions}")
    #print(f"Inspection Schedule: {report.inspection_schedule}")

    

    #! Evaluation
    context_texts = [source.chunk_text for source in report.sources]
    context = " ".join(context_texts)
    print(f"\n{'='*width}")
    print(f"  RETRIEVED CONTEXT FROM VECTOR STORE AND RERANKING")
    print(f"{'='*width}")
    print(f"  {context}")

    print(f"\n{'='*width}")
    print(f"  EVALUATION OF REPORT")
    print(f"{'='*width}")
    
    evaluator = LLMGroundingEvaluator()
    llm_results = evaluator.evaluate_report(report, context_texts)

    print(f"  Risk of hallucination: {llm_results['overall_hallucination_risk']}")
    for section, result in llm_results['sections'].items():
        print(f"  {section}: {result['hallucination_risk']} - {result['reasoning']}\n")

    
    return llm_results