from src.indexing import *
from src.evaluating import *
from src.chunking import *
from src.answering import *


def main():
    embedder  = OpenAIEmbedder(model="text-embedding-3-small")
    store     = ChromaRAGStore(dir="src/chroma_db/", collection_name="crack_detection")
    reranker  = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
    generator = AnswerGenerator(model=DEFAULT_GENERATION_MODEL)

    #! Input
    width = 62
    payload = InspectionPayload.from_dict({
        "image_id":                  "bridge_01.jpg",
        "crack_detected":            True,
        "crack_area_ratio":          0.054,
        "estimated_crack_length_px": 438,
        "num_crack_regions":         3,
        "severity":                  "High",
        "model_confidence":          0.91,
    })
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
    print_maintenance_report(report)

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


if __name__ == "__main__":
    main()