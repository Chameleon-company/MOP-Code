# Final layer of the RAG pipeline. Evaluate report output
from .answering import *
from .indexing import *

class LLMGroundingEvaluator:
    """
    LLM-based evaluation of section grounding in retrieved context.
    Uses LLM to assess semantic coherence between report output and context.
    """
    
    def __init__(self, model: str = "gpt-5.4-nano"):
        self.client = OpenAI()
        self.model = model
    
    def evaluate_section(
        self,
        section_text: str,
        section_name: str,
        context_texts: List[str],
    ) -> dict:
        """
        Use LLM to evaluate if section is grounded in context.
        
        Args:
            section_text (str): Text of that particular section.
            section_name (str): Name of that particular section.  
            context_texts (List[str]): retrieved text from report.sources
        
        Returns:
            {
                "is_grounded": bool,
                "confidence": float (0.0-1.0),
                "reasoning": str,
                "hallucination_risk": "low" | "medium" | "high"
            }
        """
        context_joined = " ".join(context_texts)
        
        prompt = f"""Evaluate if this {section_name} is grounded in the provided context.

GENERATED SECTION:
{section_text}

RETRIEVED CONTEXT:
{context_joined}

Assess:
1. Are the main claims supported by the context?
2. Does the section contain information NOT in the context (hallucination)?
3. Overall grounding quality

Respond with JSON:
{{
    "is_grounded": true/false,
    "confidence": 0.0-1.0,
    "hallucination_risk": "low|medium|high",
    "reasoning": "brief explanation"
}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def evaluate_report(
        self,
        report: MaintenanceReport,
        context_texts: List[str],
    ) -> dict:
        """Evaluate all sections and return aggregated results."""
        
        section_fields = [
            ("risk_assessment", "Risk Assessment"),
            ("repair_actions", "Repair Actions"),
            ("inspection_schedule", "Inspection Schedule"),
        ]
        
        results = {}
        for field_name, label in section_fields:
            section_text = getattr(report, field_name, "")
            if section_text:
                results[field_name] = self.evaluate_section(
                    section_text, label, context_texts
                )
        
        # Aggregate
        hallucination_risks = [
            r.get("hallucination_risk", "high") 
            for r in results.values()
        ]
        
        overall_risk = (
            "high" if "high" in hallucination_risks
            else "medium" if "medium" in hallucination_risks
            else "low"
        )
        
        return {
            "sections": results,
            "overall_hallucination_risk": overall_risk,
        }

# Test
# embedder  = OpenAIEmbedder(model="text-embedding-3-small")
# store     = ChromaRAGStore(dir="./chroma_db", collection_name="crack_detection")
# reranker  = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
# generator = AnswerGenerator(model=DEFAULT_GENERATION_MODEL)

# # Engineering maintenance report
# payload = InspectionPayload.from_dict({
#     "image_id":                  "bridge_01.jpg",
#     "crack_detected":            True,
#     "crack_area_ratio":          0.054,
#     "estimated_crack_length_px": 438,
#     "num_crack_regions":         3,
#     "severity":                  "High",
#     "model_confidence":          0.91,
# })

# chroma_results = query_by_inspection(
#     payload=payload,
#     embedder=embedder,
#     store=store,
#     top_k=DEFAULT_RETRIEVAL_TOP_K,
# )

# report = generate_maintenance_report(
#     payload=payload,
#     embedder=embedder,
#     store=store,
#     chroma_results=chroma_results,
#     reranker=reranker,
#     generator=generator,
# )

# context_texts = [source.chunk_text for source in report.sources]
# context = " ".join(context_texts)

# evaluator = LLMGroundingEvaluator()
# llm_results = evaluator.evaluate_report(report, context_texts)

# print(f"Risk: {llm_results['overall_hallucination_risk']}")
# for section, result in llm_results['sections'].items():
#     print(f"  {section}: {result['hallucination_risk']} - {result['reasoning']}")

