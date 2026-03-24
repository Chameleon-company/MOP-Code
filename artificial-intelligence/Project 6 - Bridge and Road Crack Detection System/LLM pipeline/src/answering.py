"""
Third stage of RAG pipeline: rerank -> generate -> cite.

`generate_maintenance_report()` - engineering maintenance report
- Takes a structured InspectionPayload (from indexing.py) plus the ChromaDB results from query_by_inspection(), reranks the candidates, and generates a formal engineering report with four sections:
  - Findings Summary
  - Risk Assessment
  - Recommended Repair Actions
  - Inspection Schedule / Next Steps
  Every claim is cited back to the knowledge base passage it came from, which include inline [N] citations and footnotes.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from openai import APIError, OpenAI
from sentence_transformers import CrossEncoder

from .indexing import *

# Lightweight Cross-encoder model used for reranking that runs on CPU.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_GENERATION_MODEL = "gpt-5.4-nano"

# Number of candidates fetch from ChromaDB before reranking. The more fetched the richer the reranker pool.
DEFAULT_RETRIEVAL_TOP_K = 20

# Number of top-reranked chunks to actually include in the LLM prompt. More context = more accurate answers but higher token cost.
DEFAULT_RERANK_TOP_N = 5

# gpt5 nano has 400k max token
MAX_ANSWER_TOKENS = 20000

@dataclass
class AnswerSource:
    """
    Metadata for a single reference in SOURCES.
    This is what gets rendered in the footnote bibliography.
    Each AnswerSource instance represent only one source reference. 
    """
    index: int # citation number, e.g. [1]
    doc_title: str
    section_path: str
    chunk_type: str
    chunk_text: str # The raw passage the LLM used
    
@dataclass
class Answer:
    """
    A complete RAG answer ready to display to the user.
    A Answer object contains answer_text + footnotes and the user sees the complete cited answer.

    Args:
        query : User question. This is the payload inspection
        answer_text: LLM-generated answer with in-text [N] citation markers.
        sources: Ordered list of AnswerSource objects, one per cited chunk.
        footnotes: Pre-formatted footnote block (plain text), ready to append.
        reranked_scores: Raw cross-encoder scores for each source. For debugging.
    """
    query: str
    answer_text: str
    sources: List[AnswerSource] = field(default_factory=list)
    footnotes: str = ""
    reranked_scores: List[float] = field(default_factory=list)
    token_count: int = 0

class CrossEncoderReranker:
    """
    Rerank a list of (text, metadata) candidates using a local cross-encoder.
    Reason to use rerank is as follow.
    ChromaDB uses approximate nearest-neighbour search on embedding vectors.
    Embeddings compress meaning into a fixed-size vector. This causes two texts
    that can be close in the vector space but don't provide any meaningful answer to the question.
    A cross-encoder solves this by reading the query and each candidate as a pair, and score
    their relevance. This is more accurate but slow on a full corpus. 
    Solution is to use a fast vector search to narrow to 20 candidates, then use 
    cross-encoder to pick the best 5. 
    """
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: List[dict],
        top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> List[tuple]:
        """
        Score every (query, document) pair and return top_n result
        
        Args:
            query: The user's question.
            documents: Raw chunk texts returned by ChromaDB.
            metadatas: Parallel metadata dicts, one per document.
            top_n: How many results to keep after reranking.

        Returns:
            A list of (score, document, metadata) tuples that is sorted by score descending (most relevant first) and capped at top_n.
        """
        if not documents:
            print("Reranker received an empty document list.")
            return []
        
        # Build (query, passage) pairs
        pairs = [(query, doc) for doc in documents]
        
        # Predict the relevance score. High score = more relevant
        scores = self.model.predict(pairs)
        
        # Zip scores with their corresponding document and metadata
        scored = list(zip(scores, documents, metadatas))
        
        # Sort descending by score so the best candidates come first.
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[:top_n]
        return top

def _build_report_prompt(
    payload: InspectionPayload,
    reranked: List[tuple],
) -> str:
    """
    Build the engineering report prompt from an inspection payload and the top-N reranked knowledge base passages.
    
    Prompt design decision:
    - Payload values (crack_area_ratio, confidence, etc.) are injected into LLM.
    - Explicit tell LLM to specific section headers (FINDING SUMMARY, RISK ASSESSMENT, RECOMMENDED REPAIR SECTION, INSPECTION SCHEDULE / NEXT STEPS) in the prompt to make things easier to parse.
    - LLM is instructed to cite (e.g. [1], [2]) every claim to prevent hallucination.
    - Use engineering tone. 
    """
    # Build the numbered context block from the reranked passages.
    context_lines = []
    for i, (_, doc_text, meta) in enumerate(reranked, start=1):
        label = (
            f"[{i}] {meta.get('doc_title', 'Unknown')} — "
            f"{meta.get('section_path', 'Unknown section')}")
        context_lines.append(f"{label}\n{doc_text}")

    context_block = "\n\n".join(context_lines)
    
    # Format the inspection payload as a clear data block.
    payload_block = (
        f"  Image ID                  : {payload.image_id}\n"
        f"  Crack Detected            : {payload.crack_detected}\n"
        f"  Crack Area Ratio          : {payload.crack_area_ratio:.4f}\n"
        f"  Estimated Crack Length    : {payload.estimated_crack_length_px} px\n"
        f"  Number of Crack Regions   : {payload.num_crack_regions}\n"
        f"  Severity                  : {payload.severity}\n"
        f"  Model Confidence          : {payload.model_confidence:.2f}"
    )

    prompt = (
        f"INSPECTION DATA:\n"
        f"{payload_block}\n\n"
        f"KNOWLEDGE BASE PASSAGES:\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"TASK:\n"
        f"You are a structural engineering expert writing an official maintenance "
        f"report based on the inspection data and knowledge base passages above.\n\n"
        f"Write the report using formal technical language, third person, and "
        f"passive voice where appropriate.  Do NOT use any information that is "
        f"not present in either the inspection data or the knowledge base passages.\n\n"
        f"After every factual claim that comes from the knowledge base, add the "
        f"citation marker [N] matching the passage number it came from.  "
        f"Claims derived solely from the inspection data do not need a citation.\n\n"
        f"Structure your response with EXACTLY these four section headers, "
        f"each on its own line followed by its content:\n\n"
        f"FINDINGS SUMMARY:\n"
        f"<Describe what was detected: crack presence, area ratio, length, "
        f"number of regions, and model confidence.>\n\n"
        f"RISK ASSESSMENT:\n"
        f"<Evaluate the structural risk implied by the severity level and "
        f"crack metrics.  Reference knowledge base passages where applicable.>\n\n"
        f"RECOMMENDED REPAIR ACTIONS:\n"
        f"<List concrete repair steps appropriate for this severity level, "
        f"citing knowledge base passages for each step.>\n\n"
        f"INSPECTION SCHEDULE / NEXT STEPS:\n"
        f"<Specify re-inspection intervals, follow-up requirements, and any "
        f"escalation criteria, citing knowledge base passages.>\n\n"
        f"At the very end, output a SOURCES section:\n\n"
        f"SOURCES:\n"
        f"[1] <title> — <section>\n"
        f"[2] <title> — <section>\n"
        f"...\n\n"
        f"Write the report now:"
    )
    return prompt

def _parse_report_section(raw_text: str, header: str) -> str:
    """
    Extract the content of a named section from the LLM's report output.
    
    Args:
        raw_text: Raw output from LLM
        header: Header of each section. E.g. FINDINGS SUMMARY, RISK ASSESSMENT, ...
    """
    
    # Match the header, then capture text until the next section header
    # (an ALL-CAPS word followed by optional words and a colon) or end of string.
    pattern = re.compile(
        rf"{re.escape(header)}\s*:\s*\n(.*?)(?=\n[A-Z][A-Z\s/]+:\s*\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(raw_text)
    return match.group(1).strip() if match else ""

class AnswerGenerator:
    """
    Call LLM model with a grounded prompt and returns the raw response text.
    """
    # System message/prompt that model adopts for every request
    SYSTEM_MESSAGE = (
        "You are a precise research assistant. "
        "You answer questions strictly from the provided context passages. "
        "You never fabricate information. "
        "You always cite the source passage number(s) inline using [N] markers."
    )
    
    def __init__(self, model: str = DEFAULT_GENERATION_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str) -> str:
        """
        Send the prompt to the LLM and return the raw response string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=MAX_ANSWER_TOKENS,
                temperature=0 
            )
            raw_text = response.choices[0].message.content.strip()
            return raw_text

        except APIError as exc:
            raise RuntimeError(f"OpenAI API error during generation: {exc}") from exc

@dataclass
class MaintenanceReport:
    """
    A structured engineering maintenance report produced from an inspection payload and grounded in the knowledge base.

    Args:
        image_id: Identifier of the inspected image/asset.
        severity: Severity label from the inspection payload.
        generated_at: ISO-8601 UTC timestamp of report generation.
        findings_summary: What was detected (e.g. cracks, area, confidence, etc).
        risk_assessment: Engineering risk evaluation based on the findings.
        repair_actions: Concrete repair steps cited from the knowledge base.
        inspection_schedule: Re-inspection timeline and follow-up requirements.
        sources: Knowledge base passages cited in the report.
        footnotes: Pre-formatted bibliography block.
        reranked_scores: Cross-encoder scores for each source passage.
        raw_text: The complete raw LLM output before parsing.
    """
    image_id: str
    severity: str
    generated_at: str
    findings_summary: str
    risk_assessment: str
    repair_actions: str
    inspection_schedule: str
    sources: List[AnswerSource] = field(default_factory=list)
    footnotes: str = ""
    reranked_scores: List[float] = field(default_factory=list)
    raw_text: str = ""

def generate_maintenance_report(
    payload: InspectionPayload,
    embedder: OpenAIEmbedder,
    store: ChromaRAGStore,
    chroma_results: dict,
    reranker: "CrossEncoderReranker",
    generator: "AnswerGenerator",
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> MaintenanceReport:
    """
    Generate a structured engineering maintenance report for an inspection.
    
    This takes a structured inspection payload and the ChromaDB results from query_by_inspection(),
    then produces a formal four-section report grounded in the knowledge base.

    Pipeline:
        1. Unpack ChromaDB candidates (documents + metadata).
        2. Rerank with the cross-encoder using the payload's semantic query.
        3. Build an engineering-style prompt that injects the payload data and
        the top-N knowledge base passages.
        4. Generate the report with LLM.
        5. Parse the four sections and citations into a MaintenanceReport object.
    
    Args:
        payload: InspectionPayload from indexing.InspectionPayload.from_dict() or .from_file().
        chroma_results: dict returned by indexing.query_by_inspection().
        reranker: Initialised CrossEncoderReranker.
        generator: Initialised AnswerGenerator.
        rerank_top_n: How many top chunk to include in the prompt.

    Returns:
        A fully populated MaintenanceReport dataclass.
    """
    
    # Unpack ChromaDB results
    chroma_results = query_by_inspection(
        payload=payload,
        embedder=embedder,
        store=store,
        top_k=rerank_top_n,
    )
    
    documents: List[str] = chroma_results.get("documents", [[]])[0]
    metadatas: List[dict] = chroma_results.get("metadatas", [[]])[0]

    generated_at = datetime.now(timezone.utc).isoformat()
    
    if not documents:
        print(f"No knowledge base passages/chunks found for {payload.image_id}. Report will contain inspection data only.")
        
        # Return a minimal report containing only raw payload data
        return MaintenanceReport(
                image_id=payload.image_id,
                severity=payload.severity,
                generated_at=generated_at,
                findings_summary=(
                    f"Crack detected in image '{payload.image_id}'. "
                    f"Crack area ratio: {payload.crack_area_ratio:.4f}. "
                    f"Estimated crack length: {payload.estimated_crack_length_px}px "
                    f"across {payload.num_crack_regions} region(s). "
                    f"Model confidence: {payload.model_confidence:.2f}. "
                    f"Severity assessed as {payload.severity}."
                ),
                risk_assessment=("Risk assessment could not be completed as no relevant knowledge base passages were retrieved. Manual engineering review required."),
                repair_actions=("Repair actions could not be determined as no relevant knowledge base passages were retrieved. Consult engineering standards directly."),
                inspection_schedule=(f"Inspection schedule could not be determined as no relevant knowledge base passages were retrieved. Apply default protocol for {payload.severity} severity defects."),
            )
    
    # Rerank using the payload's semantic query
    semantic_query = payload.to_semantic_query() 
    reranked = reranker.rerank(
        query=semantic_query,
        documents=documents,
        metadatas=metadatas,
        top_n=rerank_top_n,
    )
    
    # Build the engineering report prompt
    prompt = _build_report_prompt(payload=payload, reranked=reranked)
    
    # Generate
    raw_text = generator.generate(prompt)
    
    # Parse each section
    findings_summary    = _parse_report_section(raw_text, "FINDINGS SUMMARY")
    risk_assessment     = _parse_report_section(raw_text, "RISK ASSESSMENT")
    repair_actions      = _parse_report_section(raw_text, "RECOMMENDED REPAIR ACTIONS")
    inspection_schedule = _parse_report_section(raw_text, "INSPECTION SCHEDULE / NEXT STEPS")
    
    # Build citations from the full report text to prevent hallucination since LLM may make up or paraphrase citations.  
    cited_indices = {
        int(n) for n in re.findall(r"\[(\d+)\]", raw_text)
    }

    sources: List[AnswerSource] = []
    scores: List[float] = []

    for i, (score, doc_text, meta) in enumerate(reranked, start=1):
        if i not in cited_indices:
            continue
        sources.append(
            AnswerSource(
                index=i,
                doc_title=meta.get("doc_title", "Unknown"),
                section_path=meta.get("section_path", ""),
                chunk_type=meta.get("chunk_type", ""),
                chunk_text=doc_text,
            )
        )
        scores.append(float(score))
    
    # Build the footnote bibliography from authoritative metadata. 
    # LLM may paraphrase the SOURCE block. 
    footnote_lines = ["SOURCES:"]
    for src in sources:
        line = f"  [{src.index}] {src.doc_title} — {src.section_path}"
        footnote_lines.append(line)
    footnotes = "\n".join(footnote_lines) if sources else ""
    
    return MaintenanceReport(
        image_id=payload.image_id,
        severity=payload.severity,
        generated_at=generated_at,
        findings_summary=findings_summary,
        risk_assessment=risk_assessment,
        repair_actions=repair_actions,
        inspection_schedule=inspection_schedule,
        sources=sources,
        footnotes=footnotes,
        reranked_scores=scores,
        raw_text=raw_text,
    )

def print_maintenance_report(report: MaintenanceReport) -> None:
    """
    Helper function for pretty-print a MaintenanceReport in a formal report layout.
    """
    width = 62
    print(f"\n{'='*width}")
    print(f"  STRUCTURAL INSPECTION MAINTENANCE REPORT")
    print(f"{'='*width}")
    print(f"  Asset / Image  : {report.image_id}")
    print(f"  Severity       : {report.severity}")
    print(f"  Generated At   : {report.generated_at}")
    print(f"{'='*width}\n")

    sections = [
        ("1. FINDINGS SUMMARY",              report.findings_summary),
        ("2. RISK ASSESSMENT",               report.risk_assessment),
        ("3. RECOMMENDED REPAIR ACTIONS",    report.repair_actions),
        ("4. INSPECTION SCHEDULE / NEXT STEPS", report.inspection_schedule),
    ]

    for heading, content in sections:
        print(f"{'─'*width}")
        print(f"  {heading}")
        print(f"{'─'*width}")
        # Indent each line of the section content for readability.
        for line in (content or "(not generated)").splitlines():
            print(f"  {line}")
        print()

    if report.footnotes:
        print(f"{'─'*width}")
        print(f"  REFERENCES")
        print(f"{'─'*width}")
        for line in report.footnotes.splitlines():
            print(f"  {line}")
        print()

    print(f"{'='*width}")
    

# Test
# payload = InspectionPayload.from_dict({
#     "image_id":                  "bridge_01.jpg",
#     "crack_detected":            True,
#     "crack_area_ratio":          0.054,
#     "estimated_crack_length_px": 438,
#     "num_crack_regions":         3,
#     "severity":                  "High",
#     "model_confidence":          0.91,
# })
# reranker = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
# embedder = OpenAIEmbedder(model="text-embedding-3-small")
# store = ChromaRAGStore(dir="./chroma_db", collection_name="crack_detection")
# chroma_results = query_by_inspection(
#     payload=payload,
#     embedder=embedder,
#     store=store,
#     top_k=DEFAULT_RETRIEVAL_TOP_K,
# )
# generator = AnswerGenerator()

# report = generate_maintenance_report(
#     payload=payload,
#     embedder=embedder,
#     store=store,
#     chroma_results=chroma_results,
#     reranker=reranker,
#     generator=generator,
# )

# print_maintenance_report(report)