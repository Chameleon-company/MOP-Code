"""
Query engine module for the Smart Street Lighting RAG pipeline.

Uses local LM Studio for LLM reasoning and
wraps LlamaIndex's query engine with a street lighting system prompt.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

from rag.lm_studio import LMStudioLLM


SYSTEM_PROMPT = """\
You are a professional lighting design engineer writing a technical report \
for the City of Melbourne.

You have been provided with:
1. Calculated design specifications (number of lights, spacing, energy costs) \
computed from AS/NZS 1158 standards — these numbers are CORRECT, use them.
2. Retrieved context from the knowledge base with additional standards detail.

You MUST structure your report with these exact sections:
1. **Executive Summary** — 3-4 sentences summarising the design recommendation
2. **Site Analysis** — location description, traffic data, safety context
3. **Design Specification** — present the key values in a markdown table: \
P-Category, Number of Lights, Average Spacing, Technology, Luminaire Wattage, \
Annual Energy Cost
4. **Placement Rationale** — explain why lights are positioned where they are
5. **Energy & Cost Analysis** — including dimming savings and HPS comparison
6. **Standards Compliance** — cite specific AS/NZS 1158 clauses from the provided context
7. **Recommendations** — maintenance schedule, future considerations

CRITICAL RULES:
- Use the EXACT numbers provided in the calculation context. Do not round, estimate, or approximate.
- When citing standards, reference the specific clause (e.g., "AS/NZS 1158.3.1 Table 2.1")
- If a safety adjustment was applied, explain it in the Site Analysis section
- Keep the report under 1500 words
"""


def create_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 5,
    response_mode: str = "compact",
) -> RetrieverQueryEngine:
    """
    Create a query engine using local LM Studio LLM.
    """
    llm = LMStudioLLM()
    print(f"Using LLM: LM Studio / {llm.model_name}")

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=response_mode,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # Inject system prompt into the QA template
    qa_prompt = PromptTemplate(
        SYSTEM_PROMPT + "\n\n"
        "Context information from the knowledge base:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Query: {query_str}\n\n"
        "Provide a detailed, professional design report response:"
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt}
    )

    return query_engine


def query_with_context(
    query_engine: RetrieverQueryEngine,
    user_query: str,
    calculation_context: str = "",
):
    """
    Run a query with optional pre-computed calculation context prepended.
    """
    if calculation_context:
        full_query = (
            f"CALCULATED DESIGN SPECIFICATIONS (these are correct, use them):\n"
            f"{calculation_context}\n\n"
            f"USER QUERY: {user_query}\n\n"
            f"Using the calculated specs above and the retrieved knowledge base context, "
            f"write a professional design report that explains and justifies this lighting design."
        )
    else:
        full_query = user_query

    response = query_engine.query(full_query)
    return response
