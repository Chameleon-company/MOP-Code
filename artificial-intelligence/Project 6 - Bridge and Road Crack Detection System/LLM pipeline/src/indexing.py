"""
This module sits between `chunk.py` and `answering.py`. It owns 3 things:
1. Embedding - wraps OpenAI's embedding API with retries and batching.
2. Storage - wraps ChromaDB for persistent, deduplicated chunk storage.
3. Retrieval
   - `query_by_inspection()` - structured query built from a JSON inspection payload. Converts the payload into a semantic search string. Use for engineering maintenance report generation.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import APIError, OpenAI, RateLimitError

from .chunking import Chunk, chunk_markdown_document

# OpenAI embedding
DEFAULT_BATCH_SIZE = 2048

# Retry / back-off settings for API failures.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# Number of candidates to fetch from ChromaDB before handing off to the reranker in answering.py
DEFAULT_RETRIEVAL_TOP_K = 20

# Valid severity levels for inspection payloads.
VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}

@dataclass
class InspectionPayload:
    """
    Structured results from the crack metrics JSON format.
    This is the input to query_by_inspection().
    Can be constructed directly in Python or loaded from a JSON file via from_dict() or from_file().
    
    Args:
        image_id : Filename or identifier of the inspected image.
        crack_detected : Whether any crack was detected.
        crack_area_ratio : Cracked area as a fraction of total image area.
        estimated_crack_length_px: Estimated total crack length in pixels.
        num_crack_regions: Number of distinct crack regions detected.
        severity: Qualitative severity label: Low / Medium / High / Critical.
        model_confidence: Detection model's confidence score in [0, 1].
    
    """
    image_id: str
    crack_detected: bool
    crack_area_ratio: float
    estimated_crack_length_px: int
    num_crack_regions: int
    severity: str
    model_confidence: float
    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InspectionPayload":
        """
        Build an InspectionPayload from a plain Python dict.
        """
        required = [
            "image_id", "crack_detected", "crack_area_ratio",
            "estimated_crack_length_px", "num_crack_regions",
            "severity", "model_confidence",
        ]
        missing = []
        for i in required:
            if i not in data:
                missing.append(i)
        
        if missing:
            raise ValueError(f"InspectionPayload is missing required field(s): {missing}")

        severity = data["severity"]
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'. "
                f"Must be one of: {sorted(VALID_SEVERITIES)}"
            )

        return cls(
            image_id=str(data["image_id"]),
            crack_detected=bool(data["crack_detected"]),
            crack_area_ratio=float(data["crack_area_ratio"]),
            estimated_crack_length_px=int(data["estimated_crack_length_px"]),
            num_crack_regions=int(data["num_crack_regions"]),
            severity=severity,
            model_confidence=float(data["model_confidence"]),
        )
        
    @classmethod
    def from_file(cls, json_path: str) -> "InspectionPayload":
        """
        Load an InspectionPayload from a JSON file on disk.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Inspection JSON file not found: '{json_path}'")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
    
    def to_semantic_query(self) -> str:
        """
        Convert this JSON payload into a natural-language semantic search string.
        Reasons to convert instead of querying raw numbers is that vector search
        operates on sematic meanings, not numeric values. By converting to a descriptive sentence,
        we improve the retrieval recall.
        """
        severity_lower = self.severity.lower()
        detection_status = "detected" if self.crack_detected else "not detected"

        query = (
            f"{self.severity} severity crack {detection_status} on bridge structure. "
            f"Crack area ratio {self.crack_area_ratio:.4f}, "
            f"estimated length {self.estimated_crack_length_px}px "
            f"across {self.num_crack_regions} region(s). "
            f"Model confidence {self.model_confidence:.2f}. "
            f"Recommended repair actions and inspection schedule "
            f"for {severity_lower} severity concrete cracking."
        )

        return query
    

class OpenAIEmbedder:
    """
    Wrapper around OpenAI's embedding endpoint with retry logic.
    
    Args:
        model: OpenAI embedding model id.
                "text-embedding-3-small" - fast and cheap
                "text-embedding-3-large" - higher quality, higher cost.
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. ")
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings and return their vector representations.
        Each element of texts maps to one vector in the returned list in the same order. 
        """
        if not texts:
            return []

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=texts)
                embeddings = []
                for item in response.data:
                    embeddings.append(item.embedding)
                return embeddings

            except RateLimitError:
                wait = RETRY_BACKOFF_BASE ** attempt
                print(f"Rate limit hit. Attempt {attempt}.")
                time.sleep(wait)

            except APIError as exc:
                wait = RETRY_BACKOFF_BASE ** attempt
                print(f"API error. Attempt {attempt}.")
                time.sleep(wait)

        raise RuntimeError(
            f"embed_texts failed after {MAX_RETRIES} attempts. "
            "Check API key, quota, and network connection."
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single string. Convenience wrapper around embed_texts."""
        return self.embed_texts([text])[0]

def batch_embed_texts(
    embedder: OpenAIEmbedder,
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[List[float]]:
    """
    Embed a large list of strings in fixed-size batches.
    Batching prevents payload-size errors and makes each request fast enough
    to retry individually.

    Args:
        embedder (OpenAIEmbedder): OpenAIEmbedder
        texts (List[str]): Sematic query
        batch_size (int, optional): Batch size. Defaults to DEFAULT_BATCH_SIZE.

    Returns:
        List[List[float]]: flat list of embeddings in the same order as texts
    """
    if not texts:
        print("batch_embed_texts called with an empty list.")
        return []

    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        # print(f"Embedding batch {batch_num}.")
        all_embeddings.extend(embedder.embed_texts(batch))

    print("Finished embedding")
    return all_embeddings


class ChromaRAGStore:
    """
    Vector store in ChromaDB. 
    Each Chunk is stored with its text, embedding, and full metadata.
    
    Args:
        dir: Directory where ChromaDB writes its files.
        collection_name : Name of the collection. Multiple collections can coexist in the same directory.
    """
    
    def __init__(
        self,
        dir: str = "./chroma_db",
        collection_name: str = "markdown_rag",
    ):
        self.client = chromadb.PersistentClient(path=dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Insert chunks and embeddings, skipping any that already exist.
        Deduplication uses the chunk_id (SHA-256 hash) from chunker.py
        """
        if not chunks:
            print("add_chunks called with an empty list.")
            return 0
        
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )
        
        # Deduplicate within this batch first (handle duplicate IDs in the same ingestion).
        seen_ids = set()
        unique_chunks = []
        unique_embeddings = []
        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
                unique_embeddings.append(emb)
            else:
                print(f"Skipping duplicate chunk ID within batch: {chunk.chunk_id}")

        if not unique_chunks:
            print("All chunks in batch are duplicates — nothing to add.")
            return 0
        
        # Check which ids already exist in the store.
        all_ids = []
        for unique_chunk in unique_chunks:
            all_ids.append(unique_chunk.chunk_id)
        
        existing_ids = set(self.collection.get(ids=all_ids, include=[])["ids"])

        new_chunks = [unique_chunk for unique_chunk in unique_chunks if unique_chunk.chunk_id not in existing_ids]
        
        new_embeddings = []
        for unique_chunk, unique_embedding in zip(unique_chunks, unique_embeddings):
            if unique_chunk.chunk_id not in existing_ids:
                new_embeddings.append(unique_embedding)

        skipped = len(unique_chunks) - len(new_chunks)
        if skipped:
            print(f"Skipping {skipped} chunk(s) already in store.")

        if not new_chunks:
            print("All chunks already in store — nothing to add.")
            return 0

        self.collection.add(
            ids=[new_chunk.chunk_id for new_chunk in new_chunks],
            documents=[new_chunk.chunk_text for new_chunk in new_chunks],
            metadatas=[
                {
                    "doc_title": new_chunk.doc_title,
                    "source_file": new_chunk.source_file,
                    "section_path": new_chunk.section_path,
                    "prev_heading": new_chunk.prev_heading or "",
                    "chunk_type": new_chunk.chunk_type,
                    "embedding_text": new_chunk.embedding_text,
                    "token_count": new_chunk.token_count,
                }
                for new_chunk in new_chunks
            ],
            embeddings=new_embeddings,
        )

        print(f"Added {len(new_chunks)} new chunk(s). Collection total: {self.collection.count()}",)
        return len(new_chunks)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> dict:
        """
        Return the top_k most similar chunks to query_embedding.

        Args:
            query_embedding (List[float]): Query vector from OpenAIEmbedder.embed_query().
            top_k (int): Number of results to return. Defaults to  DEFAULT_RETRIEVAL_TOP_K
        
        Returns
            ChromaDB result dict with keys: ids, documents, metadatas, distances.
            Access like: results["documents"][0], results["metadatas"][0], etc.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        results = self.collection.query(**kwargs)
        count = len(results.get("ids", [[]])[0])
        print(f"Search returned {count} result(s).")
        return results
    
    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()  

def ingest_markdown_file(
    markdown_path: str,
    store: ChromaRAGStore,
    embedder: OpenAIEmbedder,
    max_tokens: int = 4000,
    overlap_tokens: int = 40,
) -> List[Chunk]:
    """
    Full ingestion pipeline for a single markdown file.
    Reads the file → chunks it → embeds each chunk → stores in ChromaDB.

    Args:
        markdown_path (str): Path to markdown file
        store (ChromaRAGStore): ChromaRAGStore
        embedder (OpenAIEmbedder): OpenAIEmbedder
        max_tokens (int, optional): Max token per chunk. Defaults to 4000.
        overlap_tokens (int, optional): Overlapping tokens between chunks if overflow. Defaults to 40.

    Returns:
        List[Chunk]: Full list of Chunk objects produced
    """
    
    path = os.path.abspath(markdown_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Markdown file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    
    if not markdown_text.strip():
        print(f"File {path} is empty. Nothing to ingest.")
        return []
    
    chunks = chunk_markdown_document(
        markdown_text=markdown_text,
        source_file=markdown_path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    
    if not chunks:
        print(f"No chunks produced from {path}.")
        return []
    
    print(f"Produced {len(chunks)} chunk(s) from {path}.")
    
    # Embed the enriched embedding_text (includes section/type prefix)
    embeddings = batch_embed_texts(
        embedder, [chunk.embedding_text for chunk in chunks]
    )

    added = store.add_chunks(chunks, embeddings)
    print(f"Ingestion done: {added} new chunk(s) stored.")
    return chunks

def query_by_inspection(
    payload: InspectionPayload,
    embedder: OpenAIEmbedder,
    store: ChromaRAGStore,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> dict:
    """
    Retrieve maintenance knowledge base chunks relevant to an inspection result.
    The raw JSON payload is converted into a descriptive engineering
    sentence via InspectionPayload.to_semantic_query(). This produces a
    query in the same language register as the maintenance knowledge base,
    which improves semantic similarity matching.
    """
    if not payload.crack_detected:
        # If no crack was found, there's nothing to retrieve repair guidance for.
        # Return an empty result rather than inputting irrelevant passages.
        print(f"No crack detected in payload {payload.image_id} — returning empty result.")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    # Convert payload into a semantic query string
    semantic_query = payload.to_semantic_query()
    
    # Embed the semantic query and search
    query_embedding = embedder.embed_query(semantic_query)
    results = store.search(
        query_embedding=query_embedding,
        top_k=top_k
    )
    return results

def print_results(results: dict) -> None:
    """
    Pretty-print ChromaDB search results. Only use in Development/debugging helper.
    """
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No results found.")
        return

    print(f"{len(documents)} result(s) returned")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        print(f"\n--- Result {i} ---")
        print(f"  Distance  : {dist:.4f}  (lower = more similar)")
        print(f"  Source    : {meta.get('source_file', '?')}")
        print(f"  Section   : {meta.get('section_path', '?')}")
        print(f"  Type      : {meta.get('chunk_type', '?')}")
        print(f"  Preview   : {doc[:120]!r}")
        

# Test
# payload = InspectionPayload.from_dict({
#     "image_id": "bridge_01.jpg",
#     "crack_detected": True,
#     "crack_area_ratio": 0.054,
#     "estimated_crack_length_px": 438,
#     "num_crack_regions": 3,
#     "severity": "High",
#     "model_confidence": 0.91,
# })
# embedder = OpenAIEmbedder(model="text-embedding-3-small")
# store = ChromaRAGStore(dir="./chroma_db", collection_name="crack_detection")

# results = query_by_inspection(
#     payload=payload,
#     embedder=embedder,
#     store=store,
#     top_k=5
# )

# print_results(results)