"""
Document ingestion module for the Smart Street Lighting RAG pipeline.

Reads knowledge base documents, chunks them, embeds them using local
LM Studio (nomic embeddings), and stores in a ChromaDB vector store.
"""

from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from rag.lm_studio import LMStudioEmbedding

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
DOWNLOADED_SOURCES_DIR = PROJECT_ROOT / "data" / "downloaded_sources"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"


def create_chroma_client():
    """Create a persistent ChromaDB client."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def ingest_knowledge_base(
    collection_name: str = "street_lighting_kb",
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> VectorStoreIndex:
    """
    Ingest all documents from the knowledge base directory into ChromaDB.
    Uses local LM Studio for embeddings.
    """
    # Load from both curated knowledge base and downloaded sources
    documents = []

    for source_dir in [KNOWLEDGE_BASE_DIR, DOWNLOADED_SOURCES_DIR]:
        if source_dir.exists():
            print(f"Loading documents from {source_dir}...")
            reader = SimpleDirectoryReader(
                input_dir=str(source_dir),
                recursive=True,
                required_exts=[".md", ".txt", ".pdf"],
            )
            docs = reader.load_data()
            documents.extend(docs)
            print(f"  Loaded {len(docs)} files from {source_dir.name}")

    print(f"Total documents loaded: {len(documents)}")

    # Add source metadata based on filename keywords
    source_rules = [
        ("as_nzs_1158", "standard", "AS/NZS 1158"),
        ("pedestrian_crossing", "standard", "AS/NZS 1158.4"),
        ("energy", "guideline", "Energy Efficiency Guidelines"),
        ("solar", "guideline", "Solar Lighting Guidelines"),
        ("adaptive_dimming", "guideline", "Adaptive Dimming Guidelines"),
        ("methodology", "guideline", "Lighting Design Methodology"),
        ("event", "guideline", "Event Lighting Guidelines"),
        ("melbourne", "urban_data", "Melbourne Urban Data"),
        ("nga_factors", "guideline", "National Greenhouse Accounts"),
        ("bom_", "urban_data", "Bureau of Meteorology"),
    ]

    for doc in documents:
        filename = doc.metadata.get("file_name", "unknown").lower()
        doc.metadata["source_type"] = "other"
        doc.metadata["source"] = doc.metadata.get("file_name", "unknown")
        for keyword, src_type, src_name in source_rules:
            if keyword in filename:
                doc.metadata["source_type"] = src_type
                doc.metadata["source"] = src_name
                break

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embed_model = LMStudioEmbedding()

    # Set up ChromaDB
    chroma_client = create_chroma_client()
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking and embedding documents (using local LM Studio)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
        show_progress=True,
    )

    num_chunks = chroma_collection.count()
    print(f"Ingestion complete. {num_chunks} chunks stored in ChromaDB.")
    return index


def load_existing_index(
    collection_name: str = "street_lighting_kb",
) -> VectorStoreIndex:
    """Load an existing ChromaDB index (no re-ingestion)."""
    chroma_client = create_chroma_client()
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = LMStudioEmbedding()

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    print(f"Loaded existing index with {chroma_collection.count()} chunks.")
    return index


if __name__ == "__main__":
    index = ingest_knowledge_base()
    print("Done. Index is ready for querying.")
