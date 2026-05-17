"""
Rebuild the RAG knowledge base index.

Re-embeds all documents from data/knowledge_base/ and data/downloaded_sources/
into ChromaDB. Run this after adding, removing, or editing knowledge base files.

Usage:
    python ingest.py
"""

from rag.ingest import ingest_knowledge_base

if __name__ == "__main__":
    ingest_knowledge_base()
