"""
Vector Store Module

Uses ChromaDB for storing and searching message embeddings.
ChromaDB is a lightweight, local vector database that's perfect for this use case.

Following Microsoft's RAG best practices:
- Hybrid search (keyword + semantic) when possible
- Source tracking for citations
- Efficient batching for indexing

Reference: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
"""

import os
from typing import Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Configuration
VECTOR_DB_PATH = os.getenv("RAG_VECTOR_DB_PATH", "./data/chroma")
COLLECTION_NAME = "slack_messages"

# Global ChromaDB client and collection
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def init_vectorstore() -> chromadb.Collection:
    """
    Initialize the ChromaDB vector store.

    Returns:
        The ChromaDB collection for slack messages
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    # Create data directory if it doesn't exist
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    # Initialize ChromaDB with persistence
    _client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=Settings(
            anonymized_telemetry=False,  # Disable telemetry
        ),
    )

    # Get or create collection
    # Using cosine similarity (best for text embeddings)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return _collection


def get_collection() -> chromadb.Collection:
    """Get the initialized collection, initializing if needed."""
    if _collection is None:
        return init_vectorstore()
    return _collection


async def add_documents(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """
    Add documents to the vector store.

    Args:
        ids: Unique IDs for each document
        texts: The document texts
        embeddings: Pre-computed embedding vectors
        metadatas: Metadata for each document (channel, user, timestamp, etc.)
    """
    collection = get_collection()

    # ChromaDB handles batching internally
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


async def search_documents(
    query_embedding: list[float],
    limit: int = 10,
    channel_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search for similar documents.

    Args:
        query_embedding: The embedding vector of the search query
        limit: Maximum number of results
        channel_filter: Optional channel name to filter by

    Returns:
        List of matching documents with scores
    """
    collection = get_collection()

    # Build filter if channel specified
    where_filter = None
    if channel_filter:
        where_filter = {"channel_name": channel_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    documents = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            # Convert distance to similarity score (cosine distance -> similarity)
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1 - distance  # Cosine similarity = 1 - cosine distance

            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": score,
            })

    return documents


async def get_document_count() -> int:
    """Get the total number of documents in the vector store."""
    collection = get_collection()
    return collection.count()


async def document_exists(doc_id: str) -> bool:
    """Check if a document with the given ID exists."""
    collection = get_collection()
    result = collection.get(ids=[doc_id])
    return len(result["ids"]) > 0


async def delete_documents(ids: list[str]) -> None:
    """Delete documents by their IDs."""
    collection = get_collection()
    collection.delete(ids=ids)


async def clear_all() -> None:
    """Clear all documents from the vector store. Use with caution!"""
    global _collection, _client

    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        init_vectorstore()
