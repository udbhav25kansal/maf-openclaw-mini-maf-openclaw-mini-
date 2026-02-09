"""
RAG (Retrieval Augmented Generation) Module

Following Microsoft Agent Framework best practices:
https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-rag

This module provides:
- Vector store for storing message embeddings (ChromaDB)
- Embedding generation (OpenAI text-embedding-3-small)
- Search tool for the agent to use
- Background indexer for periodic updates
"""

from .vectorstore import (
    init_vectorstore,
    add_documents,
    search_documents,
    get_document_count,
)

from .embeddings import create_embedding, create_embeddings

from .search_tool import search_slack_history

from .indexer import (
    index_all_channels,
    run_indexer,
    BackgroundIndexer,
    get_background_indexer,
    start_background_indexer,
    stop_background_indexer,
)

__all__ = [
    # Vector store
    "init_vectorstore",
    "add_documents",
    "search_documents",
    "get_document_count",
    # Embeddings
    "create_embedding",
    "create_embeddings",
    # Search tool
    "search_slack_history",
    # Indexer
    "index_all_channels",
    "run_indexer",
    "BackgroundIndexer",
    "get_background_indexer",
    "start_background_indexer",
    "stop_background_indexer",
]
