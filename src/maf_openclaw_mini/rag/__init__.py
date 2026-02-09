"""
RAG (Retrieval Augmented Generation) Module

Following Microsoft Agent Framework best practices:
https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-rag

This module provides:
- Vector store for storing message embeddings (ChromaDB)
- Embedding generation (OpenAI text-embedding-3-small)
- Search tool for the agent to use
"""

from .vectorstore import (
    init_vectorstore,
    add_documents,
    search_documents,
    get_document_count,
)

from .embeddings import create_embedding, create_embeddings

from .search_tool import search_slack_history

__all__ = [
    "init_vectorstore",
    "add_documents",
    "search_documents",
    "get_document_count",
    "create_embedding",
    "create_embeddings",
    "search_slack_history",
]
