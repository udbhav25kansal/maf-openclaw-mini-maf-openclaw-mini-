"""
RAG Search Tool

Following Microsoft Agent Framework RAG best practices:
https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-rag

Key patterns implemented:
1. Search tool as @tool decorated function (agent decides when to use)
2. Source citation with channel name and timestamp
3. Relevance scoring for transparency
4. Optional channel filtering
"""

from typing import Annotated, Optional
from pydantic import Field
from agent_framework import tool

from .embeddings import create_embedding
from .vectorstore import search_documents, get_document_count


def format_search_result(doc: dict) -> str:
    """
    Format a search result with source citation.

    Following MAF best practice: Always include source information for traceability.
    """
    metadata = doc.get("metadata", {})
    text = doc.get("text", "")
    score = doc.get("score", 0)

    # Build source citation
    channel = metadata.get("channel_name", "unknown")
    user = metadata.get("user_name", "unknown")
    timestamp = metadata.get("timestamp", "")

    # Format: [#channel | @user | time] message (relevance: X%)
    source = f"[#{channel} | @{user}"
    if timestamp:
        source += f" | {timestamp}"
    source += "]"

    relevance = f"({score * 100:.0f}% relevant)"

    return f"{source} {text} {relevance}"


@tool(
    name="search_slack_history",
    description="Search through past Slack messages to find relevant information. Use this when users ask about past discussions, decisions, or what someone said."
)
async def search_slack_history(
    query: Annotated[str, Field(description="What to search for in message history")],
    channel: Annotated[Optional[str], Field(description="Optional: limit search to a specific channel name (without #)")] = None,
    limit: Annotated[int, Field(description="Maximum number of results to return")] = 5,
) -> str:
    """
    Search through indexed Slack message history using semantic search.

    This tool uses vector embeddings to find messages that are semantically
    similar to the query, not just keyword matches.
    """
    # Check if we have any documents indexed
    doc_count = await get_document_count()
    if doc_count == 0:
        return "No messages have been indexed yet. The knowledge base is empty."

    # Create embedding for the search query
    query_embedding = await create_embedding(query)

    # Search for similar documents
    results = await search_documents(
        query_embedding=query_embedding,
        limit=limit,
        channel_filter=channel,
    )

    if not results:
        channel_note = f" in #{channel}" if channel else ""
        return f"No relevant messages found for '{query}'{channel_note}."

    # Format results with source citations (MAF best practice)
    formatted_results = []
    for i, doc in enumerate(results, 1):
        formatted = format_search_result(doc)
        formatted_results.append(f"{i}. {formatted}")

    header = f"Found {len(results)} relevant messages"
    if channel:
        header += f" in #{channel}"
    header += ":"

    return f"{header}\n\n" + "\n\n".join(formatted_results)
