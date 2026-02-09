"""
mem0 Memory Client

mem0 is an external memory service that stores facts about users.
This module integrates mem0 with the MAF bot.

Following Microsoft Agent Framework memory best practices:
- Extract memories from conversations
- Inject relevant memories as context before agent calls
- Persist user-specific information

Reference: https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory
mem0 API: https://docs.mem0.ai/api-reference
"""

import os
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

# mem0 API configuration
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
MEM0_BASE_URL = "https://api.mem0.ai/v1"
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"

# State
_initialized = False


def init_memory() -> bool:
    """
    Initialize mem0 connection.

    Returns:
        True if memory is enabled and configured
    """
    global _initialized

    if not MEMORY_ENABLED:
        print("Memory: Disabled by configuration")
        return False

    if not MEM0_API_KEY:
        print("Memory: No API key configured")
        return False

    _initialized = True
    return True


def is_memory_enabled() -> bool:
    """Check if memory is enabled and initialized."""
    return _initialized and MEMORY_ENABLED


async def add_memory(
    messages: list[dict],
    user_id: str,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Store conversation memories in mem0.

    This extracts facts from the conversation and stores them
    associated with the user.

    Args:
        messages: List of message dicts with 'role' and 'content'
        user_id: The Slack user ID
        metadata: Optional metadata to attach

    Returns:
        Response from mem0 API or None on error
    """
    if not is_memory_enabled():
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{MEM0_BASE_URL}/memories/",
                headers={
                    "Authorization": f"Token {MEM0_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": messages,
                    "user_id": user_id,
                    "metadata": metadata or {},
                },
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Memory: Failed to add - {response.status_code}")
                return None

    except Exception as e:
        print(f"Memory: Error adding - {e}")
        return None


async def search_memory(
    query: str,
    user_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search memories for a user.

    Args:
        query: The search query
        user_id: The Slack user ID
        limit: Maximum number of results

    Returns:
        List of relevant memories
    """
    if not is_memory_enabled():
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{MEM0_BASE_URL}/memories/search/",
                headers={
                    "Authorization": f"Token {MEM0_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "user_id": user_id,
                    "limit": limit,
                },
            )

            if response.status_code == 200:
                data = response.json()
                # Handle both list and dict responses
                if isinstance(data, list):
                    return data
                return data.get("results", data.get("memories", []))
            else:
                print(f"Memory: Search failed - {response.status_code}")
                return []

    except Exception as e:
        print(f"Memory: Error searching - {e}")
        return []


async def get_all_memories(user_id: str) -> list[dict]:
    """
    Get all memories for a user.

    Args:
        user_id: The Slack user ID

    Returns:
        List of all memories
    """
    if not is_memory_enabled():
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{MEM0_BASE_URL}/memories/",
                headers={
                    "Authorization": f"Token {MEM0_API_KEY}",
                },
                params={
                    "user_id": user_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                # Handle both list and dict responses
                if isinstance(data, list):
                    return data
                return data.get("results", data.get("memories", []))
            else:
                return []

    except Exception as e:
        print(f"Memory: Error getting all - {e}")
        return []


async def delete_memory(memory_id: str) -> bool:
    """
    Delete a specific memory.

    Args:
        memory_id: The memory ID to delete

    Returns:
        True if deleted successfully
    """
    if not is_memory_enabled():
        return False

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{MEM0_BASE_URL}/memories/{memory_id}/",
                headers={
                    "Authorization": f"Token {MEM0_API_KEY}",
                },
            )
            return response.status_code == 200

    except Exception as e:
        print(f"Memory: Error deleting - {e}")
        return False


def build_memory_context(memories: list[dict]) -> str:
    """
    Build context string from memories.

    Following MAF best practice: Format memories as additional
    context to inject before agent invocation.

    Args:
        memories: List of memory objects from mem0

    Returns:
        Formatted context string
    """
    if not memories:
        return ""

    # Extract memory text from each result
    facts = []
    for memory in memories:
        # mem0 returns memories in 'memory' field
        text = memory.get("memory", "")
        if text:
            facts.append(f"- {text}")

    if not facts:
        return ""

    # Format as context for the agent
    return (
        "## What I Remember About This User\n"
        "Use this information to personalize your responses:\n"
        + "\n".join(facts)
    )
