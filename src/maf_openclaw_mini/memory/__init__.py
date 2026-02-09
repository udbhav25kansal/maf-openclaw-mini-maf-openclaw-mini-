"""
Memory Module

Following Microsoft Agent Framework best practices for memory:
https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory

This module provides:
- mem0 API client for long-term memory storage
- Memory context builder for agent prompts
"""

from .mem0_client import (
    init_memory,
    is_memory_enabled,
    add_memory,
    search_memory,
    get_all_memories,
    delete_memory,
    build_memory_context,
)

__all__ = [
    "init_memory",
    "is_memory_enabled",
    "add_memory",
    "search_memory",
    "get_all_memories",
    "delete_memory",
    "build_memory_context",
]
