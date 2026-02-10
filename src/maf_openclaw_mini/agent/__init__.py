"""
Agent Module

Core agent logic using Microsoft Agent Framework's ChatAgent.

This module provides:
- AgentManager: Lifecycle management for the ChatAgent
- AgentContext: Per-request context (user, channel, thread)
- Prompts: System prompt and user context builder
- Tools: Slack action tools extracted from the monolith
"""

from .context import AgentContext
from .prompts import SYSTEM_PROMPT, build_user_context_instructions
from .core import AgentManager

__all__ = [
    "AgentContext",
    "AgentManager",
    "SYSTEM_PROMPT",
    "build_user_context_instructions",
]
