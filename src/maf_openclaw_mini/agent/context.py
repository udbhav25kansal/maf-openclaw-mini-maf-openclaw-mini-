"""
Agent Context

Per-request context passed through the agent system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentContext:
    """Per-request context for agent invocations."""

    user_id: str
    channel_id: str
    thread_ts: Optional[str] = None
    session_id: Optional[str] = None
