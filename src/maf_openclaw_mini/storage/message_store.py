"""
SQLite Message Store

Implements MAF's ChatMessageStoreProtocol for SQLite-based conversation storage.
All sync database calls are wrapped with asyncio.to_thread() to avoid blocking.

Reference: agent_framework/_threads.py - ChatMessageStoreProtocol
"""

import asyncio
from typing import Any, MutableMapping, Sequence

from agent_framework import ChatMessage, Role

from .database import (
    get_session_messages,
    add_message as db_add_message,
    Message,
)


class SQLiteChatMessageStore:
    """
    SQLite-based chat message store following MAF's ChatMessageStoreProtocol.

    Stores and retrieves conversation history as MAF ChatMessage instances.
    """

    def __init__(self, session_id: str, max_messages: int = 50):
        self.session_id = session_id
        self.max_messages = max_messages

    async def list_messages(self) -> list[ChatMessage]:
        """Get all messages for the session as MAF ChatMessage instances."""
        db_messages = await asyncio.to_thread(
            get_session_messages,
            session_id=self.session_id,
            limit=self.max_messages,
        )
        return [
            ChatMessage(role=Role(msg.role), text=msg.content)
            for msg in db_messages
        ]

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Persist MAF ChatMessage instances to SQLite."""
        for msg in messages:
            await asyncio.to_thread(
                db_add_message,
                session_id=self.session_id,
                role=msg.role.value,
                content=msg.text or "",
            )

    async def serialize(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the store state for persistence."""
        return {
            "session_id": self.session_id,
            "max_messages": self.max_messages,
        }

    @classmethod
    async def deserialize(
        cls,
        serialized_store_state: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> "SQLiteChatMessageStore":
        """Restore a message store from serialized state."""
        return cls(
            session_id=serialized_store_state.get("session_id", ""),
            max_messages=serialized_store_state.get("max_messages", 50),
        )

    async def update_from_state(
        self, serialized_store_state: MutableMapping[str, Any], **kwargs: Any
    ) -> None:
        """Update the current instance from serialized state data."""
        self.session_id = serialized_store_state.get("session_id", self.session_id)
        self.max_messages = serialized_store_state.get("max_messages", self.max_messages)


def build_conversation_context(messages: list[ChatMessage]) -> str:
    """
    Build a conversation context string from MAF ChatMessage instances.

    Used for injecting conversation history into agent instructions.
    """
    if not messages:
        return ""

    lines = ["## Recent Conversation History"]
    for msg in messages[-10:]:
        role_label = "User" if msg.role == Role.USER else "Assistant"
        text = msg.text or ""
        lines.append(f"{role_label}: {text[:200]}...")

    return "\n".join(lines)
