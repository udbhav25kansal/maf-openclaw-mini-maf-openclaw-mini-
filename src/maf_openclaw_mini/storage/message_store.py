"""
SQLite Message Store

Implements MAF's ChatMessageStoreProtocol for SQLite-based conversation storage.

Reference: agent_framework/_threads.py - ChatMessageStoreProtocol
"""

from typing import Any, Sequence
from dataclasses import dataclass

from .database import (
    get_session_messages,
    add_message as db_add_message,
    Message,
)


@dataclass
class ChatMessage:
    """
    Chat message compatible with MAF's message format.

    This is a simplified version of MAF's ChatMessage for our use case.
    """
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0),
        )

    @classmethod
    def from_db_message(cls, msg: Message) -> "ChatMessage":
        """Create from database Message object."""
        return cls(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
        )


class SQLiteChatMessageStore:
    """
    SQLite-based chat message store following MAF's ChatMessageStoreProtocol.

    This class manages conversation history for a specific session,
    storing messages in SQLite for persistence across restarts.

    Usage:
        store = SQLiteChatMessageStore(session_id="abc123")
        messages = await store.list_messages()
        await store.add_messages([ChatMessage(role="user", content="Hello")])
    """

    def __init__(self, session_id: str, max_messages: int = 50):
        """
        Initialize the message store.

        Args:
            session_id: The session ID to store messages for
            max_messages: Maximum messages to retrieve (for context window)
        """
        self.session_id = session_id
        self.max_messages = max_messages

    async def list_messages(self) -> list[ChatMessage]:
        """
        Get all messages for the session.

        Following MAF pattern: Returns messages for next agent invocation.
        Messages are limited to max_messages to fit context window.

        Returns:
            List of ChatMessage objects (oldest first)
        """
        db_messages = get_session_messages(
            session_id=self.session_id,
            limit=self.max_messages,
        )

        return [ChatMessage.from_db_message(msg) for msg in db_messages]

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """
        Add messages to the store.

        Following MAF pattern: Called after agent invocation to persist history.

        Args:
            messages: List of ChatMessage objects to add
        """
        for msg in messages:
            db_add_message(
                session_id=self.session_id,
                role=msg.role,
                content=msg.content,
            )

    async def serialize(self, **kwargs) -> dict[str, Any]:
        """
        Serialize the store state for persistence.

        Following MAF pattern: Enables saving/restoring conversation state.

        Returns:
            Dictionary with serialized state
        """
        messages = await self.list_messages()
        return {
            "session_id": self.session_id,
            "max_messages": self.max_messages,
            "messages": [msg.to_dict() for msg in messages],
        }

    @classmethod
    async def deserialize(
        cls,
        serialized_store_state: dict,
        **kwargs,
    ) -> "SQLiteChatMessageStore":
        """
        Restore a message store from serialized state.

        Following MAF pattern: Restores conversation state from persistence.

        Args:
            serialized_store_state: Dictionary from serialize()

        Returns:
            Restored SQLiteChatMessageStore instance
        """
        return cls(
            session_id=serialized_store_state.get("session_id", ""),
            max_messages=serialized_store_state.get("max_messages", 50),
        )

    def get_message_count(self) -> int:
        """Get the number of messages in this session."""
        messages = get_session_messages(
            session_id=self.session_id,
            limit=10000,  # Get all
        )
        return len(messages)


def build_conversation_context(messages: list[ChatMessage]) -> str:
    """
    Build a conversation context string from messages.

    Useful for injecting conversation history into agent instructions.

    Args:
        messages: List of ChatMessage objects

    Returns:
        Formatted conversation history string
    """
    if not messages:
        return ""

    lines = ["## Recent Conversation History"]
    for msg in messages[-10:]:  # Last 10 messages
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {msg.content[:200]}...")

    return "\n".join(lines)
