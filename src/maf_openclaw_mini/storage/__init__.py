"""
Storage Module

Following Microsoft Agent Framework patterns for state persistence.

This module provides:
- SQLite database for session and message storage
- ChatMessageStore implementation using MAF's ChatMessage type
- SessionContextProvider extending MAF's ContextProvider ABC
"""

# Database operations
from .database import (
    init_database,
    get_connection,
    # Session management
    get_or_create_session,
    get_session,
    cleanup_old_sessions,
    Session,
    # Message history
    add_message,
    get_session_messages,
    get_recent_messages,
    Message,
    # User management
    is_user_approved,
    approve_user,
    generate_pairing_code,
    approve_pairing,
    # Statistics
    get_stats,
)

# Message store (MAF ChatMessageStoreProtocol)
from .message_store import (
    SQLiteChatMessageStore,
    build_conversation_context,
)

# Context provider (MAF ContextProvider ABC)
from .session_provider import (
    SessionContextProvider,
)

__all__ = [
    # Database
    "init_database",
    "get_connection",
    # Session
    "get_or_create_session",
    "get_session",
    "cleanup_old_sessions",
    "Session",
    # Messages
    "add_message",
    "get_session_messages",
    "get_recent_messages",
    "Message",
    # Users
    "is_user_approved",
    "approve_user",
    "generate_pairing_code",
    "approve_pairing",
    # Stats
    "get_stats",
    # Message store
    "SQLiteChatMessageStore",
    "build_conversation_context",
    # Context provider
    "SessionContextProvider",
]
