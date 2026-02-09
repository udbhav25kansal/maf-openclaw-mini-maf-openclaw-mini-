"""
Storage Module

Following Microsoft Agent Framework patterns for state persistence:
- https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory

This module provides:
- SQLite database for session and message storage
- ChatMessageStore implementation for conversation history
- ContextProvider for session lifecycle management

Usage:
------
```python
from maf_openclaw_mini.storage import (
    init_database,
    get_or_create_session,
    add_message,
    get_session_messages,
    SessionContextProvider,
    SQLiteChatMessageStore,
)

# Initialize database at startup
init_database()

# Get or create session
session = get_or_create_session(user_id="U123", channel_id="C456")

# Add messages
add_message(session.id, "user", "Hello!")
add_message(session.id, "assistant", "Hi there!")

# Get history
messages = get_session_messages(session.id)

# Use context provider for MAF integration
provider = SessionContextProvider()
context = await provider.invoking([], user_id="U123", channel_id="C456")
# ... agent invocation ...
await provider.invoked(request_messages, response_messages, session_id=session.id)
```
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
    ChatMessage,
    build_conversation_context,
)

# Context provider (MAF ContextProvider pattern)
from .session_provider import (
    SessionContextProvider,
    Context,
    get_session_provider,
    init_session_provider,
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
    "ChatMessage",
    "build_conversation_context",
    # Context provider
    "SessionContextProvider",
    "Context",
    "get_session_provider",
    "init_session_provider",
]
