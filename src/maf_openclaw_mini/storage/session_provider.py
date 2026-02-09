"""
Session Context Provider

Implements MAF's ContextProvider pattern for session management.

This is the official MAF way to:
- Load context before agent invocation (invoking hook)
- Persist state after agent completes (invoked hook)
- Track session lifecycle (thread_created hook)

Reference: agent_framework/_memory.py - ContextProvider
"""

from typing import Optional, MutableSequence, Any
from dataclasses import dataclass

from .database import (
    get_or_create_session,
    get_session,
    add_message,
    Session,
)
from .message_store import (
    SQLiteChatMessageStore,
    ChatMessage,
    build_conversation_context,
)


@dataclass
class Context:
    """
    Per-invocation context for agent.

    Following MAF pattern: Context holds transient data that should NOT
    be stored in conversation history. It's injected before each agent.run()
    and discarded after.

    Attributes:
        instructions: Additional instructions to inject
        messages: Additional context messages (not stored)
        tools: Additional tools for this invocation
    """
    instructions: Optional[str] = None
    messages: Optional[list] = None
    tools: Optional[list] = None


class SessionContextProvider:
    """
    Context provider for session-based conversation management.

    Following MAF ContextProvider pattern:
    1. thread_created() - Initialize session when new thread created
    2. invoking() - Load context before agent invocation
    3. invoked() - Persist state after agent completes

    Usage:
        provider = SessionContextProvider()

        # Before agent invocation
        context = await provider.invoking(
            messages=[],
            user_id="U123",
            channel_id="C456",
        )

        # Inject context into agent
        agent = client.as_agent(
            instructions=base_instructions + context.instructions,
            ...
        )

        # After agent completes
        await provider.invoked(
            request_messages=[user_message],
            response_messages=[assistant_response],
            session_id=session.id,
        )
    """

    def __init__(self, max_history_messages: int = 50):
        """
        Initialize the session context provider.

        Args:
            max_history_messages: Maximum messages to include in context
        """
        self.max_history_messages = max_history_messages
        self._current_session: Optional[Session] = None

    async def thread_created(self, thread_id: Optional[str] = None) -> None:
        """
        Called when a new conversation thread is created.

        Following MAF pattern: Initialize any session state here.

        Args:
            thread_id: Optional thread identifier
        """
        # Session will be created lazily in invoking()
        pass

    async def get_or_create_session(
        self,
        user_id: str,
        channel_id: str,
        thread_ts: Optional[str] = None,
    ) -> Session:
        """
        Get or create a session for the conversation.

        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            thread_ts: Thread timestamp (for threaded conversations)

        Returns:
            Session object
        """
        session = get_or_create_session(
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
        )
        self._current_session = session
        return session

    async def invoking(
        self,
        messages: MutableSequence[Any],
        **kwargs,
    ) -> Context:
        """
        Called before agent invocation to load context.

        Following MAF pattern: Load any additional context required.
        This includes conversation history and session-specific data.

        Args:
            messages: Current messages (can be mutated to add history)
            **kwargs: Additional context (user_id, channel_id, etc.)

        Returns:
            Context object with additional instructions/messages/tools
        """
        user_id = kwargs.get("user_id")
        channel_id = kwargs.get("channel_id")
        thread_ts = kwargs.get("thread_ts")
        session_id = kwargs.get("session_id")

        # Get or create session
        if session_id:
            session = get_session(session_id)
        elif user_id and channel_id:
            session = await self.get_or_create_session(
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
        else:
            # No session context available
            return Context()

        if not session:
            return Context()

        self._current_session = session

        # Load conversation history
        message_store = SQLiteChatMessageStore(
            session_id=session.id,
            max_messages=self.max_history_messages,
        )
        history = await message_store.list_messages()

        # Build context from history
        if history:
            history_context = build_conversation_context(history)
            return Context(instructions=history_context)

        return Context()

    async def invoked(
        self,
        request_messages: list,
        response_messages: list,
        **kwargs,
    ) -> None:
        """
        Called after agent completes to persist state.

        Following MAF pattern: Store conversation and extract memories here.

        Args:
            request_messages: Messages sent to agent (user input)
            response_messages: Messages from agent (assistant response)
            **kwargs: Additional context (session_id, etc.)
        """
        session_id = kwargs.get("session_id")

        if not session_id and self._current_session:
            session_id = self._current_session.id

        if not session_id:
            return

        # Store messages in database
        for msg in request_messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                add_message(
                    session_id=session_id,
                    role=msg.role,
                    content=msg.content,
                )
            elif isinstance(msg, dict):
                add_message(
                    session_id=session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                )

        for msg in response_messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                add_message(
                    session_id=session_id,
                    role=msg.role,
                    content=msg.content,
                )
            elif isinstance(msg, dict):
                add_message(
                    session_id=session_id,
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                )

    def get_current_session(self) -> Optional[Session]:
        """Get the current session if set."""
        return self._current_session


# Singleton instance for global access
_session_provider: Optional[SessionContextProvider] = None


def get_session_provider() -> SessionContextProvider:
    """Get the global session provider instance."""
    global _session_provider
    if _session_provider is None:
        _session_provider = SessionContextProvider()
    return _session_provider


def init_session_provider(max_history_messages: int = 50) -> SessionContextProvider:
    """Initialize and return the global session provider."""
    global _session_provider
    _session_provider = SessionContextProvider(max_history_messages=max_history_messages)
    return _session_provider
