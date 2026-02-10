"""
Session Context Provider

Extends MAF's ContextProvider ABC for session-based conversation management.
All sync database calls are wrapped with asyncio.to_thread() to avoid blocking.

Reference: agent_framework/_memory.py - ContextProvider
"""

import asyncio
from typing import Optional, MutableSequence, Sequence, Any

from agent_framework import ContextProvider, Context, ChatMessage

from .database import (
    get_or_create_session,
    get_session,
    add_message,
    Session,
)
from .message_store import (
    SQLiteChatMessageStore,
    build_conversation_context,
)


class SessionContextProvider(ContextProvider):
    """
    Context provider that manages session lifecycle and conversation history.

    Extends the real MAF ContextProvider ABC:
    - invoking(): Load conversation history before agent invocation
    - invoked(): Persist messages after agent completes

    Session creation is owned by this provider — callers just pass
    user_id/channel_id/thread_ts as kwargs to agent.run().
    """

    def __init__(self, max_history_messages: int = 50):
        self.max_history_messages = max_history_messages
        self._current_session: Optional[Session] = None

    async def thread_created(self, thread_id: Optional[str] = None) -> None:
        """Called when a new conversation thread is created."""
        pass

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        """
        Load session context before agent invocation.

        kwargs (user_id, channel_id, session_id, thread_ts) flow from agent.run().
        """
        user_id = kwargs.get("user_id")
        channel_id = kwargs.get("channel_id")
        thread_ts = kwargs.get("thread_ts")
        session_id = kwargs.get("session_id")

        # Get or create session (non-blocking)
        if session_id:
            session = await asyncio.to_thread(get_session, session_id)
        elif user_id and channel_id:
            session = await asyncio.to_thread(
                get_or_create_session,
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
        else:
            return Context()

        if not session:
            return Context()

        self._current_session = session

        # Load conversation history (already async via message_store)
        message_store = SQLiteChatMessageStore(
            session_id=session.id,
            max_messages=self.max_history_messages,
        )
        history = await message_store.list_messages()

        if history:
            history_context = build_conversation_context(history)
            return Context(instructions=history_context)

        return Context()

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist messages to SQLite after agent completes."""
        session_id = kwargs.get("session_id")

        if not session_id and self._current_session:
            session_id = self._current_session.id

        if not session_id:
            return

        # Normalize to sequences
        if isinstance(request_messages, ChatMessage):
            request_messages = [request_messages]
        if isinstance(response_messages, ChatMessage):
            response_messages = [response_messages]

        for msg in request_messages:
            await asyncio.to_thread(
                add_message,
                session_id=session_id,
                role=msg.role.value,
                content=msg.text or "",
            )

        if response_messages:
            for msg in response_messages:
                await asyncio.to_thread(
                    add_message,
                    session_id=session_id,
                    role=msg.role.value,
                    content=msg.text or "",
                )

    def get_current_session(self) -> Optional[Session]:
        """Get the current session if set."""
        return self._current_session
