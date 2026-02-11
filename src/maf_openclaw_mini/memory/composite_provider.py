"""
Composite Context Provider

Merges session history (SessionContextProvider) and user memory (Mem0Provider)
into a single ContextProvider for the agent.

Also injects per-request user context (user_id, channel_id) as instructions
so tools like set_reminder receive the correct values.
"""

import os
from typing import Any, MutableSequence, Sequence, Optional

from agent_framework import ContextProvider, Context, ChatMessage

from ..agent.prompts import build_user_context_instructions
from ..storage.session_provider import SessionContextProvider

# Mem0Provider is optional — only used when configured
_mem0_available = False
try:
    from agent_framework_mem0 import Mem0Provider

    _mem0_available = True
except ImportError:
    pass


class CompositeContextProvider(ContextProvider):
    """
    Combines SessionContextProvider (conversation history) and
    Mem0Provider (long-term user memory) into one ContextProvider.

    Also injects user context instructions so the LLM knows the
    current user_id/channel_id for tool calls.

    Mem0Provider instances are lazily created and cached per user_id.
    """

    def __init__(
        self,
        session_provider: SessionContextProvider,
        mem0_api_key: Optional[str] = None,
    ):
        self._session = session_provider
        self._mem0_api_key = mem0_api_key or os.getenv("MEM0_API_KEY", "")
        self._mem0_enabled = (
            _mem0_available
            and bool(self._mem0_api_key)
            and os.getenv("MEMORY_ENABLED", "true").lower() == "true"
        )
        self._mem0_cache: dict[str, "Mem0Provider"] = {}

    def _get_mem0_provider(self, user_id: str) -> Optional["Mem0Provider"]:
        """Lazily create and cache a Mem0Provider for the given user_id."""
        if not self._mem0_enabled:
            return None
        if user_id not in self._mem0_cache:
            self._mem0_cache[user_id] = Mem0Provider(
                api_key=self._mem0_api_key,
                user_id=user_id,
            )
        return self._mem0_cache[user_id]

    @property
    def is_mem0_enabled(self) -> bool:
        return self._mem0_enabled

    async def thread_created(self, thread_id: Optional[str] = None) -> None:
        await self._session.thread_created(thread_id)

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        """
        Merge context from user info, session history, and mem0 memory.

        All three are returned as Context.instructions so the LLM sees them
        alongside the base SYSTEM_PROMPT without polluting the user message.
        """
        # Get session context (conversation history)
        session_ctx = await self._session.invoking(messages, **kwargs)

        # Build user context instructions from kwargs
        user_id = kwargs.get("user_id")
        channel_id = kwargs.get("channel_id")
        user_ctx = ""
        if user_id and channel_id:
            user_ctx = build_user_context_instructions(user_id, channel_id)

        # Get mem0 context (user memories)
        # Mem0Provider returns memories as Context.messages (not .instructions),
        # so we extract the text from the messages to inject as instructions.
        mem0_instructions = ""
        if user_id:
            mem0_provider = self._get_mem0_provider(user_id)
            if mem0_provider:
                try:
                    mem0_ctx = await mem0_provider.invoking(messages, **kwargs)
                    # Mem0Provider puts memories in messages as ChatMessage objects
                    if mem0_ctx.messages:
                        mem0_parts = []
                        for msg in mem0_ctx.messages:
                            if msg.text and msg.text.strip():
                                mem0_parts.append(msg.text)
                        mem0_instructions = "\n".join(mem0_parts)
                    elif mem0_ctx.instructions:
                        mem0_instructions = mem0_ctx.instructions
                except Exception as e:
                    print(f"[Memory] Mem0 invoking error: {e}")

        # Merge all instruction parts
        parts = []
        if user_ctx:
            parts.append(user_ctx)
        if session_ctx.instructions:
            parts.append(session_ctx.instructions)
        if mem0_instructions:
            parts.append(mem0_instructions)

        merged_instructions = "\n\n".join(parts) if parts else None

        return Context(
            instructions=merged_instructions,
            messages=session_ctx.messages,
            tools=session_ctx.tools,
        )

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Delegate to both session (persist to SQLite) and mem0 (store memories)."""
        # Persist to session/SQLite
        await self._session.invoked(
            request_messages, response_messages, invoke_exception, **kwargs
        )

        # Store in mem0
        user_id = kwargs.get("user_id")
        if user_id:
            mem0_provider = self._get_mem0_provider(user_id)
            if mem0_provider:
                try:
                    await mem0_provider.invoked(
                        request_messages, response_messages, invoke_exception, **kwargs
                    )
                except Exception as e:
                    print(f"[Memory] Mem0 invoked error: {e}")
