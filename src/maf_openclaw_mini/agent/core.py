"""
Agent Core

AgentManager — lifecycle wrapper around MAF's ChatAgent.

Creates a single ChatAgent at startup with:
- OpenAIChatClient
- MCPStdioTools (auto-discovered)
- CompositeContextProvider (session + mem0)
- All @tool functions

The ContextProvider handles per-request session/memory loading automatically.
"""

import os
from contextlib import AsyncExitStack
from typing import Optional

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from .context import AgentContext
from .prompts import SYSTEM_PROMPT
from .tools import (
    send_slack_message,
    list_slack_channels,
    list_slack_users,
    get_channel_info,
    get_current_time,
    calculate,
)

from ..mcp import build_mcp_tools
from ..memory import CompositeContextProvider
from ..storage.session_provider import SessionContextProvider
from ..rag.search_tool import search_slack_history
from ..scheduler import set_reminder, list_reminders, cancel_reminder
from ..tools import web_search, fetch_url


class AgentManager:
    """
    Manages the ChatAgent lifecycle.

    - initialize(): Build client, tools, context provider, enter async context
    - run(): Send a message through the agent
    - shutdown(): Clean up resources
    """

    def __init__(self):
        self._agent: Optional[ChatAgent] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._context_provider: Optional[CompositeContextProvider] = None

    async def initialize(self) -> None:
        """Create and start the ChatAgent with all tools and providers."""
        self._exit_stack = AsyncExitStack()

        # LLM client
        client = OpenAIChatClient(
            model_id=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Context provider (session history + mem0 memory)
        session_provider = SessionContextProvider()
        self._context_provider = CompositeContextProvider(
            session_provider=session_provider,
        )

        # Collect tools
        tools: list = [
            # Slack actions
            send_slack_message,
            list_slack_channels,
            list_slack_users,
            get_channel_info,
            # Utilities
            get_current_time,
            calculate,
            # RAG
            search_slack_history,
            # Reminders
            set_reminder,
            list_reminders,
            cancel_reminder,
            # Web
            web_search,
            fetch_url,
        ]

        # MCP tools (MCPStdioTool instances)
        mcp_tools = build_mcp_tools()
        if mcp_tools:
            tools.extend(mcp_tools)
            print(f"[Agent] Added {len(mcp_tools)} MCP tool sources")

        # OpenTelemetry (Step 7)
        _setup_observability()

        # Create the ChatAgent
        self._agent = ChatAgent(
            chat_client=client,
            instructions=SYSTEM_PROMPT,
            name="SlackAssistant",
            tools=tools,
            context_provider=self._context_provider,
        )

        # Enter async context — connects MCP servers, etc.
        await self._exit_stack.enter_async_context(self._agent)
        print("[Agent] ChatAgent initialized")

    async def run(self, message: str, context: AgentContext) -> str:
        """
        Run the agent with user message and context.

        kwargs flow to ContextProvider.invoking() automatically, where
        CompositeContextProvider injects user context, session history,
        and mem0 memories as Context.instructions.
        """
        if not self._agent:
            raise RuntimeError("AgentManager not initialized. Call initialize() first.")

        result = await self._agent.run(
            message,
            user_id=context.user_id,
            channel_id=context.channel_id,
            thread_ts=context.thread_ts,
            session_id=context.session_id,
        )
        return result.text

    async def shutdown(self) -> None:
        """Close the async exit stack (disconnects MCP, cleans up)."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            print("[Agent] Shutdown complete")

    @property
    def is_mem0_enabled(self) -> bool:
        if self._context_provider:
            return self._context_provider.is_mem0_enabled
        return False


def _setup_observability() -> None:
    """Enable OpenTelemetry instrumentation if configured."""
    try:
        from agent_framework.observability import (
            enable_instrumentation,
            configure_otel_providers,
        )

        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            # Set service name via env var (standard OTel convention)
            os.environ.setdefault("OTEL_SERVICE_NAME", "maf-openclaw-mini")
            configure_otel_providers()
            print("[OTel] Configured with OTLP exporter")
        else:
            enable_instrumentation()
    except Exception as e:
        print(f"[OTel] Instrumentation setup skipped: {e}")
