# MAF-Openclaw-Mini Architecture

This document maps the TypeScript Openclaw-mini architecture to Microsoft Agent Framework (MAF) Python implementation.

> **Validated Against**: Official Microsoft Agent Framework documentation from [Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/) and [GitHub](https://github.com/microsoft/agent-framework) as of February 2026.

---

## Executive Summary

The original Openclaw-mini is a Slack AI assistant built with TypeScript that uses:
- OpenAI SDK for LLM interactions with tool calling
- Slack Bolt for real-time Slack communication
- ChromaDB for vector storage (RAG)
- mem0 for long-term memory
- SQLite for session/message storage
- MCP (Model Context Protocol) for external integrations

**MAF-Openclaw-Mini** will replicate this functionality using Python with Microsoft Agent Framework, which combines:
- **Semantic Kernel** - For AI orchestration and tool calling
- **AutoGen** - For multi-agent patterns (if needed)
- Built-in observability via **OpenTelemetry**
- **Native MCP support** via `HostedMCPTool`
- **Built-in hosted tools** (web search, file search, code interpreter)

---

## Why Microsoft Agent Framework?

### Evidence-Based Justification

1. **Official Microsoft Support**: MAF is the official successor to both Semantic Kernel and AutoGen, built by the same team. Per [Visual Studio Magazine](https://visualstudiomagazine.com/articles/2025/10/01/semantic-kernel-autogen--open-source-microsoft-agent-framework.aspx): "Think of Microsoft Agent Framework as Semantic Kernel v2.0"

2. **Production-Ready**: According to [Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview), MAF provides "everything from simple chat agents to complex multi-agent workflows with graph-based orchestration"

3. **Built-in MCP Support**: MAF has native `HostedMCPTool` support, eliminating the need for custom MCP client implementation

4. **Active Development**: Latest version 1.0.0b260130 released January 30, 2026 on [PyPI](https://pypi.org/project/agent-framework/)

5. **Multi-Provider Support**: Works with OpenAI, Azure OpenAI, and Azure AI out of the box

---

## Component Mapping: TypeScript → Python/MAF

| TypeScript (Openclaw-mini) | Python (MAF-Openclaw-mini) | Source |
|---------------------------|---------------------------|--------|
| `OpenAI` SDK | `OpenAIChatClient` or `AzureOpenAIChatClient` | [MS Learn: Quick Start](https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start) |
| Tool definitions (JSON schema) | `@tool` decorator with `Annotated[type, Field()]` | [MS Learn: Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools) |
| `processMessage()` loop | `agent.run()` or `agent.run_stream()` | [MS Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |
| Slack Bolt (`@slack/bolt`) | `slack-bolt` (Python port) | Same library, Python version |
| ChromaDB | ChromaDB OR `HostedFileSearchTool` | [MS Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |
| mem0 REST API | mem0 REST API (same) | External service |
| SQLite via better-sqlite3 | SQLite via `sqlite3` or `aiosqlite` | Standard library |
| Custom MCP Client | **`HostedMCPTool`** (built-in!) | [MS Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |
| Web Fetcher tool | **`HostedWebSearchTool`** (built-in!) | [MS Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |

---

## Directory Structure

```
maf-openclaw-mini/
├── pyproject.toml           # Project configuration
├── .env                     # Environment variables
├── .env.example             # Template
├── .python-version          # Python 3.12.3
├── README.md
│
├── src/
│   └── maf_openclaw_mini/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       ├── config.py            # Configuration loader
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── core.py          # Main agent with MAF
│       │   ├── prompts.py       # System prompts
│       │   └── context.py       # AgentContext dataclass
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── slack_tools.py   # Slack action tools
│       │   ├── rag_tools.py     # RAG search tools
│       │   ├── web_fetcher.py   # URL content fetcher
│       │   └── scheduler.py     # Task scheduler
│       │
│       ├── channels/
│       │   ├── __init__.py
│       │   └── slack.py         # Slack Bolt handler
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── embeddings.py    # OpenAI embeddings
│       │   ├── vectorstore.py   # ChromaDB operations
│       │   ├── indexer.py       # Background indexing
│       │   └── retriever.py     # Semantic search
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── database.py      # SQLite operations
│       │   └── mem0_client.py   # mem0 API client
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── client.py        # MCP client
│       │   ├── config.py        # MCP server config
│       │   └── converter.py     # Tool format converter
│       │
│       └── utils/
│           ├── __init__.py
│           └── logger.py        # Logging setup
│
├── data/                        # Runtime data (gitignored)
│   ├── assistant.db
│   └── chroma/
│
├── tests/
│   └── ...
│
└── docs/
    ├── explanation.md
    └── architecture.md          # This file
```

---

## Core Classes and Patterns

### 1. Agent Core (`agent/core.py`)

The heart of the system. Uses MAF's agent pattern with tools.

> **Source**: [Microsoft Learn: Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools)

```python
import asyncio
from typing import Annotated
from pydantic import Field
from agent_framework import ChatAgent, tool, HostedMCPTool, HostedWebSearchTool
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential
import os

# Option 1: OpenAI directly
client = OpenAIChatClient(
    model=os.getenv("DEFAULT_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Option 2: Azure OpenAI (recommended for production)
# async with AzureCliCredential() as credential:
#     client = AzureOpenAIChatClient(
#         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         credential=credential,
#     )

# Create agent with tools (using context manager for proper resource management)
# Source: https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start
async def create_agent():
    return client.as_agent(
        name="SlackAssistant",
        instructions=SYSTEM_PROMPT,
        tools=[
            # Custom function tools
            search_knowledge_base,
            send_message,
            get_channel_history,
            schedule_message,
            set_reminder,
            list_channels,
            list_users,
            # Built-in hosted tools (no custom implementation needed!)
            HostedWebSearchTool(),  # Replaces our web_fetcher.py
            HostedMCPTool(          # Replaces our custom MCP client!
                name="GitHub MCP",
                url="https://api.github.com/mcp",  # Example
            ),
        ],
    )

# Process a message
# Source: https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools
async def process_message(user_message: str, context: AgentContext) -> AgentResponse:
    agent = await create_agent()

    # Use agent.run() - NOT invoke()
    # The agent handles the tool calling loop automatically
    result = await agent.run(user_message)

    # Or for streaming responses:
    # async for update in agent.run_stream(user_message):
    #     if update.text:
    #         yield update.text

    return AgentResponse(
        content=result.text,
        should_thread=context.thread_ts is not None,
        rag_used=False,
        sources_count=0,
    )
```

### 1b. Built-in Hosted Tools (Major Simplification!)

> **Source**: [Microsoft Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools)

MAF provides built-in "hosted tools" that eliminate the need for custom implementations:

```python
from agent_framework import (
    HostedWebSearchTool,      # Built-in web search
    HostedMCPTool,            # Native MCP server integration
    HostedFileSearchTool,     # Vector search over documents
    HostedCodeInterpreterTool # Code execution sandbox
)

# Web Search - replaces our custom web_fetcher.py!
web_search = HostedWebSearchTool(
    additional_properties={
        "user_location": {"city": "Mumbai", "country": "IN"}
    }
)

# MCP Integration - replaces our entire mcp/ module!
github_mcp = HostedMCPTool(
    name="GitHub MCP",
    url=os.getenv("GITHUB_MCP_URL"),
)

notion_mcp = HostedMCPTool(
    name="Notion MCP",
    url=os.getenv("NOTION_MCP_URL"),
)

# File Search - could replace our ChromaDB RAG!
file_search = HostedFileSearchTool(
    inputs=[HostedVectorStoreContent(vector_store_id="vs_slack_messages")],
    max_results=10,
)

# Agent with all hosted tools
agent = client.as_agent(
    name="SlackAssistant",
    instructions=SYSTEM_PROMPT,
    tools=[
        # Custom tools
        send_message,
        schedule_message,
        set_reminder,
        list_channels,
        list_users,
        # Hosted tools (no implementation needed!)
        web_search,
        github_mcp,
        notion_mcp,
        file_search,
    ],
)
```

**Why this matters**: We can potentially eliminate:
- `src/mcp/` module entirely → use `HostedMCPTool`
- `src/tools/web_fetcher.py` → use `HostedWebSearchTool`
- `src/rag/` module (optional) → use `HostedFileSearchTool`

### 2. Tool Definitions (`tools/slack_tools.py`)

MAF uses the `@tool` decorator with Pydantic-style `Annotated` types for parameters.

```python
from agent_framework import tool
from typing import Annotated
from pydantic import Field

@tool
async def search_knowledge_base(
    query: Annotated[str, Field(description="The search query - what to look for in message history")],
    channel_name: Annotated[str | None, Field(description="Optional: limit search to a specific channel")] = None,
    limit: Annotated[int, Field(description="Number of results (default 10)")] = 10,
) -> str:
    """Search through indexed Slack message history using semantic search."""
    from ..rag import retrieve

    results = await retrieve(query, limit=limit, channel_name=channel_name)

    if not results:
        return f'No relevant messages found for "{query}"'

    formatted = "\n".join(
        f"{i+1}. {r.formatted} (relevance: {r.score*100:.0f}%)"
        for i, r in enumerate(results)
    )
    return f"Found {len(results)} relevant messages:\n\n{formatted}"


@tool
async def send_message(
    target: Annotated[str, Field(description='Channel name (e.g., "general") or user name')],
    message: Annotated[str, Field(description="The message to send")],
) -> str:
    """Send a message to a Slack user or channel."""
    from ..channels.slack import web_client, find_channel, find_user

    # Resolve target to ID
    if target.startswith("#"):
        channel = await find_channel(target[1:])
        channel_id = channel.id if channel else None
    else:
        user = await find_user(target)
        channel_id = user.id if user else None

    if not channel_id:
        return f"Could not find target: {target}"

    result = await web_client.chat_postMessage(channel=channel_id, text=message)
    return f"Message sent to {target}" if result["ok"] else f"Failed: {result.get('error')}"


@tool
async def get_channel_history(
    channel_name: Annotated[str, Field(description="Channel name without # prefix")],
    limit: Annotated[int, Field(description="Number of messages (default 20)")] = 20,
) -> str:
    """Get recent messages from a Slack channel."""
    from ..channels.slack import web_client, find_channel

    channel = await find_channel(channel_name)
    if not channel:
        return f"Channel not found: {channel_name}"

    result = await web_client.conversations_history(channel=channel.id, limit=limit)
    messages = result.get("messages", [])

    if not messages:
        return f"No messages found in #{channel_name}"

    formatted = format_messages_for_context(messages)
    return f"Recent messages from #{channel_name}:\n\n{formatted}"


@tool
async def schedule_message(
    target: Annotated[str, Field(description="Channel or user name")],
    message: Annotated[str, Field(description="Message to send")],
    send_at: Annotated[str, Field(description='ISO 8601 timestamp, e.g., "2026-01-28T10:30:00+05:30"')],
) -> str:
    """Schedule a message to be sent later."""
    from datetime import datetime
    from ..channels.slack import web_client, find_channel

    try:
        scheduled_time = datetime.fromisoformat(send_at)
    except ValueError:
        return f"Invalid date format: {send_at}"

    channel = await find_channel(target)
    if not channel:
        return f"Could not find target: {target}"

    result = await web_client.chat_scheduleMessage(
        channel=channel.id,
        text=message,
        post_at=int(scheduled_time.timestamp()),
    )

    return f"Message scheduled for {scheduled_time}" if result["ok"] else f"Failed: {result.get('error')}"


@tool
async def set_reminder(
    text: Annotated[str, Field(description="Reminder text")],
    time: Annotated[str, Field(description='When to remind, e.g., "in 5 minutes", "tomorrow at 9am"')],
) -> str:
    """Set a reminder for the user."""
    # Implementation similar to TypeScript version
    # Parse natural language time, fall back to scheduled DM if no user token
    ...


@tool
async def list_channels() -> str:
    """List all accessible Slack channels."""
    from ..channels.slack import web_client

    result = await web_client.conversations_list(types="public_channel,private_channel")
    channels = result.get("channels", [])
    member_channels = [c for c in channels if c.get("is_member")]

    return f"Channels I'm in ({len(member_channels)}):\n" + \
           "\n".join(f"- #{c['name']}" for c in member_channels)


@tool
async def list_users() -> str:
    """List all users in the workspace."""
    from ..channels.slack import web_client

    result = await web_client.users_list()
    users = [u for u in result.get("members", []) if not u.get("is_bot") and not u.get("deleted")]

    user_list = "\n".join(f"- {u.get('real_name', 'Unknown')} (@{u['name']})" for u in users[:20])
    return f"Users ({len(users)}):\n{user_list}{'...' if len(users) > 20 else ''}"


@tool
async def fetch_url_content(
    url: Annotated[str, Field(description="The URL to fetch (must be http:// or https://)")],
    extract_type: Annotated[str, Field(description='What to extract: "text" or "metadata"')] = "text",
    max_length: Annotated[int, Field(description="Maximum characters to return (default 6000)")] = 6000,
) -> str:
    """Fetch and extract content from a URL."""
    import httpx
    from bs4 import BeautifulSoup

    # Security: Block internal URLs
    if is_internal_url(url):
        return "Error: Cannot fetch internal/private URLs"

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = soup.title.string if soup.title else None

    # Extract main content
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)[:max_length]

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if extract_type != "metadata":
        parts.append(f"\nContent:\n{text}")

    return "\n".join(parts) or "No content extracted from URL."
```

### 3. Slack Channel Handler (`channels/slack.py`)

Uses Python's `slack-bolt` library (official Python port).

```python
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient
import os

from ..config import config
from ..agent.core import process_message
from ..agent.context import AgentContext
from ..memory.database import get_or_create_session, is_user_approved, generate_pairing_code

# Initialize Slack app
app = AsyncApp(
    token=config.slack.bot_token,
    signing_secret=config.slack.signing_secret,
)

web_client = AsyncWebClient(token=config.slack.bot_token)
bot_user_id: str | None = None


async def get_bot_user_id() -> str:
    global bot_user_id
    if bot_user_id:
        return bot_user_id
    auth = await web_client.auth_test()
    bot_user_id = auth["user_id"]
    return bot_user_id


def is_bot_mentioned(text: str, bot_id: str) -> bool:
    return f"<@{bot_id}>" in text


def remove_bot_mention(text: str, bot_id: str) -> str:
    import re
    return re.sub(rf"<@{bot_id}>\s*", "", text).strip()


def is_direct_message(channel_id: str) -> bool:
    return channel_id.startswith("D")


@app.message()
async def handle_message(message, say, client):
    # Skip bot messages
    if message.get("subtype"):
        return

    text = message.get("text", "")
    user = message.get("user")
    channel = message.get("channel")
    ts = message.get("ts")
    thread_ts = message.get("thread_ts")

    if not text or not user:
        return

    current_bot_id = await get_bot_user_id()
    if user == current_bot_id:
        return

    is_dm = is_direct_message(channel)

    # Only respond to mentions in channels
    if not is_dm and not is_bot_mentioned(text, current_bot_id):
        return

    # DM pairing security
    if is_dm and config.security.dm_policy == "pairing" and not is_user_approved(user):
        code = generate_pairing_code(user)
        await say(
            text=f"To use this bot in DMs, please get approved by an admin.\n\n"
                 f"Your pairing code: `{code}`\n\n"
                 f"Ask an admin to approve you in a channel with: `approve {code}`"
        )
        return

    clean_text = text if is_dm else remove_bot_mention(text, current_bot_id)

    # Add "eyes" reaction
    if config.features.reactions:
        await client.reactions_add(channel=channel, timestamp=ts, name="eyes")

    try:
        session = get_or_create_session(user, channel, thread_ts)

        context = AgentContext(
            session_id=session.id,
            user_id=user,
            channel_id=channel,
            thread_ts=thread_ts,
        )

        response = await process_message(clean_text, context)

        # Remove "eyes", send response
        if config.features.reactions:
            await client.reactions_remove(channel=channel, timestamp=ts, name="eyes")

        await say(
            text=response.content,
            thread_ts=thread_ts or ts if response.should_thread else None,
        )
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        if config.features.reactions:
            await client.reactions_add(channel=channel, timestamp=ts, name="warning")
        await say(text="Sorry, I encountered an error. Please try again.", thread_ts=thread_ts or ts)


@app.event("app_mention")
async def handle_app_mention(event, say, client):
    # Similar to handle_message, for channel mentions
    ...


async def start_slack_app():
    handler = AsyncSocketModeHandler(app, config.slack.app_token)
    await handler.start_async()


async def stop_slack_app():
    # Graceful shutdown
    ...
```

### 4. Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class SlackConfig:
    bot_token: str = field(default_factory=lambda: os.getenv("SLACK_BOT_TOKEN", ""))
    app_token: str = field(default_factory=lambda: os.getenv("SLACK_APP_TOKEN", ""))
    user_token: str | None = field(default_factory=lambda: os.getenv("SLACK_USER_TOKEN"))
    signing_secret: str = field(default_factory=lambda: os.getenv("SLACK_SIGNING_SECRET", ""))


@dataclass
class AIConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o"))

    # Azure (optional)
    azure_endpoint: str | None = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT"))
    azure_api_key: str | None = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY"))
    azure_deployment: str | None = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT"))


@dataclass
class MemoryConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() == "true")
    mem0_api_key: str = field(default_factory=lambda: os.getenv("MEM0_API_KEY", ""))


@dataclass
class RAGConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("RAG_ENABLED", "true").lower() == "true")
    embedding_model: str = field(default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    vector_db_path: str = field(default_factory=lambda: os.getenv("RAG_VECTOR_DB_PATH", "./data/chroma"))
    index_interval_hours: int = field(default_factory=lambda: int(os.getenv("RAG_INDEX_INTERVAL_HOURS", "1")))
    max_results: int = field(default_factory=lambda: int(os.getenv("RAG_MAX_RESULTS", "10")))
    min_similarity: float = field(default_factory=lambda: float(os.getenv("RAG_MIN_SIMILARITY", "0.5")))


@dataclass
class SecurityConfig:
    dm_policy: str = field(default_factory=lambda: os.getenv("DM_POLICY", "pairing"))
    allowed_users: str = field(default_factory=lambda: os.getenv("ALLOWED_USERS", "*"))
    allowed_channels: str = field(default_factory=lambda: os.getenv("ALLOWED_CHANNELS", "*"))


@dataclass
class FeaturesConfig:
    thread_summary: bool = field(default_factory=lambda: os.getenv("ENABLE_THREAD_SUMMARY", "true").lower() == "true")
    task_scheduler: bool = field(default_factory=lambda: os.getenv("ENABLE_TASK_SCHEDULER", "true").lower() == "true")
    reactions: bool = field(default_factory=lambda: os.getenv("ENABLE_REACTIONS", "true").lower() == "true")
    typing_indicator: bool = field(default_factory=lambda: os.getenv("ENABLE_TYPING_INDICATOR", "true").lower() == "true")


@dataclass
class AppConfig:
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))
    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "./data/assistant.db"))
    max_history_messages: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "50")))
    session_timeout_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")))


@dataclass
class Config:
    slack: SlackConfig = field(default_factory=SlackConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    app: AppConfig = field(default_factory=AppConfig)


config = Config()
```

### 5. RAG System (`rag/`)

#### Embeddings (`rag/embeddings.py`)

```python
from openai import AsyncOpenAI
from ..config import config

client = AsyncOpenAI(api_key=config.ai.openai_api_key)


async def create_embedding(text: str) -> list[float]:
    """Create embedding for a single text."""
    response = await client.embeddings.create(
        model=config.rag.embedding_model,
        input=preprocess_text(text),
    )
    return response.data[0].embedding


async def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Create embeddings for multiple texts."""
    processed = [preprocess_text(t) for t in texts]
    response = await client.embeddings.create(
        model=config.rag.embedding_model,
        input=processed,
    )
    return [d.embedding for d in response.data]


def preprocess_text(text: str) -> str:
    """Clean text for embedding."""
    import re
    # Remove Slack formatting
    text = re.sub(r"<@[A-Z0-9]+>", "", text)  # Remove mentions
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)  # Channel links
    text = re.sub(r"<https?://[^|>]+\|([^>]+)>", r"\1", text)  # URL links
    text = re.sub(r"<https?://[^>]+>", "", text)  # Plain URLs
    return text.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

#### Vector Store (`rag/vectorstore.py`)

```python
import chromadb
from chromadb.config import Settings
from ..config import config

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path=config.rag.vector_db_path,
    settings=Settings(anonymized_telemetry=False),
)

collection = chroma_client.get_or_create_collection(
    name="slack_messages",
    metadata={"hnsw:space": "cosine"},
)


async def add_documents(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
):
    """Add documents to vector store."""
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


async def search(
    query_embedding: list[float],
    limit: int = 10,
    where: dict | None = None,
) -> list[dict]:
    """Search for similar documents."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # Convert distance to similarity
        })

    return documents


async def get_document_count() -> int:
    """Get total document count."""
    return collection.count()
```

### 6. Memory Module (`memory/`)

#### Database (`memory/database.py`)

```python
import sqlite3
from contextlib import contextmanager
from ..config import config

# Initialize database
def init_database():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                thread_ts TEXT,
                created_at INTEGER NOT NULL,
                last_active INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS approved_users (
                user_id TEXT PRIMARY KEY,
                approved_by TEXT,
                approved_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pairing_codes (
                code TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)


@contextmanager
def get_connection():
    conn = sqlite3.connect(config.app.database_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_or_create_session(user_id: str, channel_id: str, thread_ts: str | None) -> dict:
    """Get or create a session for the user."""
    import time
    import uuid

    now = int(time.time())
    session_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"

    with get_connection() as conn:
        # Try to find existing session
        row = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND channel_id = ? AND thread_ts IS ?",
            (user_id, channel_id, thread_ts),
        ).fetchone()

        if row:
            # Update last_active
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (now, row["id"]),
            )
            return dict(row)

        # Create new session
        session_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (id, user_id, channel_id, thread_ts, created_at, last_active) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_id, channel_id, thread_ts, now, now),
        )

        return {
            "id": session_id,
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "created_at": now,
            "last_active": now,
        }


def add_message(session_id: str, role: str, content: str):
    """Add a message to session history."""
    import time

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, int(time.time())),
        )


def get_session_history(session_id: str, limit: int = 50) -> list[dict]:
    """Get message history for a session."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

        return [dict(row) for row in reversed(rows)]


def is_user_approved(user_id: str) -> bool:
    """Check if user is approved for DMs."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM approved_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row is not None


def generate_pairing_code(user_id: str) -> str:
    """Generate a pairing code for DM approval."""
    import time
    import secrets

    code = secrets.token_hex(3).upper()  # 6-char hex code
    now = int(time.time())
    expires = now + 3600  # 1 hour expiry

    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pairing_codes (code, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (code, user_id, now, expires),
        )

    return code


def approve_pairing(code: str, approved_by: str) -> bool:
    """Approve a pairing code."""
    import time

    now = int(time.time())

    with get_connection() as conn:
        row = conn.execute(
            "SELECT user_id FROM pairing_codes WHERE code = ? AND expires_at > ?",
            (code.upper(), now),
        ).fetchone()

        if not row:
            return False

        user_id = row["user_id"]

        # Add to approved users
        conn.execute(
            "INSERT OR REPLACE INTO approved_users (user_id, approved_by, approved_at) VALUES (?, ?, ?)",
            (user_id, approved_by, now),
        )

        # Delete the pairing code
        conn.execute("DELETE FROM pairing_codes WHERE code = ?", (code.upper(),))

        return True
```

#### mem0 Client (`memory/mem0_client.py`)

```python
import httpx
from ..config import config

MEM0_BASE_URL = "https://api.mem0.ai/v1"

_initialized = False


async def initialize_memory():
    """Initialize mem0 connection."""
    global _initialized
    if not config.memory.mem0_api_key:
        return
    _initialized = True


def is_memory_enabled() -> bool:
    return _initialized and config.memory.enabled


async def add_memory(messages: list[dict], user_id: str):
    """Store conversation in mem0."""
    if not is_memory_enabled():
        return

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{MEM0_BASE_URL}/memories/",
            headers={"Authorization": f"Token {config.memory.mem0_api_key}"},
            json={
                "messages": messages,
                "user_id": user_id,
            },
        )


async def search_memory(query: str, user_id: str, limit: int = 5) -> list[dict]:
    """Search memories for a user."""
    if not is_memory_enabled():
        return []

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MEM0_BASE_URL}/memories/search/",
            headers={"Authorization": f"Token {config.memory.mem0_api_key}"},
            json={
                "query": query,
                "user_id": user_id,
                "limit": limit,
            },
        )

        data = response.json()
        return data.get("results", [])


def build_memory_context(memories: list[dict]) -> str:
    """Build context string from memories."""
    if not memories:
        return ""

    facts = [m.get("memory", "") for m in memories if m.get("memory")]

    if not facts:
        return ""

    return (
        "## What I Know About This User\n"
        + "\n".join(f"- {fact}" for fact in facts)
    )
```

### 7. Main Entry Point (`main.py`)

```python
import asyncio
import signal
import logging

from .config import config
from .utils.logger import setup_logging
from .memory.database import init_database
from .memory.mem0_client import initialize_memory, is_memory_enabled
from .rag.vectorstore import get_document_count
from .rag.indexer import start_indexer, stop_indexer
from .mcp import initialize_mcp, shutdown_mcp, is_mcp_enabled
from .tools.scheduler import task_scheduler
from .channels.slack import start_slack_app, stop_slack_app

logger = logging.getLogger(__name__)


async def main():
    setup_logging(config.app.log_level)

    logger.info("=" * 50)
    logger.info("Starting Slack AI Assistant (MAF)")
    logger.info("=" * 50)

    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()

        # Initialize mem0 memory
        if config.memory.enabled:
            logger.info("Initializing mem0 memory...")
            await initialize_memory()
            logger.info(f"Memory: {'Enabled' if is_memory_enabled() else 'Disabled'}")

        # Initialize RAG
        if config.rag.enabled:
            logger.info("Initializing RAG system...")
            count = await get_document_count()
            logger.info(f"Vector store initialized ({count} documents)")
            start_indexer()
            logger.info("Background indexer started")

        # Initialize MCP
        logger.info("Initializing MCP servers...")
        await initialize_mcp()
        if is_mcp_enabled():
            logger.info("MCP enabled")

        # Start task scheduler
        logger.info("Starting task scheduler...")
        task_scheduler.start()

        # Start Slack app
        logger.info("Starting Slack app...")
        await start_slack_app()

        logger.info("=" * 50)
        logger.info("Slack AI Assistant is running!")
        logger.info("=" * 50)

        # Keep running
        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Shutdown requested...")
    finally:
        await shutdown()


async def shutdown():
    logger.info("Shutting down...")

    await stop_slack_app()
    task_scheduler.stop()
    await shutdown_mcp()

    if config.rag.enabled:
        stop_indexer()

    logger.info("Shutdown complete")


def run():
    """Entry point for the application."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()


if __name__ == "__main__":
    run()
```

---

## Key MAF Patterns (Validated Against Official Docs)

### 1. Tool Definition Pattern

> **Source**: [Microsoft Learn: Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools)

```python
from agent_framework import tool
from typing import Annotated
from pydantic import Field

# Basic pattern - docstring becomes description
def get_weather(
    location: Annotated[str, Field(description="The location to get weather for.")],
) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is sunny."

# With explicit @tool decorator for custom name/description
@tool(name="weather_tool", description="Retrieves weather information")
async def get_weather_v2(
    location: Annotated[str, Field(description="The location to get weather for.")],
) -> str:
    return f"The weather in {location} is sunny."
```

**Best Practices** (from official docs):
1. Use `Annotated[type, Field(description="...")]` for all parameters
2. Include descriptive docstrings - they become tool descriptions
3. Use `@tool` decorator when you need custom name/description
4. Async functions are fully supported

### 2. Agent Creation Pattern

> **Source**: [Microsoft Learn: Quick Start](https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start)

```python
import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential

async def main():
    # Use context managers for proper resource management
    async with (
        AzureCliCredential() as credential,
        AzureOpenAIChatClient(credential=credential).as_agent(
            name="MyAgent",
            instructions="You are a helpful assistant.",
            tools=[get_weather],
        ) as agent,
    ):
        # Use agent.run() - NOT invoke()!
        result = await agent.run("What's the weather in Paris?")
        print(result.text)

asyncio.run(main())
```

**Key Points**:
- Use `async with ... as agent:` context manager pattern
- Call `agent.run()` for single response
- Call `agent.run_stream()` for streaming

### 3. Tool Precedence (Run-level vs Agent-level)

> **Source**: [Microsoft Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools)

```python
# Agent with base tools
agent = client.as_agent(
    instructions="You are a helpful assistant",
    tools=[get_time]  # Agent-level tool
)

# Run with additional tools - they are COMBINED
result = await agent.run(
    "What's the weather and time in New York?",
    tools=[get_weather]  # Run-level tool
)

# Both get_time AND get_weather are available
# Run-level tools take precedence in case of conflicts
```

### 4. Tool Grouping Pattern (Class-based)

> **Source**: [Microsoft Learn: Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools)

```python
class WeatherTools:
    """Group related tools with shared state."""

    def __init__(self):
        self.last_location = None

    def get_weather(
        self,
        location: Annotated[str, Field(description="Location for weather")],
    ) -> str:
        """Get weather for a location."""
        self.last_location = location
        return f"Weather in {location}: Sunny, 22°C"

    def get_weather_details(self) -> str:
        """Get detailed weather for last requested location."""
        if not self.last_location:
            return "No location specified yet."
        return f"Detailed weather for {self.last_location}: Sunny, High 22°C, Low 15°C, Humidity 60%"

# Register all methods from instance
tools = WeatherTools()
agent = client.as_agent(
    instructions="Weather assistant",
    tools=[tools.get_weather, tools.get_weather_details]
)
```

### 5. Streaming Pattern

> **Source**: [Microsoft Learn: Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools)

```python
# Streaming responses for real-time updates
async for update in agent.run_stream("Tell me about the weather"):
    if update.text:
        print(update.text, end="", flush=True)
```

---

## Implementation Phases

### Phase 1: Core Agent
1. Set up project structure
2. Implement config loading
3. Create basic agent with MAF
4. Add Slack tools

### Phase 2: Slack Integration
1. Set up Slack Bolt handler
2. Connect to agent
3. Implement message handling
4. Add reaction indicators

### Phase 3: RAG System
1. Set up ChromaDB
2. Implement embeddings
3. Add indexer
4. Connect retriever to agent

### Phase 4: Memory & MCP
1. Integrate mem0
2. Add MCP client
3. Connect external tools

### Phase 5: Production
1. Add error handling
2. Implement logging/telemetry
3. Add tests
4. Optimize performance

---

## Dependencies

```toml
[project]
dependencies = [
    # Microsoft Agent Framework
    "agent-framework>=1.0.0b260130",

    # Slack
    "slack-bolt>=1.18.0",
    "slack-sdk>=3.27.0",

    # AI/ML
    "openai>=1.0.0",
    "chromadb>=0.4.0",

    # Web
    "httpx>=0.25.0",
    "beautifulsoup4>=4.12.0",

    # Utils
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]
```

---

## Architecture Validation Summary

### What's Correct (Verified Against Official Docs)

| Pattern | Status | Source |
|---------|--------|--------|
| `@tool` decorator | ✅ Correct | [Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools) |
| `Annotated[type, Field(description=...)]` | ✅ Correct | [Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools) |
| `client.as_agent()` | ✅ Correct | [Quick Start](https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start) |
| Class-based tool grouping | ✅ Correct | [Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools) |
| Async function tools | ✅ Correct | [Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools) |
| Docstring as tool description | ✅ Correct | [Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools) |

### What Was Corrected

| Original | Corrected | Source |
|----------|-----------|--------|
| `agent.invoke()` | `agent.run()` | [Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |
| Custom MCP client | `HostedMCPTool` (built-in) | [Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |
| Custom web fetcher | `HostedWebSearchTool` (built-in) | [Agent Tools](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools) |
| Manual tool calling loop | Automatic via `agent.run()` | [Quick Start](https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start) |

### Architectural Decisions Justified

1. **Single Agent Pattern** ✅
   - Per [MAF Overview](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview): "Use AI Agents for autonomous decision-making, unstructured tasks, conversation-based interactions"
   - Our Slack bot fits this pattern perfectly

2. **Tool-based Design** ✅
   - Per [Function Tools](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools): Tools are the recommended way to extend agent capabilities
   - Both custom (`@tool`) and hosted tools supported

3. **ChromaDB for RAG** ✅ (with alternative)
   - Can keep ChromaDB for full control, OR
   - Use `HostedFileSearchTool` for simpler implementation

4. **MCP Integration** ✅ (simplified!)
   - Originally planned custom MCP client
   - Now using built-in `HostedMCPTool` - much simpler

5. **Slack Bolt for Python** ✅
   - Same library pattern as TypeScript version
   - Async support with `AsyncApp`

---

## Summary

The MAF-Openclaw-Mini implementation follows the same architecture as the TypeScript version:

1. **Single Agent Pattern** - One AI agent with multiple tools
2. **Tool Calling Loop** - Agent automatically handles tool calling via `agent.run()`
3. **Modular Design** - Separate modules for RAG, memory, MCP, Slack
4. **Configuration Driven** - Environment variables control behavior

### Key Advantages of MAF Over Manual Implementation

1. **Built-in MCP Support** - `HostedMCPTool` eliminates custom MCP client code
2. **Built-in Web Search** - `HostedWebSearchTool` eliminates custom web fetcher
3. **Automatic Tool Calling** - No manual loop needed
4. **Type-Safe Tools** - `Annotated` with `Field` provides automatic schema generation
5. **OpenTelemetry Built-in** - Observability without extra setup
6. **Context Managers** - Proper resource management

### Sources

- [Microsoft Agent Framework Overview](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview)
- [Quick Start Guide](https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start)
- [Function Tools Tutorial](https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools)
- [Agent Tools Reference](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools)
- [GitHub Repository](https://github.com/microsoft/agent-framework)
- [PyPI Package](https://pypi.org/project/agent-framework/)
