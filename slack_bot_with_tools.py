"""
Step 7-15: Slack Bot with Action Tools + RAG + Memory + MCP + Storage + Scheduler + Web Search
Full-featured bot with all capabilities!

Following Microsoft Agent Framework best practices:
- https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools
- https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-rag
- https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory
- https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools
"""

import asyncio
import os
import sys
import re
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import Field

load_dotenv()

# Add src to path for RAG imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Slack imports
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

# MAF imports
# Following official MAF documentation:
# https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools
from agent_framework import tool
from agent_framework.openai import OpenAIChatClient

# RAG imports (following MAF RAG best practices)
# https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-rag
from maf_openclaw_mini.rag.search_tool import search_slack_history
from maf_openclaw_mini.rag.vectorstore import init_vectorstore, get_document_count
from maf_openclaw_mini.rag.indexer import start_background_indexer, stop_background_indexer

# Memory imports (following MAF memory best practices)
# https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory
from maf_openclaw_mini.memory.mem0_client import (
    init_memory,
    is_memory_enabled,
    search_memory,
    add_memory,
    build_memory_context,
)

# MCP imports (following MAF tool integration patterns)
# https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools
from maf_openclaw_mini.mcp import (
    initialize_mcp,
    shutdown_mcp,
    is_mcp_enabled,
    get_connected_servers,
    mcp_tools_to_maf_tools,
)

# Storage imports (following MAF ContextProvider pattern)
# https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory
from maf_openclaw_mini.storage import (
    init_database,
    get_or_create_session,
    add_message as db_add_message,
    get_session_messages,
    get_stats,
    init_session_provider,
    build_conversation_context,
)

# Scheduler imports (task scheduling and reminders)
from maf_openclaw_mini.scheduler import (
    start_task_scheduler,
    stop_task_scheduler,
    set_reminder,
    list_reminders,
    cancel_reminder,
)

# Web search imports (following MAF tool patterns)
from maf_openclaw_mini.tools import web_search, fetch_url

# Utility imports (production hardening)
from maf_openclaw_mini.utils import (
    setup_logging,
    get_logger,
    log_agent_response,
    format_error_for_user,
)

# ======================
# SLACK APP SETUP
# ======================

app = AsyncApp(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
)

# Async web client for API calls
web_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))

bot_user_id = None

# ======================
# HELPER FUNCTIONS
# ======================

async def find_channel(name: str):
    """Find a channel by name."""
    # Remove # prefix if present
    name = name.lstrip("#")

    result = await web_client.conversations_list(types="public_channel,private_channel")
    channels = result.get("channels", [])

    for channel in channels:
        if channel["name"].lower() == name.lower():
            return channel
    return None


async def find_user(name: str):
    """Find a user by name or display name."""
    # Remove @ prefix if present
    name = name.lstrip("@")

    result = await web_client.users_list()
    members = result.get("members", [])

    for user in members:
        if user.get("deleted"):
            continue
        if (user.get("name", "").lower() == name.lower() or
            user.get("real_name", "").lower() == name.lower()):
            return user
    return None


# ======================
# TOOLS - Slack Actions
# ======================

@tool(name="send_slack_message", description="Send a message to a Slack channel or user")
async def send_slack_message(
    target: Annotated[str, Field(description="Channel name (e.g., 'general') or username (e.g., 'john')")],
    message: Annotated[str, Field(description="The message to send")]
) -> str:
    """Send a message to a Slack channel or user."""
    try:
        # Try to find as channel first
        channel = await find_channel(target)
        if channel:
            result = await web_client.chat_postMessage(
                channel=channel["id"],
                text=message
            )
            if result["ok"]:
                return f"Message sent to #{target}"
            else:
                return f"Failed to send: {result.get('error', 'Unknown error')}"

        # Try to find as user
        user = await find_user(target)
        if user:
            # Open DM channel with user
            dm_result = await web_client.conversations_open(users=[user["id"]])
            if dm_result["ok"]:
                channel_id = dm_result["channel"]["id"]
                result = await web_client.chat_postMessage(
                    channel=channel_id,
                    text=message
                )
                if result["ok"]:
                    return f"Message sent to @{target}"

        return f"Could not find channel or user: {target}"
    except Exception as e:
        return f"Error sending message: {str(e)}"


@tool(name="list_slack_channels", description="List all Slack channels the bot has access to")
async def list_slack_channels() -> str:
    """List all Slack channels the bot has access to."""
    try:
        result = await web_client.conversations_list(types="public_channel,private_channel")
        channels = result.get("channels", [])

        # Filter to channels bot is member of
        member_channels = [c for c in channels if c.get("is_member")]

        if not member_channels:
            return "I'm not a member of any channels yet."

        channel_list = "\n".join([f"- #{c['name']}" for c in member_channels[:20]])
        return f"Channels I'm in ({len(member_channels)}):\n{channel_list}"
    except Exception as e:
        return f"Error listing channels: {str(e)}"


@tool(name="list_slack_users", description="List all users in the Slack workspace")
async def list_slack_users() -> str:
    """List all users in the Slack workspace."""
    try:
        result = await web_client.users_list()
        members = result.get("members", [])

        # Filter out bots and deleted users
        real_users = [u for u in members if not u.get("is_bot") and not u.get("deleted")]

        user_list = "\n".join([
            f"- {u.get('real_name', 'Unknown')} (@{u['name']})"
            for u in real_users[:20]
        ])

        return f"Users ({len(real_users)}):\n{user_list}{'...' if len(real_users) > 20 else ''}"
    except Exception as e:
        return f"Error listing users: {str(e)}"


@tool(name="get_channel_info", description="Get information about a Slack channel")
async def get_channel_info(
    channel_name: Annotated[str, Field(description="Channel name without # prefix")]
) -> str:
    """Get information about a Slack channel."""
    try:
        channel = await find_channel(channel_name)
        if not channel:
            return f"Channel not found: {channel_name}"

        info = [
            f"Channel: #{channel['name']}",
            f"ID: {channel['id']}",
            f"Members: {channel.get('num_members', 'Unknown')}",
            f"Topic: {channel.get('topic', {}).get('value', 'No topic set')}",
            f"Purpose: {channel.get('purpose', {}).get('value', 'No purpose set')}",
        ]
        return "\n".join(info)
    except Exception as e:
        return f"Error getting channel info: {str(e)}"


# Basic utility tools
@tool(name="get_current_time", description="Get the current date and time")
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%I:%M %p on %A, %B %d, %Y')}"


@tool(name="calculate", description="Calculate a simple math expression")
def calculate(
    expression: Annotated[str, Field(description="Math expression like '2 + 2' or '10 * 5'")]
) -> str:
    """Calculate a simple math expression."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "I can only do basic math (+, -, *, /)"
    except Exception as e:
        return f"Error: {str(e)}"


# ======================
# CREATE AGENT
# ======================

async def create_agent_with_memory(user_id: str, user_message: str, session_id: Optional[str] = None):
    """
    Create an agent with memory and session context.

    Following MAF ContextProvider best practices:
    - Search memories before agent invocation (mem0)
    - Load conversation history from SQLite (session)
    - Inject as additional context
    - Store after conversation

    Reference: https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory
    """
    client = OpenAIChatClient(
        model_id=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # MEMORY: Search for relevant memories about this user (MAF invoking pattern)
    memory_context = ""
    if is_memory_enabled():
        memories = await search_memory(user_message, user_id, limit=5)
        memory_context = build_memory_context(memories)
        if memory_context:
            print(f"[Memory] Found {len(memories)} relevant memories for user {user_id}")

    # SESSION: Load conversation history from SQLite (MAF ContextProvider pattern)
    session_context = ""
    if session_id:
        messages = get_session_messages(session_id, limit=10)
        if messages:
            session_context = build_conversation_context([
                type('ChatMessage', (), {'role': m.role, 'content': m.content, 'timestamp': m.timestamp})()
                for m in messages
            ])
            print(f"[Session] Loaded {len(messages)} messages from history")

    # Build instructions with memory context
    base_instructions = """You are a helpful Slack assistant. You MUST use your tools to answer questions.

IMPORTANT: Always use your tools! Do NOT say "check your device" or "I don't know" - USE THE TOOLS.

Available tools and WHEN TO USE THEM:

SEARCH & KNOWLEDGE:
- search_slack_history: Use for questions about past discussions, what someone said, or finding information from message history. ALWAYS use this when asked "what did we discuss", "what did X say", "find messages about", etc.

SLACK ACTIONS:
- send_slack_message: Use to send messages to channels or users
- list_slack_channels: Use when asked about channels
- list_slack_users: Use when asked about users/people/who is here
- get_channel_info: Use for details about a specific channel

UTILITIES:
- get_current_time: ALWAYS use this when asked about time, date, or "what time is it"
- calculate: ALWAYS use this for any math questions

REMINDERS:
- set_reminder: Set one-time or recurring reminders (e.g., "remind me in 5 minutes", "every monday at 9am")
- list_reminders: List active reminders for the user
- cancel_reminder: Cancel a reminder by ID

WEB SEARCH:
- web_search: Search the web for current information, news, documentation
- fetch_url: Fetch and read content from a specific URL

MCP TOOLS (if available):
- github_* tools: Use for GitHub operations (issues, repos, PRs)
- notion_* tools: Use for Notion operations (pages, databases)

Examples:
- "What time is it?" -> USE get_current_time tool
- "What did John say about the project?" -> USE search_slack_history tool
- "Find discussions about meetings" -> USE search_slack_history tool
- "List channels" -> USE list_slack_channels tool
- "Remind me in 30 minutes to check email" -> USE set_reminder tool
- "Set a daily reminder at 9am" -> USE set_reminder tool
- "Create a GitHub issue" -> USE the appropriate github_* tool
- "Search Notion pages" -> USE the appropriate notion_* tool
- "What's the latest news about Python?" -> USE web_search tool
- "Read this article: https://..." -> USE fetch_url tool

When citing search results, mention the channel and who said it.
Keep responses concise. Use Slack formatting: *bold*, _italic_, `code`."""

    # Inject context (memory + session history)
    instructions = base_instructions
    if session_context:
        instructions = f"{instructions}\n\n{session_context}"
    if memory_context:
        instructions = f"{instructions}\n\n{memory_context}"

    # Build tools list
    tools = [
        # RAG tool (following MAF best practices)
        search_slack_history,
        # Slack action tools
        send_slack_message,
        list_slack_channels,
        list_slack_users,
        get_channel_info,
        # Utility tools
        get_current_time,
        calculate,
        # Reminder tools (task scheduler)
        set_reminder,
        list_reminders,
        cancel_reminder,
        # Web search tools
        web_search,
        fetch_url,
    ]

    # Add MCP tools if available (following MAF tool integration patterns)
    if is_mcp_enabled():
        mcp_tools = mcp_tools_to_maf_tools()
        tools.extend(mcp_tools)
        print(f"[MCP] Added {len(mcp_tools)} MCP tools to agent")

    return client.as_agent(
        name="SlackAssistant",
        instructions=instructions,
        tools=tools,
    )


# ======================
# MESSAGE HANDLERS
# ======================

def remove_bot_mention(text: str, bot_id: str) -> str:
    return re.sub(rf"<@{bot_id}>\s*", "", text).strip()


@app.event("app_mention")
async def handle_mention(event, say):
    global bot_user_id

    text = event.get("text", "")
    user = event.get("user")
    channel = event.get("channel")
    thread_ts = event.get("thread_ts")

    if bot_user_id:
        clean_text = remove_bot_mention(text, bot_user_id)
    else:
        clean_text = text

    print(f"[Mention] User {user} in {channel}: {clean_text}")

    if not clean_text:
        await say("Hi! How can I help you? I can send messages, list channels/users, and more!")
        return

    try:
        # SESSION: Get or create session (MAF ContextProvider pattern)
        session = get_or_create_session(user, channel, thread_ts)

        # Create agent with memory + session context (MAF pattern)
        agent = await create_agent_with_memory(user, clean_text, session.id)
        result = await agent.run(clean_text)
        await say(result.text)
        print(f"[Response] {result.text[:100]}...")

        # SESSION: Store messages in SQLite (MAF invoked pattern)
        db_add_message(session.id, "user", clean_text)
        db_add_message(session.id, "assistant", result.text)

        # MEMORY: Store conversation for future reference (MAF invoked pattern)
        if is_memory_enabled():
            asyncio.create_task(add_memory(
                messages=[
                    {"role": "user", "content": clean_text},
                    {"role": "assistant", "content": result.text},
                ],
                user_id=user,
            ))
    except Exception as e:
        print(f"[Error] {e}")
        await say(format_error_for_user(e))


@app.event("message")
async def handle_dm(event, say):
    channel = event.get("channel", "")
    if not channel.startswith("D"):
        return

    if event.get("subtype") == "bot_message" or event.get("bot_id"):
        return

    text = event.get("text", "")
    user = event.get("user")

    if not text:
        return

    print(f"[DM] User {user}: {text}")

    try:
        # SESSION: Get or create session for DM (MAF ContextProvider pattern)
        session = get_or_create_session(user, channel)

        # Create agent with memory + session context (MAF pattern)
        agent = await create_agent_with_memory(user, text, session.id)
        result = await agent.run(text)
        await say(result.text)
        print(f"[Response] {result.text[:100]}...")

        # SESSION: Store messages in SQLite (MAF invoked pattern)
        db_add_message(session.id, "user", text)
        db_add_message(session.id, "assistant", result.text)

        # MEMORY: Store conversation for future reference (MAF invoked pattern)
        if is_memory_enabled():
            asyncio.create_task(add_memory(
                messages=[
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": result.text},
                ],
                user_id=user,
            ))
    except Exception as e:
        print(f"[Error] {e}")
        await say(format_error_for_user(e))


# ======================
# MAIN
# ======================

async def main():
    global bot_user_id

    # Initialize logging (production hardening)
    setup_logging()
    logger = get_logger(__name__)

    print("=" * 50)
    print("MAF Slack Bot - Full Featured!")
    print("=" * 50)

    # Initialize SQLite database (MAF ContextProvider pattern)
    print("Initializing database...")
    init_database()
    stats = get_stats()
    print(f"Database: {stats['sessions']} sessions, {stats['messages']} messages")

    # Initialize RAG vector store
    print("Initializing RAG system...")
    init_vectorstore()
    doc_count = await get_document_count()
    print(f"RAG: {doc_count} documents indexed")

    # Start background indexer
    bg_indexer = start_background_indexer()
    print(f"RAG: Background indexer started (interval: {os.getenv('RAG_INDEX_INTERVAL_HOURS', '1')}h)")

    if doc_count == 0:
        print("  TIP: Run 'python test_rag.py' to index messages first")

    # Initialize Memory system (MAF pattern)
    print("Initializing Memory system...")
    init_memory()
    print(f"Memory: {'Enabled' if is_memory_enabled() else 'Disabled'}")

    # Initialize MCP system (MAF tool integration pattern)
    print("Initializing MCP system...")
    try:
        await initialize_mcp()
        if is_mcp_enabled():
            servers = get_connected_servers()
            print(f"MCP: Enabled ({', '.join(servers)})")
        else:
            print("MCP: No servers connected")
    except Exception as e:
        print(f"MCP: Failed to initialize - {e}")

    # Initialize Task Scheduler (reminders)
    print("Initializing Task Scheduler...")
    start_task_scheduler()
    print("Scheduler: Started")

    # Get bot user ID
    auth = await web_client.auth_test()
    bot_user_id = auth["user_id"]
    bot_name = auth.get("user", "Bot")
    print(f"Bot: @{bot_name} (ID: {bot_user_id})")

    print("")
    print("Features:")
    print("  - Storage: SQLite session & message history")
    print("  - RAG: Search past Slack messages")
    print("  - Memory: Remember facts about users (mem0)")
    print("  - Scheduler: Reminders and scheduled messages")
    print("  - Web Search: Search the web for current info")
    print("  - Tools: Slack actions, time, math")
    if is_mcp_enabled():
        print(f"  - MCP: {', '.join(get_connected_servers())} integration")
    print("")
    print("Try these commands:")
    print('  "What did we discuss about hello?"')
    print('  "I love Python programming"  (bot will remember!)')
    print('  "What do you remember about me?"')
    print('  "What time is it?"')
    print('  "Remind me in 30 minutes to check email"')
    print('  "Set a daily reminder at 9am to standup"')
    print('  "Search the web for Python 3.12 new features"')
    if is_mcp_enabled():
        if "github" in get_connected_servers():
            print('  "List my GitHub repos"')
        if "notion" in get_connected_servers():
            print('  "Search Notion for meeting notes"')
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
        await handler.start_async()
    finally:
        # Cleanup on exit
        print("\nShutting down...")

        # Stop task scheduler
        await stop_task_scheduler()

        # Stop background indexer
        await stop_background_indexer()

        # Stop MCP
        if is_mcp_enabled():
            await shutdown_mcp()


if __name__ == "__main__":
    asyncio.run(main())
