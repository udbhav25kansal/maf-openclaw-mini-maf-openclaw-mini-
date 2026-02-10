"""
MAF Slack Bot — Full-featured Slack assistant using Microsoft Agent Framework.

All agent creation, memory, session management, and MCP lifecycle are handled
by AgentManager + CompositeContextProvider. This file is just the Slack glue.
"""

import asyncio
import os
import sys
import re

from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

from maf_openclaw_mini.agent import AgentManager, AgentContext
from maf_openclaw_mini.agent.tools import set_web_client
from maf_openclaw_mini.storage import init_database, get_stats
from maf_openclaw_mini.rag.vectorstore import init_vectorstore, get_document_count
from maf_openclaw_mini.rag.indexer import start_background_indexer, stop_background_indexer
from maf_openclaw_mini.scheduler import start_task_scheduler, stop_task_scheduler
from maf_openclaw_mini.utils import setup_logging, get_logger, format_error_for_user

# ── Slack App Setup ──────────────────────────────────────

app = AsyncApp(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
)
web_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))

bot_user_id = None
agent_manager = AgentManager()

# ── Helpers ──────────────────────────────────────────────


def remove_bot_mention(text: str, bot_id: str) -> str:
    return re.sub(rf"<@{bot_id}>\s*", "", text).strip()


# ── Event Handlers ───────────────────────────────────────


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
        # Session creation is owned by the ContextProvider — just pass kwargs
        ctx = AgentContext(
            user_id=user,
            channel_id=channel,
            thread_ts=thread_ts,
        )
        response = await agent_manager.run(clean_text, ctx)
        await say(response)
        print(f"[Response] {response[:100]}...")
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
        ctx = AgentContext(
            user_id=user,
            channel_id=channel,
        )
        response = await agent_manager.run(text, ctx)
        await say(response)
        print(f"[Response] {response[:100]}...")
    except Exception as e:
        print(f"[Error] {e}")
        await say(format_error_for_user(e))


# ── Main ─────────────────────────────────────────────────


async def main():
    global bot_user_id

    setup_logging()
    logger = get_logger(__name__)

    print("=" * 50)
    print("MAF Slack Bot - Full Featured!")
    print("=" * 50)

    # Database
    print("Initializing database...")
    init_database()
    stats = get_stats()
    print(f"Database: {stats['sessions']} sessions, {stats['messages']} messages")

    # RAG
    print("Initializing RAG system...")
    init_vectorstore()
    doc_count = await get_document_count()
    print(f"RAG: {doc_count} documents indexed")
    bg_indexer = start_background_indexer()
    print(f"RAG: Background indexer started (interval: {os.getenv('RAG_INDEX_INTERVAL_HOURS', '1')}h)")

    # Scheduler
    print("Initializing Task Scheduler...")
    start_task_scheduler()
    print("Scheduler: Started")

    # Slack client for tools
    set_web_client(web_client)

    # Agent (creates ChatAgent, connects MCP, sets up OTel)
    print("Initializing Agent...")
    await agent_manager.initialize()
    print(f"Memory: {'Enabled' if agent_manager.is_mem0_enabled else 'Disabled'}")

    # Bot identity
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
    print("  - MCP: External tool servers (if configured)")
    print("  - OTel: Instrumentation (if collector configured)")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
        await handler.start_async()
    finally:
        print("\nShutting down...")
        await agent_manager.shutdown()
        await stop_task_scheduler()
        await stop_background_indexer()


if __name__ == "__main__":
    asyncio.run(main())
