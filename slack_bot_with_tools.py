"""
Step 7 & 8: Slack Bot with Action Tools + RAG
The bot can now DO things in Slack AND search message history!

Following Microsoft Agent Framework best practices:
- https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools
- https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-rag
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

# Memory imports (following MAF memory best practices)
# https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory
from maf_openclaw_mini.memory.mem0_client import (
    init_memory,
    is_memory_enabled,
    search_memory,
    add_memory,
    build_memory_context,
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

async def create_agent_with_memory(user_id: str, user_message: str):
    """
    Create an agent with memory context.

    Following MAF memory best practices:
    - Search memories before agent invocation
    - Inject relevant memories as additional context
    - Store memories after conversation

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

Examples:
- "What time is it?" -> USE get_current_time tool
- "What did John say about the project?" -> USE search_slack_history tool
- "Find discussions about meetings" -> USE search_slack_history tool
- "List channels" -> USE list_slack_channels tool

When citing search results, mention the channel and who said it.
Keep responses concise. Use Slack formatting: *bold*, _italic_, `code`."""

    # Inject memory context if available
    if memory_context:
        instructions = f"{base_instructions}\n\n{memory_context}"
    else:
        instructions = base_instructions

    return client.as_agent(
        name="SlackAssistant",
        instructions=instructions,
        tools=[
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
        ],
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

    if bot_user_id:
        clean_text = remove_bot_mention(text, bot_user_id)
    else:
        clean_text = text

    print(f"[Mention] User {user} in {channel}: {clean_text}")

    if not clean_text:
        await say("Hi! How can I help you? I can send messages, list channels/users, and more!")
        return

    try:
        # Create agent with memory context (MAF pattern)
        agent = await create_agent_with_memory(user, clean_text)
        result = await agent.run(clean_text)
        await say(result.text)
        print(f"[Response] {result.text[:100]}...")

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
        await say(f"Sorry, I encountered an error: {str(e)}")


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
        # Create agent with memory context (MAF pattern)
        agent = await create_agent_with_memory(user, text)
        result = await agent.run(text)
        await say(result.text)
        print(f"[Response] {result.text[:100]}...")

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
        await say(f"Sorry, I encountered an error: {str(e)}")


# ======================
# MAIN
# ======================

async def main():
    global bot_user_id

    print("=" * 50)
    print("MAF Slack Bot - With Action Tools + RAG!")
    print("=" * 50)

    # Initialize RAG vector store
    print("Initializing RAG system...")
    init_vectorstore()
    doc_count = await get_document_count()
    print(f"RAG: {doc_count} documents indexed")

    if doc_count == 0:
        print("  TIP: Run 'python test_rag.py' to index messages first")

    # Initialize Memory system (MAF pattern)
    print("Initializing Memory system...")
    init_memory()
    print(f"Memory: {'Enabled' if is_memory_enabled() else 'Disabled'}")

    # Get bot user ID
    auth = await web_client.auth_test()
    bot_user_id = auth["user_id"]
    bot_name = auth.get("user", "Bot")
    print(f"Bot: @{bot_name} (ID: {bot_user_id})")

    print("")
    print("Features:")
    print("  - RAG: Search past Slack messages")
    print("  - Memory: Remember facts about users (mem0)")
    print("  - Tools: Slack actions, time, math")
    print("")
    print("Try these commands:")
    print('  "What did we discuss about hello?"')
    print('  "I love Python programming"  (bot will remember!)')
    print('  "What do you remember about me?"')
    print('  "What time is it?"')
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
