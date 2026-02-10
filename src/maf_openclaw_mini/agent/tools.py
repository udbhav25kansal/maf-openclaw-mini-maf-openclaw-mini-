"""
Slack Action Tools

MAF @tool decorated functions for Slack operations.
Extracted from the monolith for use with ChatAgent.
"""

from typing import Annotated, Optional
from datetime import datetime
from pydantic import Field
from agent_framework import tool
from slack_sdk.web.async_client import AsyncWebClient

# Module-level web client reference, set at startup via set_web_client()
_web_client: Optional[AsyncWebClient] = None


def set_web_client(client: AsyncWebClient) -> None:
    """Set the Slack web client reference for tools to use."""
    global _web_client
    _web_client = client


def _get_client() -> AsyncWebClient:
    if _web_client is None:
        raise RuntimeError("Slack web client not initialized. Call set_web_client() first.")
    return _web_client


# ── Helpers ──────────────────────────────────────────────

async def find_channel(name: str):
    """Find a channel by name."""
    name = name.lstrip("#")
    client = _get_client()
    result = await client.conversations_list(types="public_channel,private_channel")
    channels = result.get("channels", [])
    for channel in channels:
        if channel["name"].lower() == name.lower():
            return channel
    return None


async def find_user(name: str):
    """Find a user by name or display name."""
    name = name.lstrip("@")
    client = _get_client()
    result = await client.users_list()
    members = result.get("members", [])
    for user in members:
        if user.get("deleted"):
            continue
        if (
            user.get("name", "").lower() == name.lower()
            or user.get("real_name", "").lower() == name.lower()
        ):
            return user
    return None


# ── Slack Action Tools ───────────────────────────────────

@tool(name="send_slack_message", description="Send a message to a Slack channel or user")
async def send_slack_message(
    target: Annotated[str, Field(description="Channel name (e.g., 'general') or username (e.g., 'john')")],
    message: Annotated[str, Field(description="The message to send")],
) -> str:
    """Send a message to a Slack channel or user."""
    client = _get_client()
    try:
        channel = await find_channel(target)
        if channel:
            result = await client.chat_postMessage(channel=channel["id"], text=message)
            if result["ok"]:
                return f"Message sent to #{target}"
            else:
                return f"Failed to send: {result.get('error', 'Unknown error')}"

        user = await find_user(target)
        if user:
            dm_result = await client.conversations_open(users=[user["id"]])
            if dm_result["ok"]:
                channel_id = dm_result["channel"]["id"]
                result = await client.chat_postMessage(channel=channel_id, text=message)
                if result["ok"]:
                    return f"Message sent to @{target}"

        return f"Could not find channel or user: {target}"
    except Exception as e:
        return f"Error sending message: {str(e)}"


@tool(name="list_slack_channels", description="List all Slack channels the bot has access to")
async def list_slack_channels() -> str:
    """List all Slack channels the bot has access to."""
    client = _get_client()
    try:
        result = await client.conversations_list(types="public_channel,private_channel")
        channels = result.get("channels", [])
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
    client = _get_client()
    try:
        result = await client.users_list()
        members = result.get("members", [])
        real_users = [u for u in members if not u.get("is_bot") and not u.get("deleted")]
        user_list = "\n".join(
            [f"- {u.get('real_name', 'Unknown')} (@{u['name']})" for u in real_users[:20]]
        )
        return f"Users ({len(real_users)}):\n{user_list}{'...' if len(real_users) > 20 else ''}"
    except Exception as e:
        return f"Error listing users: {str(e)}"


@tool(name="get_channel_info", description="Get information about a Slack channel")
async def get_channel_info(
    channel_name: Annotated[str, Field(description="Channel name without # prefix")],
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


# ── Utility Tools ────────────────────────────────────────

@tool(name="get_current_time", description="Get the current date and time")
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%I:%M %p on %A, %B %d, %Y')}"


@tool(name="calculate", description="Calculate a simple math expression")
def calculate(
    expression: Annotated[str, Field(description="Math expression like '2 + 2' or '10 * 5'")],
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
