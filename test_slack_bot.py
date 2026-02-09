"""
Step 5: Slack Bot with MAF Agent
This connects your MAF agent to Slack.
"""

import asyncio
import os
import re
from dotenv import load_dotenv
from typing import Annotated
from pydantic import Field

load_dotenv()

# Slack imports
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

# MAF imports
from agent_framework.openai import OpenAIChatClient

# ======================
# STEP 1: Set up Slack App
# ======================

app = AsyncApp(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
)

# Store the bot's user ID (so we can ignore our own messages)
bot_user_id = None

# ======================
# STEP 2: Define Tools
# ======================

def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current time: {now.strftime('%I:%M %p on %B %d, %Y')}"


def calculate(
    expression: Annotated[str, Field(description="Math expression like '2 + 2' or '10 * 5'")]
) -> str:
    """Calculate a math expression."""
    try:
        # Simple and safe eval for basic math
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        else:
            return "Sorry, I can only do basic math (+, -, *, /)"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# ======================
# STEP 3: Create MAF Agent
# ======================

def create_agent():
    client = OpenAIChatClient(
        model_id=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    return client.as_agent(
        name="SlackBot",
        instructions="""You are a helpful Slack assistant.
Keep responses concise and friendly.
Use Slack formatting: *bold*, _italic_, `code`.
Use your tools when appropriate.""",
        tools=[get_time, calculate],
    )


# ======================
# STEP 4: Handle Messages
# ======================

def remove_bot_mention(text: str, bot_id: str) -> str:
    """Remove the @bot mention from the message."""
    return re.sub(rf"<@{bot_id}>\s*", "", text).strip()


@app.event("app_mention")
async def handle_mention(event, say):
    """Handle when someone @mentions the bot."""
    global bot_user_id

    user = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")

    print(f"[Mention] User {user} in {channel}: {text}")

    # Remove the @bot mention from the text
    if bot_user_id:
        clean_text = remove_bot_mention(text, bot_user_id)
    else:
        clean_text = text

    # Skip empty messages
    if not clean_text:
        await say("Hi! How can I help you?")
        return

    # Create agent and get response
    try:
        agent = create_agent()
        result = await agent.run(clean_text)
        await say(result.text)
    except Exception as e:
        print(f"Error: {e}")
        await say(f"Sorry, I encountered an error: {str(e)}")


@app.event("message")
async def handle_dm(event, say):
    """Handle direct messages to the bot."""
    # Only respond to DMs (channel starts with 'D')
    channel = event.get("channel", "")
    if not channel.startswith("D"):
        return  # Not a DM, ignore

    # Ignore bot messages
    if event.get("subtype") == "bot_message":
        return

    user = event.get("user")
    text = event.get("text", "")

    print(f"[DM] User {user}: {text}")

    # Skip empty messages
    if not text:
        return

    # Create agent and get response
    try:
        agent = create_agent()
        result = await agent.run(text)
        await say(result.text)
    except Exception as e:
        print(f"Error: {e}")
        await say(f"Sorry, I encountered an error: {str(e)}")


# ======================
# STEP 5: Start the Bot
# ======================

async def main():
    global bot_user_id

    print("=" * 50)
    print("Starting Slack Bot with MAF Agent...")
    print("=" * 50)

    # Get the bot's user ID
    from slack_sdk.web.async_client import AsyncWebClient
    client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    auth = await client.auth_test()
    bot_user_id = auth["user_id"]
    print(f"Bot User ID: {bot_user_id}")

    # Start the bot using Socket Mode
    handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))

    print("")
    print("Bot is running! Try:")
    print("  1. @mention the bot in a channel")
    print("  2. Send a direct message to the bot")
    print("")
    print("Example messages:")
    print('  - "What time is it?"')
    print('  - "Calculate 25 * 4"')
    print('  - "Hello!"')
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
