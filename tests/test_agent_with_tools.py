"""
Step 4: Agent with Tools
This shows how to give your agent the ability to DO things.
"""

import asyncio
import os
from dotenv import load_dotenv
from typing import Annotated
from pydantic import Field

load_dotenv()

from agent_framework.openai import OpenAIChatClient

# ======================
# STEP 1: Define a Tool
# ======================
# A tool is just a Python function with special type hints

def get_weather(
    location: Annotated[str, Field(description="The city to get weather for")]
) -> str:
    """Get the current weather for a location."""
    # In real life, this would call a weather API
    # For now, we'll fake it
    fake_weather = {
        "mumbai": "Sunny, 32C",
        "london": "Rainy, 15C",
        "new york": "Cloudy, 22C",
        "tokyo": "Clear, 25C",
    }

    weather = fake_weather.get(location.lower(), "Unknown location")
    return f"Weather in {location}: {weather}"


def add_numbers(
    a: Annotated[int, Field(description="First number")],
    b: Annotated[int, Field(description="Second number")]
) -> str:
    """Add two numbers together."""
    result = a + b
    return f"The sum of {a} + {b} = {result}"


# ======================
# STEP 2: Create Agent with Tools
# ======================

async def main():
    # Create the client
    client = OpenAIChatClient(
        model_id=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create agent WITH TOOLS
    agent = client.as_agent(
        name="ToolAgent",
        instructions="You are a helpful assistant. Use your tools when needed. Keep responses short.",
        tools=[get_weather, add_numbers],  # <-- Tools go here!
    )

    # Test 1: Ask about weather (agent should use get_weather tool)
    print("=" * 50)
    print("Test 1: Asking about weather...")
    print("=" * 50)
    result = await agent.run("What's the weather like in Mumbai?")
    print(f"Response: {result.text}")

    # Test 2: Ask to add numbers (agent should use add_numbers tool)
    print("\n" + "=" * 50)
    print("Test 2: Asking to add numbers...")
    print("=" * 50)
    result = await agent.run("Can you add 15 and 27 for me?")
    print(f"Response: {result.text}")

    # Test 3: Ask something without a tool (agent just responds normally)
    print("\n" + "=" * 50)
    print("Test 3: Asking a general question (no tool needed)...")
    print("=" * 50)
    result = await agent.run("What is the capital of France?")
    print(f"Response: {result.text}")

    print("\n[SUCCESS] Agent with tools is working!")


if __name__ == "__main__":
    asyncio.run(main())
