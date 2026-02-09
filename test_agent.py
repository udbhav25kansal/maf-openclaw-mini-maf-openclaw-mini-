"""
Step 3: Your First MAF Agent
This is a simple test to verify MAF works with your OpenAI API key.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import MAF components
from agent_framework.openai import OpenAIChatClient

async def main():
    # Step 1: Create the AI client (connects to OpenAI)
    # Note: parameter is "model_id" not "model"
    client = OpenAIChatClient(
        model_id=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Step 2: Create an agent with instructions
    agent = client.as_agent(
        name="TestAgent",
        instructions="You are a friendly assistant. Keep responses short (1-2 sentences).",
    )

    # Step 3: Run the agent with a test message
    print("Sending message to agent...")
    result = await agent.run("Hello! What is 2 + 2?")

    # Step 4: Print the response
    print(f"\nAgent response: {result.text}")
    print("\n[SUCCESS] Your MAF agent is working!")

if __name__ == "__main__":
    asyncio.run(main())
