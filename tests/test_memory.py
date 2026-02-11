"""
Test Memory System

Tests the MAF-native CompositeContextProvider which wraps
SessionContextProvider + Mem0Provider.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

load_dotenv()

from agent_framework import ChatMessage

from maf_openclaw_mini.storage import init_database
from maf_openclaw_mini.storage.session_provider import SessionContextProvider
from maf_openclaw_mini.memory.composite_provider import CompositeContextProvider


async def main():
    print("=" * 50)
    print("Memory System Test (MAF-native)")
    print("=" * 50)

    # Initialize database (needed for session provider)
    init_database()

    # Build composite provider
    session_provider = SessionContextProvider()
    provider = CompositeContextProvider(session_provider=session_provider)

    print(f"\nMem0 enabled: {provider.is_mem0_enabled}")

    test_user = "U_TEST_USER"
    test_channel = "C_TEST_CHANNEL"

    # Test 1: invoking() — should return context with user instructions
    print("\n" + "-" * 40)
    print("Test 1: invoking() — context injection")
    print("-" * 40)

    msg = ChatMessage(role="user", text="I love Python programming and coffee")
    context = await provider.invoking(
        msg,
        user_id=test_user,
        channel_id=test_channel,
    )

    print(f"Context instructions present: {bool(context.instructions)}")
    if context.instructions:
        print(f"Instructions preview: {context.instructions[:200]}...")

    # Test 2: invoked() — should persist to session + mem0
    print("\n" + "-" * 40)
    print("Test 2: invoked() — persistence")
    print("-" * 40)

    request = ChatMessage(role="user", text="I love Python programming and coffee")
    response = ChatMessage(role="assistant", text="That's great! Python is a wonderful language.")

    await provider.invoked(
        request,
        response,
        user_id=test_user,
        channel_id=test_channel,
    )
    print("invoked() completed (messages persisted)")

    # Test 3: invoking() again — should include history
    print("\n" + "-" * 40)
    print("Test 3: invoking() again — conversation history")
    print("-" * 40)

    msg2 = ChatMessage(role="user", text="What do you remember about me?")
    context2 = await provider.invoking(
        msg2,
        user_id=test_user,
        channel_id=test_channel,
    )

    print(f"Context instructions present: {bool(context2.instructions)}")
    if context2.instructions:
        print(f"Instructions:\n{context2.instructions}")

    print("\n" + "=" * 50)
    print("Memory Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
