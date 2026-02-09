"""
Step 9: Test Memory System

This tests the mem0 integration for storing and retrieving user memories.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from maf_openclaw_mini.memory.mem0_client import (
    init_memory,
    is_memory_enabled,
    add_memory,
    search_memory,
    get_all_memories,
    build_memory_context,
)


async def main():
    print("=" * 50)
    print("Memory System Test (mem0)")
    print("=" * 50)

    # Initialize memory
    if not init_memory():
        print("\nMemory system not available.")
        print("Check MEM0_API_KEY and MEMORY_ENABLED in .env")
        return

    print(f"\nMemory enabled: {is_memory_enabled()}")

    # Test user ID
    test_user = "U_TEST_USER"

    # Test 1: Add a memory
    print("\n" + "-" * 40)
    print("Test 1: Adding memory...")
    print("-" * 40)

    messages = [
        {"role": "user", "content": "I love Python programming and coffee"},
        {"role": "assistant", "content": "That's great! Python is a wonderful language."},
    ]

    result = await add_memory(messages, test_user)
    if result:
        print("Memory added successfully!")
        print(f"Response: {result}")
    else:
        print("Failed to add memory")

    # Test 2: Search memories
    print("\n" + "-" * 40)
    print("Test 2: Searching memories...")
    print("-" * 40)

    memories = await search_memory("programming", test_user)
    print(f"Found {len(memories)} memories")
    for mem in memories:
        print(f"  - {mem.get('memory', 'N/A')}")

    # Test 3: Get all memories
    print("\n" + "-" * 40)
    print("Test 3: Getting all memories...")
    print("-" * 40)

    all_memories = await get_all_memories(test_user)
    print(f"Total memories for user: {len(all_memories)}")

    # Test 4: Build context
    print("\n" + "-" * 40)
    print("Test 4: Building memory context...")
    print("-" * 40)

    context = build_memory_context(memories)
    if context:
        print("Memory context:")
        print(context)
    else:
        print("No context to build (no memories found)")

    print("\n" + "=" * 50)
    print("Memory Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
