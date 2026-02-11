"""
Step 11: Test SQLite Storage

Tests the database, message store, and session provider.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maf_openclaw_mini.storage import (
    init_database,
    get_or_create_session,
    add_message,
    get_session_messages,
    get_stats,
    SQLiteChatMessageStore,
    ChatMessage,
    SessionContextProvider,
    cleanup_old_sessions,
)


async def main():
    print("=" * 50)
    print("SQLite Storage Test")
    print("=" * 50)

    # Test 1: Initialize database
    print("\n" + "-" * 40)
    print("Test 1: Initializing database...")
    print("-" * 40)

    init_database()
    print("[OK] Database initialized")

    # Test 2: Create session
    print("\n" + "-" * 40)
    print("Test 2: Creating session...")
    print("-" * 40)

    session = get_or_create_session(
        user_id="U_TEST_123",
        channel_id="C_TEST_456",
    )
    print(f"[OK] Session created: {session.id}")
    print(f"     User: {session.user_id}")
    print(f"     Channel: {session.channel_id}")

    # Test 3: Add messages
    print("\n" + "-" * 40)
    print("Test 3: Adding messages...")
    print("-" * 40)

    msg1 = add_message(session.id, "user", "Hello, bot!")
    print(f"[OK] Added user message: {msg1.id}")

    msg2 = add_message(session.id, "assistant", "Hello! How can I help?")
    print(f"[OK] Added assistant message: {msg2.id}")

    msg3 = add_message(session.id, "user", "What's the weather?")
    print(f"[OK] Added user message: {msg3.id}")

    msg4 = add_message(session.id, "assistant", "I can check that for you!")
    print(f"[OK] Added assistant message: {msg4.id}")

    # Test 4: Get session messages
    print("\n" + "-" * 40)
    print("Test 4: Getting session messages...")
    print("-" * 40)

    messages = get_session_messages(session.id)
    print(f"[OK] Retrieved {len(messages)} messages")
    for msg in messages:
        print(f"     [{msg.role}] {msg.content[:40]}...")

    # Test 5: SQLite Message Store
    print("\n" + "-" * 40)
    print("Test 5: Testing SQLiteChatMessageStore...")
    print("-" * 40)

    store = SQLiteChatMessageStore(session_id=session.id)
    chat_messages = await store.list_messages()
    print(f"[OK] Store returned {len(chat_messages)} messages")

    # Add via store
    await store.add_messages([
        ChatMessage(role="user", content="New message via store"),
        ChatMessage(role="assistant", content="Response via store"),
    ])
    print("[OK] Added messages via store")

    # Verify
    updated_messages = await store.list_messages()
    print(f"[OK] Store now has {len(updated_messages)} messages")

    # Test 6: Session Context Provider
    print("\n" + "-" * 40)
    print("Test 6: Testing SessionContextProvider...")
    print("-" * 40)

    provider = SessionContextProvider()

    # Invoking hook
    context = await provider.invoking(
        messages=[],
        user_id="U_TEST_123",
        channel_id="C_TEST_456",
    )
    print(f"[OK] Context provider invoking() called")
    if context.instructions:
        print(f"     Context includes conversation history")

    # Invoked hook
    await provider.invoked(
        request_messages=[{"role": "user", "content": "Test from provider"}],
        response_messages=[{"role": "assistant", "content": "Provider response"}],
        session_id=session.id,
    )
    print("[OK] Context provider invoked() called")

    # Test 7: Statistics
    print("\n" + "-" * 40)
    print("Test 7: Database statistics...")
    print("-" * 40)

    stats = get_stats()
    print(f"[OK] Statistics:")
    print(f"     Sessions: {stats['sessions']}")
    print(f"     Messages: {stats['messages']}")
    print(f"     Unique users: {stats['unique_users']}")

    # Test 8: Serialize/Deserialize
    print("\n" + "-" * 40)
    print("Test 8: Serialize/Deserialize...")
    print("-" * 40)

    serialized = await store.serialize()
    print(f"[OK] Serialized: {len(serialized['messages'])} messages")

    restored_store = await SQLiteChatMessageStore.deserialize(serialized)
    restored_messages = await restored_store.list_messages()
    print(f"[OK] Deserialized: {len(restored_messages)} messages")

    print("\n" + "=" * 50)
    print("[SUCCESS] All storage tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
