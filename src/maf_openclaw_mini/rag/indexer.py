"""
Slack Message Indexer

Fetches messages from Slack channels and indexes them in the vector store.

Following Microsoft RAG best practices:
- Chunking: Messages are naturally chunked (one per document)
- Metadata: Each document includes source information for citation
- Batch processing: Efficient embedding generation

Reference: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
"""

import os
import asyncio
from typing import Optional
from datetime import datetime
from slack_sdk.web.async_client import AsyncWebClient
from dotenv import load_dotenv

from .embeddings import create_embeddings
from .vectorstore import add_documents, document_exists, get_document_count, init_vectorstore

load_dotenv()

# Slack client
web_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))


async def get_channels() -> list[dict]:
    """Get all channels the bot is a member of."""
    result = await web_client.conversations_list(
        types="public_channel,private_channel"
    )
    channels = result.get("channels", [])
    # Filter to channels bot is a member of
    return [c for c in channels if c.get("is_member")]


async def get_channel_messages(
    channel_id: str,
    limit: int = 100,
    oldest: Optional[float] = None,
) -> list[dict]:
    """Get messages from a channel."""
    kwargs = {
        "channel": channel_id,
        "limit": limit,
    }
    if oldest:
        kwargs["oldest"] = str(oldest)

    result = await web_client.conversations_history(**kwargs)
    return result.get("messages", [])


async def get_user_name(user_id: str) -> str:
    """Get username from user ID."""
    try:
        result = await web_client.users_info(user=user_id)
        user = result.get("user", {})
        return user.get("real_name") or user.get("name") or "Unknown"
    except Exception:
        return "Unknown"


def format_timestamp(ts: str) -> str:
    """Convert Slack timestamp to readable format."""
    try:
        timestamp = float(ts)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


async def index_channel(
    channel_id: str,
    channel_name: str,
    limit: int = 100,
) -> int:
    """
    Index messages from a single channel.

    Returns:
        Number of new messages indexed
    """
    messages = await get_channel_messages(channel_id, limit=limit)

    # Filter to actual messages (not system messages, etc.)
    user_messages = [
        m for m in messages
        if m.get("type") == "message"
        and not m.get("subtype")  # No system messages
        and m.get("text")  # Has text content
        and m.get("user")  # Has a user
    ]

    if not user_messages:
        return 0

    # Check which messages are already indexed
    new_messages = []
    for msg in user_messages:
        doc_id = f"{channel_id}_{msg['ts']}"
        if not await document_exists(doc_id):
            new_messages.append(msg)

    if not new_messages:
        return 0

    # Get user names for all messages
    user_ids = list(set(m["user"] for m in new_messages))
    user_names = {}
    for user_id in user_ids:
        user_names[user_id] = await get_user_name(user_id)

    # Prepare documents for indexing
    ids = []
    texts = []
    metadatas = []

    for msg in new_messages:
        doc_id = f"{channel_id}_{msg['ts']}"
        ids.append(doc_id)
        texts.append(msg["text"])
        metadatas.append({
            "channel_id": channel_id,
            "channel_name": channel_name,
            "user_id": msg["user"],
            "user_name": user_names.get(msg["user"], "Unknown"),
            "timestamp": format_timestamp(msg["ts"]),
            "ts": msg["ts"],
        })

    # Create embeddings in batch (more efficient)
    embeddings = await create_embeddings(texts)

    # Add to vector store
    await add_documents(ids, texts, embeddings, metadatas)

    return len(new_messages)


async def index_all_channels(limit_per_channel: int = 100) -> dict:
    """
    Index messages from all channels the bot is a member of.

    Returns:
        Summary of indexing results
    """
    # Initialize vector store
    init_vectorstore()

    channels = await get_channels()

    results = {
        "channels_processed": 0,
        "messages_indexed": 0,
        "errors": [],
    }

    for channel in channels:
        try:
            count = await index_channel(
                channel_id=channel["id"],
                channel_name=channel["name"],
                limit=limit_per_channel,
            )
            results["channels_processed"] += 1
            results["messages_indexed"] += count
            print(f"Indexed {count} messages from #{channel['name']}")
        except Exception as e:
            results["errors"].append(f"#{channel['name']}: {str(e)}")
            print(f"Error indexing #{channel['name']}: {e}")

    return results


async def run_indexer() -> None:
    """Run the indexer as a one-time job."""
    print("=" * 50)
    print("Starting Slack Message Indexer")
    print("=" * 50)

    initial_count = await get_document_count()
    print(f"Documents before indexing: {initial_count}")

    results = await index_all_channels()

    final_count = await get_document_count()
    print("")
    print("=" * 50)
    print("Indexing Complete!")
    print("=" * 50)
    print(f"Channels processed: {results['channels_processed']}")
    print(f"New messages indexed: {results['messages_indexed']}")
    print(f"Total documents: {final_count}")

    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(run_indexer())
