"""
Step 8: Test RAG System

This script:
1. Indexes messages from Slack channels
2. Tests the search functionality
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from maf_openclaw_mini.rag.indexer import run_indexer, index_all_channels
from maf_openclaw_mini.rag.search_tool import search_slack_history
from maf_openclaw_mini.rag.vectorstore import get_document_count, init_vectorstore


async def test_search(query: str):
    """Test the search tool with a query."""
    print(f"\nSearching for: '{query}'")
    print("-" * 40)
    result = await search_slack_history(query=query, limit=3)
    print(result)


async def main():
    print("=" * 50)
    print("RAG System Test")
    print("=" * 50)

    # Initialize vector store
    init_vectorstore()

    # Check current document count
    count = await get_document_count()
    print(f"\nCurrent documents in vector store: {count}")

    if count == 0:
        print("\nNo documents indexed. Running indexer...")
        await run_indexer()
        count = await get_document_count()

    if count == 0:
        print("\nNo messages to index. Make sure:")
        print("  1. The bot is added to some channels")
        print("  2. There are messages in those channels")
        return

    print(f"\n{count} documents available for search")

    # Test searches
    print("\n" + "=" * 50)
    print("Testing Search")
    print("=" * 50)

    # Test 1: General search
    await test_search("hello")

    # Test 2: Search for something specific
    await test_search("meeting")

    print("\n" + "=" * 50)
    print("RAG Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
