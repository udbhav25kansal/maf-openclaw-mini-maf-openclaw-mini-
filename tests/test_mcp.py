"""
Step 10: Test MCP Integration

This tests the MCP (Model Context Protocol) integration for GitHub and Notion.

Requirements:
- GITHUB_PERSONAL_ACCESS_TOKEN in .env for GitHub MCP
- NOTION_API_TOKEN in .env for Notion MCP
- Node.js/npx installed (for running MCP servers)
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from maf_openclaw_mini.mcp import (
    initialize_mcp,
    shutdown_mcp,
    is_mcp_enabled,
    get_connected_servers,
    get_all_mcp_tools,
    mcp_tools_to_maf_tools,
)


async def main():
    print("=" * 50)
    print("MCP Integration Test")
    print("=" * 50)

    # Check environment
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    notion_token = os.getenv("NOTION_API_TOKEN")

    print("\nEnvironment:")
    print(f"  GITHUB_PERSONAL_ACCESS_TOKEN: {'Set' if github_token else 'Not set'}")
    print(f"  NOTION_API_TOKEN: {'Set' if notion_token else 'Not set'}")

    if not github_token and not notion_token:
        print("\n[WARNING] No MCP tokens configured.")
        print("Set GITHUB_PERSONAL_ACCESS_TOKEN or NOTION_API_TOKEN in .env")
        print("to enable MCP integration.")
        return

    # Test 1: Initialize MCP
    print("\n" + "-" * 40)
    print("Test 1: Initializing MCP...")
    print("-" * 40)

    try:
        await initialize_mcp()
    except Exception as e:
        print(f"[ERROR] Failed to initialize MCP: {e}")
        return

    # Test 2: Check status
    print("\n" + "-" * 40)
    print("Test 2: Checking MCP status...")
    print("-" * 40)

    print(f"MCP Enabled: {is_mcp_enabled()}")
    print(f"Connected Servers: {get_connected_servers()}")

    # Test 3: List available tools
    print("\n" + "-" * 40)
    print("Test 3: Listing available tools...")
    print("-" * 40)

    all_tools = get_all_mcp_tools()
    print(f"Total tools: {len(all_tools)}")

    for server_name, tool in all_tools[:10]:
        desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
        print(f"  [{server_name}] {tool.name}: {desc}")

    if len(all_tools) > 10:
        print(f"  ... and {len(all_tools) - 10} more")

    # Test 4: Convert to MAF tools
    print("\n" + "-" * 40)
    print("Test 4: Converting to MAF tools...")
    print("-" * 40)

    maf_tools = mcp_tools_to_maf_tools()
    print(f"Converted {len(maf_tools)} tools to MAF format")

    for tool_func in maf_tools[:5]:
        # FunctionTool objects have name attribute, not __name__
        name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
        print(f"  - {name}")

    if len(maf_tools) > 5:
        print(f"  ... and {len(maf_tools) - 5} more")

    # Test 5: Cleanup
    print("\n" + "-" * 40)
    print("Test 5: Shutting down MCP...")
    print("-" * 40)

    await shutdown_mcp()

    print("\n" + "=" * 50)
    print("MCP Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
