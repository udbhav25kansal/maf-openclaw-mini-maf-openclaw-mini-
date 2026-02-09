"""
MCP (Model Context Protocol) Module

Provides integration with external MCP servers like GitHub and Notion.

Following Microsoft Agent Framework patterns:
- https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools

This module:
1. Spawns MCP server processes (e.g., @modelcontextprotocol/server-github)
2. Discovers available tools via JSON-RPC
3. Converts MCP tools to MAF @tool functions
4. Enables the agent to use GitHub, Notion, and other MCP tools

Usage:
------
```python
from maf_openclaw_mini.mcp import (
    initialize_mcp,
    shutdown_mcp,
    is_mcp_enabled,
    mcp_tools_to_maf_tools,
)

# Initialize at startup
await initialize_mcp()

# Get tools for agent
mcp_tools = mcp_tools_to_maf_tools()

# Create agent with MCP tools
agent = client.as_agent(
    name="Assistant",
    instructions="...",
    tools=[...other_tools, *mcp_tools],
)

# Cleanup on shutdown
await shutdown_mcp()
```
"""

from .client import (
    initialize_mcp,
    shutdown_mcp,
    is_mcp_enabled,
    get_connected_servers,
    get_all_mcp_tools,
    execute_mcp_tool,
    parse_tool_name,
)

from .tool_converter import (
    mcp_tools_to_maf_tools,
    create_mcp_tool_function,
)

from .config import (
    load_mcp_config,
    validate_mcp_config,
    MCPConfig,
    MCPServerConfig,
)

__all__ = [
    # Client
    "initialize_mcp",
    "shutdown_mcp",
    "is_mcp_enabled",
    "get_connected_servers",
    "get_all_mcp_tools",
    "execute_mcp_tool",
    "parse_tool_name",
    # Tool converter
    "mcp_tools_to_maf_tools",
    "create_mcp_tool_function",
    # Config
    "load_mcp_config",
    "validate_mcp_config",
    "MCPConfig",
    "MCPServerConfig",
]
