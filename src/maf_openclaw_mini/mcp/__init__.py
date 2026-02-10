"""
MCP (Model Context Protocol) Module

Uses MAF's native MCPStdioTool for MCP server integration.

MCPStdioTool handles:
- Subprocess lifecycle (spawn, connect, shutdown)
- Tool discovery via JSON-RPC
- Automatic conversion to FunctionTool
- Async context manager lifecycle
"""

from .config import (
    build_mcp_tools,
    load_mcp_config,
    validate_mcp_config,
    MCPConfig,
    MCPServerConfig,
)

__all__ = [
    "build_mcp_tools",
    "load_mcp_config",
    "validate_mcp_config",
    "MCPConfig",
    "MCPServerConfig",
]
