"""
MCP (Model Context Protocol) Client Manager

Manages connections to MCP servers (GitHub, Notion, etc.)
and provides a unified interface for tool discovery and execution.

HOW IT WORKS:
-------------
1. On startup, spawns configured MCP servers as child processes
2. Communicates via stdio (stdin/stdout JSON-RPC)
3. Discovers available tools from each server
4. Routes tool calls to the appropriate server

EXAMPLE:
--------
User: "Create a GitHub issue for the login bug"

1. Agent sees MCP tool: github_create_issue
2. Agent calls: execute_mcp_tool('github', 'create_issue', {...})
3. MCP client sends request to GitHub server
4. Server creates issue via GitHub API
5. Returns: "Created issue #42"

Following Microsoft Agent Framework patterns:
- https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Optional, Any
from dataclasses import dataclass, field

from .config import MCPServerConfig, load_mcp_config


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


@dataclass
class MCPServer:
    """Connected MCP server instance."""
    name: str
    config: MCPServerConfig
    process: Optional[subprocess.Popen] = None
    tools: list[MCPTool] = field(default_factory=list)
    request_id: int = 0
    pending_requests: dict = field(default_factory=dict)
    buffer: str = ""
    _read_task: Optional[asyncio.Task] = None


# Active MCP servers
_servers: dict[str, MCPServer] = {}


async def initialize_mcp() -> None:
    """
    Initialize all configured MCP servers.

    This spawns each configured MCP server as a subprocess
    and discovers available tools from each.
    """
    print("MCP: Initializing servers...")

    config = load_mcp_config()

    if not config.servers:
        print("MCP: No servers configured")
        return

    for server_config in config.servers:
        try:
            await _connect_server(server_config)
        except Exception as e:
            print(f"MCP: Failed to connect to {server_config.name}: {e}")

    connected_count = len(_servers)
    print(f"MCP: {connected_count}/{len(config.servers)} servers connected")


async def _connect_server(config: MCPServerConfig) -> None:
    """
    Connect to a single MCP server.

    Spawns the server process, initializes the connection,
    and discovers available tools.
    """
    print(f"MCP: Connecting to {config.name}...")

    # Merge environment variables
    env = {**os.environ, **config.env}

    # Build command
    # On Windows, we need shell=True for npx to work
    is_windows = sys.platform == "win32"

    if is_windows:
        # Windows: use shell=True
        cmd = [config.command] + config.args
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            shell=True,
            text=True,
            bufsize=1,
        )
    else:
        # Unix: no shell needed
        process = subprocess.Popen(
            [config.command] + config.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

    server = MCPServer(
        name=config.name,
        config=config,
        process=process,
        tools=[],
        request_id=0,
        pending_requests={},
        buffer="",
    )

    _servers[config.name] = server

    # Start async reader for stdout
    server._read_task = asyncio.create_task(_read_server_output(server))

    # Wait for process to start
    await asyncio.sleep(1.0)

    # Check if process is still running
    if process.poll() is not None:
        stderr = process.stderr.read() if process.stderr else ""
        raise Exception(f"Process exited immediately: {stderr}")

    try:
        # Initialize the connection (MCP protocol)
        await _send_request(server, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "maf-openclaw-mini",
                "version": "1.0.0",
            },
        })

        # Send initialized notification
        _send_notification(server, "notifications/initialized", {})

        # Discover available tools
        tools_response = await _send_request(server, "tools/list", {})
        raw_tools = tools_response.get("tools", []) if isinstance(tools_response, dict) else []

        server.tools = [
            MCPTool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in raw_tools
        ]

        print(f"MCP: {config.name} connected with {len(server.tools)} tools")
        for tool in server.tools[:5]:  # Show first 5 tools
            print(f"  - {tool.name}")
        if len(server.tools) > 5:
            print(f"  ... and {len(server.tools) - 5} more")

    except Exception as e:
        print(f"MCP: Failed to initialize {config.name}: {e}")
        process.kill()
        del _servers[config.name]
        raise


async def _read_server_output(server: MCPServer) -> None:
    """
    Continuously read output from the MCP server process.

    Runs in a background task and resolves pending requests
    when responses are received.
    """
    loop = asyncio.get_event_loop()

    while server.process and server.process.poll() is None:
        try:
            # Read line in executor to avoid blocking
            line = await loop.run_in_executor(
                None,
                server.process.stdout.readline
            )

            if not line:
                await asyncio.sleep(0.1)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)

                # Handle response
                if "id" in message:
                    request_id = message["id"]
                    if request_id in server.pending_requests:
                        future = server.pending_requests.pop(request_id)

                        if "error" in message:
                            future.set_exception(
                                Exception(message["error"].get("message", "Unknown error"))
                            )
                        else:
                            future.set_result(message.get("result"))

            except json.JSONDecodeError:
                # Non-JSON output (server logs)
                pass

        except Exception as e:
            if server.process and server.process.poll() is None:
                print(f"MCP: Error reading from {server.name}: {e}")
            break


async def _send_request(server: MCPServer, method: str, params: dict) -> Any:
    """
    Send a JSON-RPC request to an MCP server.

    Returns:
        The result from the server response
    """
    server.request_id += 1
    request_id = server.request_id

    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    # Create future for response
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    server.pending_requests[request_id] = future

    # Send request
    message = json.dumps(request) + "\n"
    server.process.stdin.write(message)
    server.process.stdin.flush()

    # Wait for response with timeout
    try:
        result = await asyncio.wait_for(future, timeout=30.0)
        return result
    except asyncio.TimeoutError:
        server.pending_requests.pop(request_id, None)
        raise Exception(f"Request timeout: {method}")


def _send_notification(server: MCPServer, method: str, params: dict) -> None:
    """
    Send a JSON-RPC notification (no response expected).
    """
    notification = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    }

    message = json.dumps(notification) + "\n"
    server.process.stdin.write(message)
    server.process.stdin.flush()


def get_all_mcp_tools() -> list[tuple[str, MCPTool]]:
    """
    Get all available tools from all connected MCP servers.

    Returns:
        List of (server_name, tool) tuples
    """
    all_tools = []

    for server_name, server in _servers.items():
        for tool in server.tools:
            all_tools.append((server_name, tool))

    return all_tools


async def execute_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: dict,
) -> Any:
    """
    Execute a tool on an MCP server.

    Args:
        server_name: Name of the MCP server (e.g., "github")
        tool_name: Name of the tool (e.g., "create_issue")
        arguments: Tool arguments

    Returns:
        Tool execution result
    """
    server = _servers.get(server_name)

    if not server:
        raise Exception(f"MCP server not connected: {server_name}")

    print(f"MCP: Executing {server_name}/{tool_name}")

    try:
        result = await _send_request(server, "tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        # Handle MCP tool result format
        if isinstance(result, dict):
            content = result.get("content", [])
            if content and isinstance(content, list):
                # Extract text from content array
                texts = [
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                ]
                if texts:
                    return "\n".join(texts)

        return result

    except Exception as e:
        print(f"MCP: Tool execution failed: {e}")
        raise


def parse_tool_name(prefixed_name: str) -> Optional[tuple[str, str]]:
    """
    Parse a prefixed tool name (e.g., "github_create_issue")
    into (server_name, tool_name).

    Returns:
        Tuple of (server_name, tool_name) or None if not found
    """
    for server_name in _servers.keys():
        if prefixed_name.startswith(f"{server_name}_"):
            tool_name = prefixed_name[len(server_name) + 1:]
            return (server_name, tool_name)
    return None


def is_mcp_enabled() -> bool:
    """Check if MCP is initialized with any servers."""
    return len(_servers) > 0


def get_connected_servers() -> list[str]:
    """Get list of connected server names."""
    return list(_servers.keys())


async def shutdown_mcp() -> None:
    """Shutdown all MCP servers."""
    print("MCP: Shutting down servers...")

    for name, server in _servers.items():
        try:
            # Cancel read task
            if server._read_task:
                server._read_task.cancel()

            # Kill process
            if server.process:
                server.process.kill()
                server.process.wait()

            print(f"MCP: Stopped {name}")
        except Exception as e:
            print(f"MCP: Error stopping {name}: {e}")

    _servers.clear()
    print("MCP: Shutdown complete")
