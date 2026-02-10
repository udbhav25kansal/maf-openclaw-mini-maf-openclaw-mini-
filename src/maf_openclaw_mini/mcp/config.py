"""
MCP Configuration

Loads MCP server configurations from:
1. mcp-config.json file (if exists)
2. Environment variables (fallback)

Following Microsoft Agent Framework patterns for external tool integration.
Reference: https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools
"""

import os
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv

from agent_framework import MCPStdioTool

load_dotenv()


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPConfig:
    """Configuration for all MCP servers."""
    servers: list[MCPServerConfig] = field(default_factory=list)


def load_mcp_config() -> MCPConfig:
    """
    Load MCP configuration.

    Priority:
    1. mcp-config.json in project root
    2. Environment variables (GITHUB_PERSONAL_ACCESS_TOKEN, NOTION_API_TOKEN)

    Returns:
        MCPConfig with list of configured servers
    """
    # Try loading from config file first
    config_path = os.path.join(os.getcwd(), "mcp-config.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = json.load(f)

            servers = []
            for server_data in data.get("servers", []):
                # Substitute environment variables in env values
                env = server_data.get("env", {})
                for key, value in env.items():
                    if isinstance(value, str) and value.startswith("$"):
                        env_var = value[1:]
                        env[key] = os.getenv(env_var, "")

                servers.append(MCPServerConfig(
                    name=server_data["name"],
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=env,
                ))

            print(f"MCP: Loaded config from {config_path}")
            return MCPConfig(servers=servers)

        except Exception as e:
            print(f"MCP: Failed to load config file: {e}")

    # Fallback: build config from environment variables
    print("MCP: Building config from environment variables")

    servers = []

    # GitHub server
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN") or os.getenv("GITHUB_TOKEN")
    if github_token:
        servers.append(MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
        ))
        print("MCP: GitHub server configured")
    else:
        print("MCP: GitHub server not configured (missing GITHUB_PERSONAL_ACCESS_TOKEN)")

    # Notion server
    notion_token = os.getenv("NOTION_API_TOKEN") or os.getenv("NOTION_TOKEN")
    if notion_token:
        servers.append(MCPServerConfig(
            name="notion",
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={
                "OPENAPI_MCP_HEADERS": json.dumps({
                    "Authorization": f"Bearer {notion_token}",
                    "Notion-Version": "2022-06-28",
                }),
            },
        ))
        print("MCP: Notion server configured")
    else:
        print("MCP: Notion server not configured (missing NOTION_API_TOKEN)")

    return MCPConfig(servers=servers)


def validate_mcp_config(config: MCPConfig) -> list[str]:
    """
    Validate MCP configuration.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for server in config.servers:
        if not server.name:
            errors.append("Server missing name")
        if not server.command:
            errors.append(f"Server {server.name}: missing command")

    return errors


def build_mcp_tools() -> list[MCPStdioTool]:
    """
    Build MAF MCPStdioTool instances from configuration.

    MCPStdioTool auto-discovers tools, auto-converts to FunctionTool,
    and manages subprocess lifecycle via async with.
    """
    config = load_mcp_config()
    return [
        MCPStdioTool(
            name=s.name,
            command=s.command,
            args=s.args,
            env=s.env,
        )
        for s in config.servers
    ]
