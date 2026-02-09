"""
MCP Tool Converter

Converts MCP tool definitions to MAF @tool decorated functions.

Following Microsoft Agent Framework patterns:
- https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools

The key insight is that MAF agents accept tools as a list of functions
decorated with @tool. We dynamically create these functions for each
MCP tool so they can be seamlessly integrated with the agent.
"""

from typing import Annotated, Any, Callable
from pydantic import Field
from agent_framework import tool

from .client import MCPTool, execute_mcp_tool, get_all_mcp_tools


def mcp_tools_to_maf_tools() -> list[Callable]:
    """
    Convert all available MCP tools to MAF-compatible @tool functions.

    This creates wrapper functions that:
    1. Accept arguments as defined by the MCP tool schema
    2. Call the MCP server to execute the tool
    3. Return the result

    Returns:
        List of @tool decorated functions ready for MAF agent
    """
    maf_tools = []

    for server_name, mcp_tool in get_all_mcp_tools():
        # Create unique function name
        func_name = f"{server_name}_{mcp_tool.name}"

        # Create the tool function
        tool_func = _create_mcp_tool_function(server_name, mcp_tool)

        # Add to list
        maf_tools.append(tool_func)

    return maf_tools


def _create_mcp_tool_function(server_name: str, mcp_tool: MCPTool) -> Callable:
    """
    Create a MAF @tool function that wraps an MCP tool.

    The function accepts a JSON string of arguments (for flexibility)
    and calls the MCP server to execute the tool.
    """
    # Build description
    description = mcp_tool.description or f"Execute {mcp_tool.name} on {server_name}"

    # Add parameter hints from schema
    schema = mcp_tool.input_schema
    if schema.get("properties"):
        props = schema["properties"]
        required = schema.get("required", [])

        param_hints = []
        for prop_name, prop_info in props.items():
            prop_desc = prop_info.get("description", "")
            is_required = prop_name in required
            hint = f"- {prop_name}"
            if prop_desc:
                hint += f": {prop_desc}"
            if is_required:
                hint += " (required)"
            param_hints.append(hint)

        if param_hints:
            description += "\n\nParameters:\n" + "\n".join(param_hints)

    # Create the wrapper function
    # We use a factory pattern to capture server_name and tool_name in closure
    async def make_tool_func(s_name: str, t_name: str, desc: str):
        @tool(
            name=f"{s_name}_{t_name}",
            description=desc,
        )
        async def mcp_tool_wrapper(
            arguments: Annotated[str, Field(
                description="JSON string of tool arguments. Example: {\"owner\": \"microsoft\", \"repo\": \"agent-framework\"}"
            )]
        ) -> str:
            """Execute an MCP tool with the given arguments."""
            import json

            try:
                # Parse arguments
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON arguments: {e}"

            try:
                result = await execute_mcp_tool(s_name, t_name, args)

                # Format result
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return json.dumps(result, indent=2)
                else:
                    return str(result)

            except Exception as e:
                return f"Error executing {s_name}/{t_name}: {e}"

        return mcp_tool_wrapper

    # Create the function synchronously by running the async factory
    import asyncio

    # Get or create event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, we need a different approach
        # Just create the function directly

        @tool(
            name=f"{server_name}_{mcp_tool.name}",
            description=description,
        )
        async def mcp_tool_func(
            arguments: Annotated[str, Field(
                description="JSON string of tool arguments"
            )] = "{}"
        ) -> str:
            """Execute an MCP tool with the given arguments."""
            import json

            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON arguments: {e}"

            try:
                result = await execute_mcp_tool(server_name, mcp_tool.name, args)

                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return json.dumps(result, indent=2)
                else:
                    return str(result)

            except Exception as e:
                return f"Error executing {server_name}/{mcp_tool.name}: {e}"

        return mcp_tool_func

    except RuntimeError:
        # No running event loop - create function directly
        @tool(
            name=f"{server_name}_{mcp_tool.name}",
            description=description,
        )
        async def mcp_tool_func(
            arguments: Annotated[str, Field(
                description="JSON string of tool arguments"
            )] = "{}"
        ) -> str:
            """Execute an MCP tool with the given arguments."""
            import json

            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON arguments: {e}"

            try:
                result = await execute_mcp_tool(server_name, mcp_tool.name, args)

                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return json.dumps(result, indent=2)
                else:
                    return str(result)

            except Exception as e:
                return f"Error executing {server_name}/{mcp_tool.name}: {e}"

        return mcp_tool_func


def create_mcp_tool_function(server_name: str, mcp_tool: MCPTool) -> Callable:
    """
    Public API for creating a single MCP tool function.

    This is useful when you want to create tools on-demand
    rather than all at once.
    """
    return _create_mcp_tool_function(server_name, mcp_tool)
