"""
Utilities Module

Common utilities for the MAF-Openclaw-Mini bot.
"""

from .logger import (
    setup_logging,
    get_logger,
    log_tool_call,
    log_agent_response,
    with_logging,
)

from .errors import (
    BotError,
    ToolError,
    ConfigError,
    SlackError,
    DatabaseError,
    RateLimitError,
    safe_execute,
    format_error_for_user,
    format_error_for_log,
    retry_async,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "log_tool_call",
    "log_agent_response",
    "with_logging",
    # Errors
    "BotError",
    "ToolError",
    "ConfigError",
    "SlackError",
    "DatabaseError",
    "RateLimitError",
    "safe_execute",
    "format_error_for_user",
    "format_error_for_log",
    "retry_async",
]
