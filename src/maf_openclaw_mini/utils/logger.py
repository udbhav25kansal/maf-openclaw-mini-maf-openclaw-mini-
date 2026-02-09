"""
Logging Utilities

Provides structured logging for the MAF-Openclaw-Mini bot.

Following best practices:
- Structured JSON logging for production
- Console logging for development
- Per-module loggers
- OpenTelemetry-ready
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Any, Optional
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

# Log level from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

# Whether to use JSON format
JSON_LOGGING = os.getenv("JSON_LOGGING", "false").lower() == "true"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Console formatter with colors and structure."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Color the level
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Build message
        msg = f"{color}[{timestamp}] [{record.levelname:7}]{reset} [{record.name}] {record.getMessage()}"

        # Add extra fields if present
        if hasattr(record, "extra_fields") and record.extra_fields:
            extras = " ".join(f"{k}={v}" for k, v in record.extra_fields.items())
            msg += f" ({extras})"

        # Add exception if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def setup_logging(level: Optional[str] = None) -> None:
    """
    Set up logging for the application.

    Args:
        level: Log level (debug, info, warning, error). Uses LOG_LEVEL env if not provided.
    """
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Set formatter based on mode
    if JSON_LOGGING:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())

    root_logger.addHandler(handler)

    # Set level for third-party loggers
    logging.getLogger("slack_bolt").setLevel(logging.WARNING)
    logging.getLogger("slack_sdk").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that supports extra fields."""

    def process(self, msg: str, kwargs: dict) -> tuple:
        extra = kwargs.get("extra", {})
        if "extra_fields" not in extra:
            extra["extra_fields"] = {}

        # Merge any extra fields passed to the log call
        if hasattr(self, "extra") and self.extra:
            extra["extra_fields"].update(self.extra)

        kwargs["extra"] = extra
        return msg, kwargs


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    args: dict,
    result: Optional[str] = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """
    Log a tool call with structured data.

    Args:
        logger: Logger instance
        tool_name: Name of the tool called
        args: Tool arguments
        result: Tool result (if successful)
        error: Error message (if failed)
        duration_ms: Execution time in milliseconds
    """
    extra_fields = {
        "tool": tool_name,
        "args": json.dumps(args) if args else "{}",
    }

    if duration_ms is not None:
        extra_fields["duration_ms"] = duration_ms

    if error:
        extra_fields["error"] = error
        logger.error(f"Tool {tool_name} failed", extra={"extra_fields": extra_fields})
    else:
        if result:
            extra_fields["result_length"] = len(result)
        logger.info(f"Tool {tool_name} executed", extra={"extra_fields": extra_fields})


def log_agent_response(
    logger: logging.Logger,
    user_id: str,
    channel_id: str,
    user_message: str,
    response: str,
    duration_ms: Optional[float] = None,
    tools_used: Optional[list[str]] = None,
) -> None:
    """
    Log an agent response with structured data.

    Args:
        logger: Logger instance
        user_id: Slack user ID
        channel_id: Slack channel ID
        user_message: User's input message
        response: Agent's response
        duration_ms: Total processing time
        tools_used: List of tools that were called
    """
    extra_fields = {
        "user_id": user_id,
        "channel_id": channel_id,
        "message_length": len(user_message),
        "response_length": len(response),
    }

    if duration_ms is not None:
        extra_fields["duration_ms"] = duration_ms

    if tools_used:
        extra_fields["tools_used"] = ",".join(tools_used)
        extra_fields["tool_count"] = len(tools_used)

    logger.info("Agent response generated", extra={"extra_fields": extra_fields})


def with_logging(logger: logging.Logger):
    """
    Decorator that logs function entry and exit.

    Usage:
        @with_logging(logger)
        async def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")

            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.exception(f"Error in {func_name}: {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.exception(f"Error in {func_name}: {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
