"""
Error Handling Utilities

Provides structured error handling and recovery for the bot.
"""

from typing import Optional, Callable, Any
from functools import wraps
import traceback


class BotError(Exception):
    """Base exception for bot errors."""

    def __init__(self, message: str, user_message: Optional[str] = None):
        """
        Initialize bot error.

        Args:
            message: Technical error message for logging
            user_message: User-friendly message to display
        """
        super().__init__(message)
        self.user_message = user_message or "Sorry, something went wrong. Please try again."


class ToolError(BotError):
    """Error during tool execution."""

    def __init__(self, tool_name: str, message: str, user_message: Optional[str] = None):
        self.tool_name = tool_name
        super().__init__(
            f"Tool {tool_name} failed: {message}",
            user_message or f"I couldn't complete the {tool_name} action. Please try again.",
        )


class ConfigError(BotError):
    """Configuration error."""

    def __init__(self, config_name: str, message: str):
        super().__init__(
            f"Configuration error for {config_name}: {message}",
            "There's a configuration issue. Please contact the administrator.",
        )


class SlackError(BotError):
    """Slack API error."""

    def __init__(self, operation: str, message: str):
        super().__init__(
            f"Slack {operation} failed: {message}",
            "I couldn't complete the Slack action. Please try again.",
        )


class DatabaseError(BotError):
    """Database operation error."""

    def __init__(self, operation: str, message: str):
        super().__init__(
            f"Database {operation} failed: {message}",
            "I couldn't save or retrieve data. Please try again.",
        )


class RateLimitError(BotError):
    """Rate limit exceeded error."""

    def __init__(self, service: str, retry_after: Optional[int] = None):
        msg = f"Rate limit exceeded for {service}"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(
            msg,
            "I'm being rate limited. Please wait a moment and try again.",
        )
        self.retry_after = retry_after


def safe_execute(
    default_value: Any = None,
    error_message: str = "Operation failed",
    reraise: bool = False,
):
    """
    Decorator for safe execution with error handling.

    Args:
        default_value: Value to return on error
        error_message: Message to log on error
        reraise: Whether to reraise the exception

    Usage:
        @safe_execute(default_value=[], error_message="Failed to fetch data")
        async def fetch_data():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"[Error] {error_message}: {e}")
                if reraise:
                    raise
                return default_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[Error] {error_message}: {e}")
                if reraise:
                    raise
                return default_value

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def format_error_for_user(error: Exception) -> str:
    """
    Format an error for display to the user.

    Args:
        error: The exception

    Returns:
        User-friendly error message
    """
    if isinstance(error, BotError):
        return error.user_message

    # Generic error message
    return "Sorry, I encountered an unexpected error. Please try again."


def format_error_for_log(error: Exception) -> str:
    """
    Format an error for logging.

    Args:
        error: The exception

    Returns:
        Detailed error message with traceback
    """
    error_type = type(error).__name__
    error_msg = str(error)
    tb = traceback.format_exc()

    return f"{error_type}: {error_msg}\n{tb}"


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    exponential_backoff: bool = True,
    exceptions: tuple = (Exception,),
):
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay_seconds: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff
        exceptions: Exception types to catch and retry

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    import asyncio

    last_error = None
    delay = delay_seconds

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_error = e
            if attempt < max_retries:
                print(f"[Retry] Attempt {attempt + 1}/{max_retries} failed: {e}")
                await asyncio.sleep(delay)
                if exponential_backoff:
                    delay *= 2

    raise last_error
