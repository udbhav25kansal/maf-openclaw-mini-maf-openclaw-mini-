"""
Reminder Tools

MAF @tool decorated functions for setting reminders and scheduled messages.

Following Microsoft Agent Framework patterns:
- https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/function-tools
"""

from typing import Annotated
from pydantic import Field
from agent_framework import tool
from datetime import datetime

from .task_scheduler import (
    get_task_scheduler,
    parse_time_expression,
    parse_cron_expression,
    is_recurring,
)


@tool(
    name="set_reminder",
    description="Set a reminder that will notify the user at the specified time. Supports one-time and recurring reminders."
)
async def set_reminder(
    text: Annotated[str, Field(description="The reminder message/text")],
    time: Annotated[str, Field(description="When to remind. Examples: 'in 5 minutes', 'tomorrow at 9am', 'every monday at 10am', 'at 2pm'")],
    user_id: Annotated[str, Field(description="The Slack user ID to remind")],
    channel_id: Annotated[str, Field(description="The channel ID to post the reminder in")],
) -> str:
    """
    Set a reminder for a user.

    Supports:
    - One-time: "in 5 minutes", "tomorrow at 9am", "at 2pm"
    - Recurring: "every monday at 10am", "every day at 5pm"
    """
    scheduler = get_task_scheduler()

    if not scheduler.is_running():
        return "Error: Task scheduler is not running."

    # Check if recurring
    if is_recurring(time):
        # Parse cron expression
        cron_expr = parse_cron_expression(time)
        if not cron_expr:
            return f"Could not understand the recurring time '{time}'. Try 'every monday at 9am' or 'every day at 5pm'."

        task = await scheduler.schedule_task(
            user_id=user_id,
            channel_id=channel_id,
            description=text,
            cron_expression=cron_expr,
        )

        return f"Recurring reminder set (ID: {task.id}). I'll remind you '{text}' {time}."

    else:
        # Parse one-time
        scheduled_time = parse_time_expression(time)
        if not scheduled_time:
            return f"Could not understand the time '{time}'. Try 'in 5 minutes', 'tomorrow at 9am', or 'at 2pm'."

        task = await scheduler.schedule_task(
            user_id=user_id,
            channel_id=channel_id,
            description=text,
            scheduled_time=scheduled_time,
        )

        time_str = scheduled_time.strftime("%I:%M %p on %A, %B %d")
        return f"Reminder set (ID: {task.id}). I'll remind you '{text}' at {time_str}."


@tool(
    name="list_reminders",
    description="List all active reminders for a user"
)
async def list_reminders(
    user_id: Annotated[str, Field(description="The Slack user ID to list reminders for")]
) -> str:
    """List all active reminders for a user."""
    scheduler = get_task_scheduler()

    if not scheduler.is_running():
        return "Error: Task scheduler is not running."

    tasks = scheduler.get_user_tasks(user_id)

    if not tasks:
        return "You have no active reminders."

    lines = ["Your active reminders:"]
    for task in tasks:
        if task.cron_expression:
            time_str = f"Recurring ({task.cron_expression})"
        elif task.scheduled_time:
            time_str = task.scheduled_time.strftime("%I:%M %p on %b %d")
        else:
            time_str = "Unknown"

        lines.append(f"- *{task.id}*: {task.description} ({time_str})")

    return "\n".join(lines)


@tool(
    name="cancel_reminder",
    description="Cancel a scheduled reminder by its ID"
)
async def cancel_reminder(
    reminder_id: Annotated[int, Field(description="The reminder ID to cancel")]
) -> str:
    """Cancel a scheduled reminder."""
    scheduler = get_task_scheduler()

    if not scheduler.is_running():
        return "Error: Task scheduler is not running."

    task = scheduler.get_task(reminder_id)
    if not task:
        return f"Reminder {reminder_id} not found."

    success = await scheduler.cancel_task(reminder_id)
    if success:
        return f"Reminder {reminder_id} cancelled: '{task.description}'"
    else:
        return f"Could not cancel reminder {reminder_id}."


@tool(
    name="schedule_message",
    description="Schedule a message to be sent to a channel at a specific time"
)
async def schedule_message(
    target: Annotated[str, Field(description="Channel name or user name to send to")],
    message: Annotated[str, Field(description="The message to send")],
    time: Annotated[str, Field(description="When to send. Examples: 'in 30 minutes', 'tomorrow at 9am', 'at 2pm'")],
    channel_id: Annotated[str, Field(description="The channel ID to send the message to")],
) -> str:
    """
    Schedule a message to be sent at a specific time.

    Note: This creates a scheduled task that will post the message
    when the time comes.
    """
    scheduler = get_task_scheduler()

    if not scheduler.is_running():
        return "Error: Task scheduler is not running."

    # Parse time
    scheduled_time = parse_time_expression(time)
    if not scheduled_time:
        return f"Could not understand the time '{time}'. Try 'in 30 minutes', 'tomorrow at 9am', or 'at 2pm'."

    # Schedule as a task
    task = await scheduler.schedule_task(
        user_id="system",  # System-initiated
        channel_id=channel_id,
        description=f"[Scheduled to {target}] {message}",
        scheduled_time=scheduled_time,
    )

    time_str = scheduled_time.strftime("%I:%M %p on %A, %B %d")
    return f"Message scheduled (ID: {task.id}). Will send to {target} at {time_str}."
