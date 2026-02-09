"""
Scheduler Module

Task scheduling and reminder system following best practices.

This module provides:
- TaskScheduler: Core scheduler using APScheduler
- Reminder tools: MAF @tool functions for agent use
- Time parsing: Natural language to datetime/cron

Usage:
------
```python
from maf_openclaw_mini.scheduler import (
    start_task_scheduler,
    stop_task_scheduler,
    get_task_scheduler,
    set_reminder,
    list_reminders,
    cancel_reminder,
    schedule_message,
)

# Start scheduler
scheduler = start_task_scheduler()

# Create agent with reminder tools
agent = client.as_agent(
    name="Assistant",
    tools=[set_reminder, list_reminders, cancel_reminder],
)

# Stop on shutdown
await stop_task_scheduler()
```
"""

from .task_scheduler import (
    TaskScheduler,
    ScheduledTask,
    TaskStatus,
    get_task_scheduler,
    start_task_scheduler,
    stop_task_scheduler,
    parse_time_expression,
    parse_cron_expression,
    is_recurring,
)

from .reminder_tools import (
    set_reminder,
    list_reminders,
    cancel_reminder,
    schedule_message,
)

__all__ = [
    # Scheduler
    "TaskScheduler",
    "ScheduledTask",
    "TaskStatus",
    "get_task_scheduler",
    "start_task_scheduler",
    "stop_task_scheduler",
    # Time parsing
    "parse_time_expression",
    "parse_cron_expression",
    "is_recurring",
    # Tools
    "set_reminder",
    "list_reminders",
    "cancel_reminder",
    "schedule_message",
]
