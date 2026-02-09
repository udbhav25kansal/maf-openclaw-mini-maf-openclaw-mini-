"""
Task Scheduler

Manages scheduled tasks and reminders using APScheduler.

Following best practices:
- Async-compatible scheduling
- Persistent task tracking
- Cron and one-time tasks
- Graceful shutdown

Reference: TypeScript implementation at Openclaw-mini/src/tools/scheduler.ts
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from slack_sdk.web.async_client import AsyncWebClient
from dotenv import load_dotenv

load_dotenv()


class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Scheduled task data class."""
    id: int
    user_id: str
    channel_id: str
    description: str
    scheduled_time: Optional[datetime] = None  # For one-time tasks
    cron_expression: Optional[str] = None  # For recurring tasks
    thread_ts: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    error: Optional[str] = None


class TaskScheduler:
    """
    Task scheduler for reminders and scheduled messages.

    Supports:
    - One-time tasks: Executed at a specific datetime
    - Recurring tasks: Executed based on cron expression

    Usage:
        scheduler = TaskScheduler()
        scheduler.start()

        # One-time task
        task = await scheduler.schedule_task(
            user_id="U123",
            channel_id="C456",
            description="Meeting reminder",
            scheduled_time=datetime.now() + timedelta(hours=1)
        )

        # Recurring task
        task = await scheduler.schedule_task(
            user_id="U123",
            channel_id="C456",
            description="Daily standup",
            cron_expression="0 9 * * 1-5"  # 9am weekdays
        )
    """

    def __init__(self):
        """Initialize the task scheduler."""
        self._scheduler = AsyncIOScheduler()
        self._tasks: dict[int, ScheduledTask] = {}
        self._next_id = 1
        self._web_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
        self._started = False

    def start(self) -> None:
        """Start the scheduler."""
        if self._started:
            return

        self._scheduler.start()
        self._started = True
        print("[Scheduler] Started")

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._started:
            return

        self._scheduler.shutdown(wait=True)
        self._started = False
        print("[Scheduler] Stopped")

    async def schedule_task(
        self,
        user_id: str,
        channel_id: str,
        description: str,
        scheduled_time: Optional[datetime] = None,
        cron_expression: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> ScheduledTask:
        """
        Schedule a new task.

        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID for notification
            description: Task/reminder text
            scheduled_time: For one-time tasks
            cron_expression: For recurring tasks (e.g., "0 9 * * 1-5")
            thread_ts: Optional thread timestamp

        Returns:
            Created ScheduledTask
        """
        task_id = self._next_id
        self._next_id += 1

        task = ScheduledTask(
            id=task_id,
            user_id=user_id,
            channel_id=channel_id,
            description=description,
            scheduled_time=scheduled_time,
            cron_expression=cron_expression,
            thread_ts=thread_ts,
        )

        self._tasks[task_id] = task

        # Schedule the job
        if cron_expression:
            # Recurring task with cron
            trigger = CronTrigger.from_crontab(cron_expression)
            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task_id],
                id=f"task_{task_id}",
                replace_existing=True,
            )
            print(f"[Scheduler] Scheduled recurring task {task_id}: {cron_expression}")

        elif scheduled_time:
            # One-time task
            trigger = DateTrigger(run_date=scheduled_time)
            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task_id],
                id=f"task_{task_id}",
                replace_existing=True,
            )
            print(f"[Scheduler] Scheduled one-time task {task_id}: {scheduled_time}")

        return task

    async def _execute_task(self, task_id: int) -> None:
        """Execute a scheduled task."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now()

        try:
            # Send reminder message
            await self._web_client.chat_postMessage(
                channel=task.channel_id,
                text=f"*Reminder*: {task.description}",
                thread_ts=task.thread_ts,
            )

            # Update status
            if task.cron_expression:
                # Recurring: reset to pending
                task.status = TaskStatus.PENDING
            else:
                # One-time: mark completed
                task.status = TaskStatus.COMPLETED

            print(f"[Scheduler] Executed task {task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            print(f"[Scheduler] Task {task_id} failed: {e}")

    async def cancel_task(self, task_id: int) -> bool:
        """
        Cancel a scheduled task.

        Returns:
            True if cancelled, False if not found
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        try:
            self._scheduler.remove_job(f"task_{task_id}")
        except Exception:
            pass

        task.status = TaskStatus.CANCELLED
        print(f"[Scheduler] Cancelled task {task_id}")
        return True

    def get_task(self, task_id: int) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_user_tasks(self, user_id: str) -> list[ScheduledTask]:
        """Get all tasks for a user."""
        return [
            task for task in self._tasks.values()
            if task.user_id == user_id and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]

    def get_all_tasks(self) -> list[ScheduledTask]:
        """Get all active tasks."""
        return [
            task for task in self._tasks.values()
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._started


# ======================
# TIME PARSING UTILITIES
# ======================

# Day name to cron day number mapping
DAY_MAP = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}


def parse_time_expression(time_str: str) -> Optional[datetime]:
    """
    Parse natural language time expression to datetime.

    Supports:
    - "in 5 minutes"
    - "in 2 hours"
    - "tomorrow at 9am"
    - "at 2pm"
    - "at 14:30"

    Args:
        time_str: Natural language time expression

    Returns:
        Parsed datetime or None if parsing fails
    """
    lower = time_str.lower().strip()
    now = datetime.now()

    # Pattern: "in X minutes/hours/days"
    duration_match = re.match(r"in\s+(\d+)\s+(minute|minutes|min|hour|hours|hr|day|days)", lower)
    if duration_match:
        amount = int(duration_match.group(1))
        unit = duration_match.group(2)

        if "min" in unit:
            return now + timedelta(minutes=amount)
        elif "hour" in unit or "hr" in unit:
            return now + timedelta(hours=amount)
        elif "day" in unit:
            return now + timedelta(days=amount)

    # Pattern: "tomorrow at Xam/pm"
    tomorrow_match = re.match(r"tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
    if tomorrow_match:
        hours = int(tomorrow_match.group(1))
        mins = int(tomorrow_match.group(2) or 0)
        ampm = tomorrow_match.group(3)

        if ampm == "pm" and hours < 12:
            hours += 12
        elif ampm == "am" and hours == 12:
            hours = 0

        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=hours, minute=mins, second=0, microsecond=0)

    # Pattern: "at Xam/pm" or "at HH:MM"
    at_match = re.match(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
    if at_match:
        hours = int(at_match.group(1))
        mins = int(at_match.group(2) or 0)
        ampm = at_match.group(3)

        if ampm == "pm" and hours < 12:
            hours += 12
        elif ampm == "am" and hours == 12:
            hours = 0

        result = now.replace(hour=hours, minute=mins, second=0, microsecond=0)

        # If time is in the past, schedule for tomorrow
        if result <= now:
            result += timedelta(days=1)

        return result

    return None


def parse_cron_expression(time_str: str) -> Optional[str]:
    """
    Parse natural language recurring time to cron expression.

    Supports:
    - "every monday at 9am"
    - "every day at 5pm"
    - "every weekday at 8am"
    - "every weekend at 10am"

    Args:
        time_str: Natural language recurring time

    Returns:
        Cron expression or None if parsing fails
    """
    lower = time_str.lower().strip()

    if "every" not in lower:
        return None

    # Extract time
    time_match = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
    if not time_match:
        return None

    hours = int(time_match.group(1))
    mins = int(time_match.group(2) or 0)
    ampm = time_match.group(3)

    if ampm == "pm" and hours < 12:
        hours += 12
    elif ampm == "am" and hours == 12:
        hours = 0

    # Determine day pattern
    # Check for specific day first (before checking "day")
    day_part = None
    for day_name, day_num in DAY_MAP.items():
        if day_name in lower:
            day_part = str(day_num)
            break

    if day_part is None:
        if "weekday" in lower:
            day_part = "1-5"  # Mon-Fri
        elif "weekend" in lower:
            day_part = "0,6"  # Sun, Sat
        elif "day" in lower:
            day_part = "*"  # Every day
        else:
            day_part = "*"  # Default to every day

    # Build cron: minute hour * * day-of-week
    return f"{mins} {hours} * * {day_part}"


def is_recurring(time_str: str) -> bool:
    """Check if time expression is recurring."""
    return "every" in time_str.lower()


# ======================
# GLOBAL INSTANCE
# ======================

_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler() -> TaskScheduler:
    """Get the global task scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def start_task_scheduler() -> TaskScheduler:
    """Start the global task scheduler."""
    scheduler = get_task_scheduler()
    scheduler.start()
    return scheduler


async def stop_task_scheduler() -> None:
    """Stop the global task scheduler."""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
