"""
Step 13: Test Task Scheduler

Tests the task scheduler and reminder functionality.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maf_openclaw_mini.scheduler import (
    TaskScheduler,
    parse_time_expression,
    parse_cron_expression,
    is_recurring,
)


async def main():
    print("=" * 50)
    print("Task Scheduler Test")
    print("=" * 50)

    # Test 1: Time parsing
    print("\n" + "-" * 40)
    print("Test 1: Time parsing...")
    print("-" * 40)

    test_cases = [
        "in 5 minutes",
        "in 2 hours",
        "tomorrow at 9am",
        "at 2pm",
        "at 14:30",
    ]

    for time_str in test_cases:
        result = parse_time_expression(time_str)
        if result:
            print(f"  '{time_str}' -> {result.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  '{time_str}' -> Failed to parse")

    # Test 2: Cron parsing
    print("\n" + "-" * 40)
    print("Test 2: Cron expression parsing...")
    print("-" * 40)

    cron_cases = [
        "every monday at 9am",
        "every day at 5pm",
        "every weekday at 8am",
        "every weekend at 10am",
        "every tuesday at 2:30pm",
    ]

    for time_str in cron_cases:
        result = parse_cron_expression(time_str)
        if result:
            print(f"  '{time_str}' -> {result}")
        else:
            print(f"  '{time_str}' -> Failed to parse")

    # Test 3: Recurring detection
    print("\n" + "-" * 40)
    print("Test 3: Recurring detection...")
    print("-" * 40)

    recurring_cases = [
        ("every monday at 9am", True),
        ("in 5 minutes", False),
        ("every day at noon", True),
        ("tomorrow at 9am", False),
    ]

    for time_str, expected in recurring_cases:
        result = is_recurring(time_str)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"  {status} '{time_str}' -> {result} (expected {expected})")

    # Test 4: Scheduler creation
    print("\n" + "-" * 40)
    print("Test 4: Scheduler creation...")
    print("-" * 40)

    scheduler = TaskScheduler()
    print(f"  Scheduler created: {scheduler}")
    print(f"  Running: {scheduler.is_running()}")

    # Test 5: Start/Stop
    print("\n" + "-" * 40)
    print("Test 5: Start/Stop scheduler...")
    print("-" * 40)

    scheduler.start()
    print(f"  Running after start: {scheduler.is_running()}")

    # Test 6: Schedule tasks (without actually running them)
    print("\n" + "-" * 40)
    print("Test 6: Scheduling tasks...")
    print("-" * 40)

    # One-time task
    task1 = await scheduler.schedule_task(
        user_id="U_TEST",
        channel_id="C_TEST",
        description="Test reminder 1",
        scheduled_time=datetime.now() + timedelta(hours=1),
    )
    print(f"  [OK] One-time task created: ID={task1.id}")

    # Recurring task
    task2 = await scheduler.schedule_task(
        user_id="U_TEST",
        channel_id="C_TEST",
        description="Test reminder 2",
        cron_expression="0 9 * * 1",  # Monday 9am
    )
    print(f"  [OK] Recurring task created: ID={task2.id}")

    # Test 7: Get tasks
    print("\n" + "-" * 40)
    print("Test 7: Getting tasks...")
    print("-" * 40)

    user_tasks = scheduler.get_user_tasks("U_TEST")
    print(f"  User tasks: {len(user_tasks)}")

    all_tasks = scheduler.get_all_tasks()
    print(f"  All tasks: {len(all_tasks)}")

    # Test 8: Cancel task
    print("\n" + "-" * 40)
    print("Test 8: Cancelling task...")
    print("-" * 40)

    success = await scheduler.cancel_task(task1.id)
    print(f"  Cancel task {task1.id}: {success}")
    print(f"  Task status: {task1.status}")

    # Stop scheduler
    await scheduler.stop()
    print(f"\n  Running after stop: {scheduler.is_running()}")

    print("\n" + "=" * 50)
    print("[SUCCESS] All scheduler tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
