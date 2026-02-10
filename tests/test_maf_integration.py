"""
Comprehensive MAF Integration Tests

Tests every module and feature of the MAF-native architecture:
1. Imports and module structure
2. Storage layer (SQLite + MAF ChatMessage/Role types)
3. CompositeContextProvider (session + mem0)
4. MCP config and MCPStdioTool creation
5. AgentManager lifecycle
6. RAG, scheduler, web search, and tools modules
"""

import asyncio
import os
import sys
import traceback

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

# Use a test database to avoid polluting real data
os.environ.setdefault("DATABASE_PATH", "./data/test_assistant.db")

# Track results
passed = 0
failed = 0
errors = []


def report(test_name: str, success: bool, detail: str = ""):
    global passed, failed
    if success:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        errors.append((test_name, detail))
        print(f"  FAIL  {test_name}: {detail}")


# ================================================================
# 1. IMPORTS AND MODULE STRUCTURE
# ================================================================

def test_imports():
    print("\n" + "=" * 60)
    print("1. IMPORTS AND MODULE STRUCTURE")
    print("=" * 60)

    # Core MAF types
    try:
        from agent_framework import ChatAgent, ChatMessage, Role, Context, ContextProvider, tool
        from agent_framework.openai import OpenAIChatClient
        report("agent_framework core imports", True)
    except Exception as e:
        report("agent_framework core imports", False, str(e))

    # MCPStdioTool
    try:
        from agent_framework import MCPStdioTool
        report("MCPStdioTool import", True)
    except Exception as e:
        report("MCPStdioTool import", False, str(e))

    # Mem0Provider (optional)
    try:
        from agent_framework_mem0 import Mem0Provider
        report("Mem0Provider import", True)
    except ImportError:
        report("Mem0Provider import (optional, not installed)", True)
    except Exception as e:
        report("Mem0Provider import", False, str(e))

    # Agent module
    try:
        from maf_openclaw_mini.agent import AgentContext, AgentManager, SYSTEM_PROMPT, build_user_context_instructions
        report("agent module imports", True)
    except Exception as e:
        report("agent module imports", False, str(e))

    # Agent tools
    try:
        from maf_openclaw_mini.agent.tools import (
            set_web_client, send_slack_message, list_slack_channels,
            list_slack_users, get_channel_info, get_current_time, calculate,
        )
        report("agent.tools imports", True)
    except Exception as e:
        report("agent.tools imports", False, str(e))

    # Storage module
    try:
        from maf_openclaw_mini.storage import (
            init_database, get_connection, get_or_create_session, get_session,
            cleanup_old_sessions, Session, add_message, get_session_messages,
            get_recent_messages, Message, is_user_approved, approve_user,
            generate_pairing_code, approve_pairing, get_stats,
            SQLiteChatMessageStore, build_conversation_context,
            SessionContextProvider,
        )
        report("storage module imports", True)
    except Exception as e:
        report("storage module imports", False, str(e))

    # Memory module
    try:
        from maf_openclaw_mini.memory import CompositeContextProvider
        report("memory module imports", True)
    except Exception as e:
        report("memory module imports", False, str(e))

    # MCP module
    try:
        from maf_openclaw_mini.mcp import build_mcp_tools, load_mcp_config, validate_mcp_config, MCPConfig, MCPServerConfig
        report("mcp module imports", True)
    except Exception as e:
        report("mcp module imports", False, str(e))

    # RAG module
    try:
        from maf_openclaw_mini.rag.search_tool import search_slack_history
        from maf_openclaw_mini.rag.vectorstore import init_vectorstore, get_document_count
        report("rag module imports", True)
    except Exception as e:
        report("rag module imports", False, str(e))

    # Scheduler module
    try:
        from maf_openclaw_mini.scheduler import (
            TaskScheduler, ScheduledTask, TaskStatus,
            get_task_scheduler, start_task_scheduler, stop_task_scheduler,
            parse_time_expression, parse_cron_expression, is_recurring,
            set_reminder, list_reminders, cancel_reminder, schedule_message,
        )
        report("scheduler module imports", True)
    except Exception as e:
        report("scheduler module imports", False, str(e))

    # Tools module (web search)
    try:
        from maf_openclaw_mini.tools import web_search, fetch_url
        report("tools module imports", True)
    except Exception as e:
        report("tools module imports", False, str(e))

    # Utils module
    try:
        from maf_openclaw_mini.utils import (
            setup_logging, get_logger, format_error_for_user,
        )
        report("utils module imports", True)
    except Exception as e:
        report("utils module imports", False, str(e))


# ================================================================
# 2. MAF TYPE COMPLIANCE
# ================================================================

def test_maf_type_compliance():
    print("\n" + "=" * 60)
    print("2. MAF TYPE COMPLIANCE")
    print("=" * 60)

    from agent_framework import ChatMessage, Role, ContextProvider, Context

    # ChatMessage creation
    try:
        msg = ChatMessage(role="user", text="Hello")
        assert msg.text == "Hello", f"Expected 'Hello', got '{msg.text}'"
        report("ChatMessage(role='user', text='Hello')", True)
    except Exception as e:
        report("ChatMessage creation", False, str(e))

    # ChatMessage with Role enum
    try:
        msg = ChatMessage(role=Role("assistant"), text="Hi there")
        assert msg.role.value == "assistant", f"Expected 'assistant', got '{msg.role.value}'"
        assert msg.text == "Hi there"
        report("ChatMessage with Role enum", True)
    except Exception as e:
        report("ChatMessage with Role enum", False, str(e))

    # Role.USER / Role.ASSISTANT
    try:
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        report("Role.USER/ASSISTANT values", True)
    except Exception as e:
        report("Role values", False, str(e))

    # Context creation
    try:
        ctx = Context()
        assert ctx.instructions is None or ctx.instructions == ""
        report("Context() default", True)
    except Exception as e:
        report("Context default", False, str(e))

    # Context with instructions
    try:
        ctx = Context(instructions="Test instructions")
        assert ctx.instructions == "Test instructions"
        report("Context(instructions='...')", True)
    except Exception as e:
        report("Context with instructions", False, str(e))

    # SessionContextProvider is-a ContextProvider
    try:
        from maf_openclaw_mini.storage.session_provider import SessionContextProvider
        provider = SessionContextProvider()
        assert isinstance(provider, ContextProvider), "SessionContextProvider must extend ContextProvider"
        report("SessionContextProvider isinstance(ContextProvider)", True)
    except Exception as e:
        report("SessionContextProvider ABC check", False, str(e))

    # CompositeContextProvider is-a ContextProvider
    try:
        from maf_openclaw_mini.memory import CompositeContextProvider
        from maf_openclaw_mini.storage.session_provider import SessionContextProvider
        session_p = SessionContextProvider()
        composite = CompositeContextProvider(session_provider=session_p)
        assert isinstance(composite, ContextProvider), "CompositeContextProvider must extend ContextProvider"
        report("CompositeContextProvider isinstance(ContextProvider)", True)
    except Exception as e:
        report("CompositeContextProvider ABC check", False, str(e))


# ================================================================
# 3. STORAGE LAYER (SQLite + MAF types)
# ================================================================

async def test_storage_layer():
    print("\n" + "=" * 60)
    print("3. STORAGE LAYER (SQLite + MAF Types)")
    print("=" * 60)

    import uuid
    from maf_openclaw_mini.storage import (
        init_database, get_or_create_session, get_session,
        add_message, get_session_messages, get_stats,
    )
    from maf_openclaw_mini.storage.message_store import SQLiteChatMessageStore, build_conversation_context
    from agent_framework import ChatMessage, Role

    # Use unique IDs to avoid stale data from prior runs
    unique_suffix = uuid.uuid4().hex[:8]
    test_user = f"U_TEST_{unique_suffix}"
    test_channel = f"C_TEST_{unique_suffix}"

    # init_database
    try:
        init_database()
        stats = get_stats()
        assert isinstance(stats, dict)
        assert "sessions" in stats
        assert "messages" in stats
        report("init_database + get_stats", True)
    except Exception as e:
        report("init_database", False, str(e))

    # get_or_create_session
    try:
        session = get_or_create_session(user_id=test_user, channel_id=test_channel)
        assert session.id is not None
        assert session.user_id == test_user
        assert session.channel_id == test_channel
        report("get_or_create_session", True)
    except Exception as e:
        report("get_or_create_session", False, str(e))
        return

    # get_session
    try:
        fetched = get_session(session.id)
        assert fetched is not None
        assert fetched.id == session.id
        report("get_session", True)
    except Exception as e:
        report("get_session", False, str(e))

    # add_message
    try:
        msg = add_message(session_id=session.id, role="user", content="Test message")
        assert msg.role == "user"
        assert msg.content == "Test message"
        report("add_message", True)
    except Exception as e:
        report("add_message", False, str(e))

    # add_message (assistant)
    try:
        msg2 = add_message(session_id=session.id, role="assistant", content="Test reply")
        assert msg2.role == "assistant"
        report("add_message (assistant)", True)
    except Exception as e:
        report("add_message (assistant)", False, str(e))

    # get_session_messages
    try:
        messages = get_session_messages(session_id=session.id)
        assert len(messages) >= 2, f"Expected >= 2 messages, got {len(messages)}: {[m.content for m in messages]}"
        assert messages[-2].content == "Test message", f"Expected 'Test message', got '{messages[-2].content}'"
        assert messages[-1].content == "Test reply", f"Expected 'Test reply', got '{messages[-1].content}'"
        report("get_session_messages", True)
    except Exception as e:
        report("get_session_messages", False, str(e))

    # SQLiteChatMessageStore returns MAF ChatMessage
    try:
        store = SQLiteChatMessageStore(session_id=session.id)
        chat_messages = await store.list_messages()
        assert len(chat_messages) >= 2, f"Expected >= 2 messages, got {len(chat_messages)}"
        assert isinstance(chat_messages[0], ChatMessage), f"Expected ChatMessage, got {type(chat_messages[0])}"
        # Verify role is MAF Role
        assert chat_messages[-2].role == Role.USER or chat_messages[-2].role.value == "user", f"Expected user role, got {chat_messages[-2].role}"
        assert chat_messages[-2].text == "Test message", f"Expected 'Test message', got '{chat_messages[-2].text}'"
        report("SQLiteChatMessageStore.list_messages() -> MAF ChatMessage", True)
    except Exception as e:
        report("SQLiteChatMessageStore MAF types", False, str(e))

    # SQLiteChatMessageStore.add_messages with MAF ChatMessage
    try:
        maf_msg = ChatMessage(role="user", text="MAF test message")
        await store.add_messages([maf_msg])
        updated = await store.list_messages()
        assert updated[-1].text == "MAF test message", f"Expected 'MAF test message', got '{updated[-1].text}'"
        report("SQLiteChatMessageStore.add_messages(MAF ChatMessage)", True)
    except Exception as e:
        report("SQLiteChatMessageStore.add_messages", False, str(e))

    # build_conversation_context
    try:
        context_str = build_conversation_context(chat_messages)
        assert "Recent Conversation History" in context_str
        assert "User:" in context_str or "Assistant:" in context_str
        report("build_conversation_context", True)
    except Exception as e:
        report("build_conversation_context", False, str(e))


# ================================================================
# 4. SESSION CONTEXT PROVIDER (asyncio.to_thread)
# ================================================================

async def test_session_context_provider():
    print("\n" + "=" * 60)
    print("4. SESSION CONTEXT PROVIDER (ContextProvider ABC)")
    print("=" * 60)

    from maf_openclaw_mini.storage.session_provider import SessionContextProvider
    from maf_openclaw_mini.storage import init_database, add_message, get_or_create_session
    from agent_framework import ChatMessage, Context, ContextProvider

    init_database()

    provider = SessionContextProvider()

    # Verify ABC compliance
    try:
        assert isinstance(provider, ContextProvider)
        assert hasattr(provider, 'invoking')
        assert hasattr(provider, 'invoked')
        assert hasattr(provider, 'thread_created')
        report("SessionContextProvider ABC methods present", True)
    except Exception as e:
        report("SessionContextProvider ABC methods", False, str(e))

    # invoking() with user_id/channel_id (session auto-created)
    try:
        msg = ChatMessage(role="user", text="Hello provider")
        ctx = await provider.invoking(msg, user_id="U_PROV_TEST", channel_id="C_PROV_TEST")
        assert isinstance(ctx, Context), f"Expected Context, got {type(ctx)}"
        report("invoking() returns Context", True)
    except Exception as e:
        report("invoking() returns Context", False, str(e))

    # invoking() without kwargs returns empty Context
    try:
        msg = ChatMessage(role="user", text="No context")
        ctx = await provider.invoking(msg)
        assert isinstance(ctx, Context)
        report("invoking() without kwargs -> empty Context", True)
    except Exception as e:
        report("invoking() without kwargs", False, str(e))

    # invoked() persists messages
    try:
        req = ChatMessage(role="user", text="Provider test request")
        resp = ChatMessage(role="assistant", text="Provider test response")
        await provider.invoked(req, resp, user_id="U_PROV_TEST", channel_id="C_PROV_TEST")
        report("invoked() persists without error", True)
    except Exception as e:
        report("invoked()", False, str(e))

    # invoking() again should include history
    try:
        msg2 = ChatMessage(role="user", text="Follow-up")
        ctx2 = await provider.invoking(msg2, user_id="U_PROV_TEST", channel_id="C_PROV_TEST")
        assert ctx2.instructions is not None, "Expected instructions with history"
        assert "Provider test request" in ctx2.instructions or "Recent Conversation History" in ctx2.instructions
        report("invoking() includes conversation history", True)
    except Exception as e:
        report("invoking() with history", False, str(e))


# ================================================================
# 5. COMPOSITE CONTEXT PROVIDER (Memory System)
# ================================================================

async def test_composite_context_provider():
    print("\n" + "=" * 60)
    print("5. COMPOSITE CONTEXT PROVIDER (Memory System)")
    print("=" * 60)

    from maf_openclaw_mini.memory import CompositeContextProvider
    from maf_openclaw_mini.storage.session_provider import SessionContextProvider
    from maf_openclaw_mini.storage import init_database
    from agent_framework import ChatMessage, Context, ContextProvider

    init_database()

    session_provider = SessionContextProvider()
    provider = CompositeContextProvider(session_provider=session_provider)

    # ABC compliance
    try:
        assert isinstance(provider, ContextProvider)
        report("CompositeContextProvider isinstance(ContextProvider)", True)
    except Exception as e:
        report("CompositeContextProvider ABC", False, str(e))

    # is_mem0_enabled property
    try:
        enabled = provider.is_mem0_enabled
        assert isinstance(enabled, bool)
        report(f"is_mem0_enabled = {enabled}", True)
    except Exception as e:
        report("is_mem0_enabled", False, str(e))

    # invoking() with user context injection
    try:
        msg = ChatMessage(role="user", text="Test composite")
        ctx = await provider.invoking(msg, user_id="U_COMP_TEST", channel_id="C_COMP_TEST")
        assert isinstance(ctx, Context)
        # Should contain user context instructions
        assert ctx.instructions is not None, "Expected user context instructions"
        assert "U_COMP_TEST" in ctx.instructions, f"Expected user_id in instructions"
        assert "C_COMP_TEST" in ctx.instructions, f"Expected channel_id in instructions"
        report("invoking() injects user context", True)
    except Exception as e:
        report("invoking() user context injection", False, str(e))

    # invoked() delegates to session provider
    try:
        req = ChatMessage(role="user", text="Composite test request")
        resp = ChatMessage(role="assistant", text="Composite test response")
        await provider.invoked(req, resp, user_id="U_COMP_TEST", channel_id="C_COMP_TEST")
        report("invoked() delegates without error", True)
    except Exception as e:
        report("invoked() delegation", False, str(e))

    # invoking() again should include both user context and history
    try:
        msg2 = ChatMessage(role="user", text="Follow-up composite")
        ctx2 = await provider.invoking(msg2, user_id="U_COMP_TEST", channel_id="C_COMP_TEST")
        assert ctx2.instructions is not None
        # Should have user context
        assert "U_COMP_TEST" in ctx2.instructions
        # Should have conversation history
        assert "Composite test request" in ctx2.instructions or "Recent Conversation History" in ctx2.instructions
        report("invoking() merges user context + history", True)
    except Exception as e:
        report("invoking() merged context", False, str(e))

    # thread_created() delegates
    try:
        await provider.thread_created()
        report("thread_created() delegates without error", True)
    except Exception as e:
        report("thread_created()", False, str(e))


# ================================================================
# 6. MCP CONFIG AND MCPStdioTool
# ================================================================

def test_mcp_config():
    print("\n" + "=" * 60)
    print("6. MCP CONFIG AND MCPStdioTool")
    print("=" * 60)

    from maf_openclaw_mini.mcp import build_mcp_tools, load_mcp_config, validate_mcp_config, MCPConfig, MCPServerConfig

    # load_mcp_config
    try:
        config = load_mcp_config()
        assert isinstance(config, MCPConfig)
        assert isinstance(config.servers, list)
        report(f"load_mcp_config (found {len(config.servers)} servers)", True)
    except Exception as e:
        report("load_mcp_config", False, str(e))

    # validate_mcp_config
    try:
        errors_list = validate_mcp_config(config)
        assert isinstance(errors_list, list)
        report(f"validate_mcp_config (errors: {len(errors_list)})", True)
    except Exception as e:
        report("validate_mcp_config", False, str(e))

    # build_mcp_tools returns list of MCPStdioTool
    try:
        from agent_framework import MCPStdioTool
        tools = build_mcp_tools()
        assert isinstance(tools, list)
        for t in tools:
            assert isinstance(t, MCPStdioTool), f"Expected MCPStdioTool, got {type(t)}"
        report(f"build_mcp_tools -> {len(tools)} MCPStdioTool instances", True)
    except Exception as e:
        report("build_mcp_tools", False, str(e))

    # MCPServerConfig dataclass
    try:
        server = MCPServerConfig(name="test", command="echo", args=["hello"], env={"KEY": "val"})
        assert server.name == "test"
        assert server.command == "echo"
        report("MCPServerConfig dataclass", True)
    except Exception as e:
        report("MCPServerConfig", False, str(e))


# ================================================================
# 7. AGENT MODULE
# ================================================================

def test_agent_module():
    print("\n" + "=" * 60)
    print("7. AGENT MODULE (Context, Prompts, Tools)")
    print("=" * 60)

    # AgentContext
    try:
        from maf_openclaw_mini.agent import AgentContext
        ctx = AgentContext(user_id="U123", channel_id="C456", thread_ts="123.456")
        assert ctx.user_id == "U123"
        assert ctx.channel_id == "C456"
        assert ctx.thread_ts == "123.456"
        assert ctx.session_id is None
        report("AgentContext dataclass", True)
    except Exception as e:
        report("AgentContext", False, str(e))

    # SYSTEM_PROMPT
    try:
        from maf_openclaw_mini.agent import SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100
        assert "search_slack_history" in SYSTEM_PROMPT
        assert "get_current_time" in SYSTEM_PROMPT
        assert "set_reminder" in SYSTEM_PROMPT
        assert "web_search" in SYSTEM_PROMPT
        report("SYSTEM_PROMPT contains all tool references", True)
    except Exception as e:
        report("SYSTEM_PROMPT", False, str(e))

    # build_user_context_instructions
    try:
        from maf_openclaw_mini.agent import build_user_context_instructions
        instr = build_user_context_instructions("U_TEST", "C_TEST")
        assert "U_TEST" in instr
        assert "C_TEST" in instr
        assert "user_id" in instr
        assert "channel_id" in instr
        report("build_user_context_instructions", True)
    except Exception as e:
        report("build_user_context_instructions", False, str(e))

    # Tools are @tool decorated
    try:
        from maf_openclaw_mini.agent.tools import (
            send_slack_message, list_slack_channels, list_slack_users,
            get_channel_info, get_current_time, calculate,
        )
        # get_current_time should work without Slack client
        result = get_current_time()
        assert "Current time:" in result
        report("get_current_time tool works", True)
    except Exception as e:
        report("get_current_time", False, str(e))

    # calculate tool
    try:
        from maf_openclaw_mini.agent.tools import calculate
        result = calculate("2 + 2")
        assert "4" in result
        report("calculate('2 + 2') = 4", True)
    except Exception as e:
        report("calculate tool", False, str(e))


# ================================================================
# 8. RAG MODULE
# ================================================================

async def test_rag_module():
    print("\n" + "=" * 60)
    print("8. RAG MODULE")
    print("=" * 60)

    # Init vectorstore
    try:
        from maf_openclaw_mini.rag.vectorstore import init_vectorstore, get_document_count
        collection = init_vectorstore()
        assert collection is not None
        report("init_vectorstore", True)
    except Exception as e:
        report("init_vectorstore", False, str(e))

    # get_document_count
    try:
        count = await get_document_count()
        assert isinstance(count, int)
        report(f"get_document_count = {count}", True)
    except Exception as e:
        report("get_document_count", False, str(e))

    # search_slack_history is a tool
    try:
        from maf_openclaw_mini.rag.search_tool import search_slack_history
        # Should be callable
        assert callable(search_slack_history)
        report("search_slack_history is callable @tool", True)
    except Exception as e:
        report("search_slack_history", False, str(e))

    # format_search_result
    try:
        from maf_openclaw_mini.rag.search_tool import format_search_result
        result = format_search_result({
            "text": "Test message",
            "metadata": {"channel_name": "general", "user_name": "test"},
            "score": 0.85,
        })
        assert "#general" in result
        assert "85%" in result
        report("format_search_result", True)
    except Exception as e:
        report("format_search_result", False, str(e))


# ================================================================
# 9. SCHEDULER MODULE
# ================================================================

async def test_scheduler_module():
    print("\n" + "=" * 60)
    print("9. SCHEDULER MODULE")
    print("=" * 60)

    from maf_openclaw_mini.scheduler import (
        parse_time_expression, parse_cron_expression, is_recurring,
        TaskScheduler, TaskStatus,
    )
    from datetime import datetime, timedelta

    # parse_time_expression: "in 5 minutes"
    try:
        result = parse_time_expression("in 5 minutes")
        assert result is not None
        assert result > datetime.now()
        assert result < datetime.now() + timedelta(minutes=6)
        report("parse_time_expression('in 5 minutes')", True)
    except Exception as e:
        report("parse_time_expression in minutes", False, str(e))

    # parse_time_expression: "in 2 hours"
    try:
        result = parse_time_expression("in 2 hours")
        assert result is not None
        assert result > datetime.now() + timedelta(hours=1)
        report("parse_time_expression('in 2 hours')", True)
    except Exception as e:
        report("parse_time_expression in hours", False, str(e))

    # parse_time_expression: "tomorrow at 9am"
    try:
        result = parse_time_expression("tomorrow at 9am")
        assert result is not None
        assert result.hour == 9
        report("parse_time_expression('tomorrow at 9am')", True)
    except Exception as e:
        report("parse_time_expression tomorrow", False, str(e))

    # parse_cron_expression: "every monday at 10am"
    try:
        cron = parse_cron_expression("every monday at 10am")
        assert cron is not None
        assert "10" in cron
        report(f"parse_cron_expression('every monday at 10am') = '{cron}'", True)
    except Exception as e:
        report("parse_cron_expression", False, str(e))

    # is_recurring
    try:
        assert is_recurring("every day at 5pm") is True
        assert is_recurring("in 5 minutes") is False
        report("is_recurring()", True)
    except Exception as e:
        report("is_recurring", False, str(e))

    # TaskScheduler start/stop
    try:
        scheduler = TaskScheduler()
        scheduler.start()
        assert scheduler.is_running() is True

        # Schedule a one-time task
        task = await scheduler.schedule_task(
            user_id="U_SCHED_TEST",
            channel_id="C_SCHED_TEST",
            description="Test reminder",
            scheduled_time=datetime.now() + timedelta(hours=1),
        )
        assert task.id is not None
        assert task.status == TaskStatus.PENDING

        # Get user tasks
        tasks = scheduler.get_user_tasks("U_SCHED_TEST")
        assert len(tasks) >= 1

        # Cancel task
        cancelled = await scheduler.cancel_task(task.id)
        assert cancelled is True
        assert scheduler.get_task(task.id).status == TaskStatus.CANCELLED

        await scheduler.stop()
        assert scheduler.is_running() is False
        report("TaskScheduler full lifecycle (start/schedule/cancel/stop)", True)
    except Exception as e:
        report("TaskScheduler lifecycle", False, str(e))

    # Reminder tools are callable
    try:
        from maf_openclaw_mini.scheduler import set_reminder, list_reminders, cancel_reminder
        assert callable(set_reminder)
        assert callable(list_reminders)
        assert callable(cancel_reminder)
        report("Reminder tools are callable @tool functions", True)
    except Exception as e:
        report("Reminder tools", False, str(e))


# ================================================================
# 10. WEB SEARCH AND TOOLS MODULE
# ================================================================

def test_tools_module():
    print("\n" + "=" * 60)
    print("10. WEB SEARCH AND TOOLS MODULE")
    print("=" * 60)

    # web_search is a tool
    try:
        from maf_openclaw_mini.tools import web_search, fetch_url
        assert callable(web_search)
        assert callable(fetch_url)
        report("web_search and fetch_url are callable @tool functions", True)
    except Exception as e:
        report("web_search/fetch_url", False, str(e))


# ================================================================
# 11. AGENT MANAGER (without live API keys)
# ================================================================

async def test_agent_manager_structure():
    print("\n" + "=" * 60)
    print("11. AGENT MANAGER STRUCTURE")
    print("=" * 60)

    from maf_openclaw_mini.agent.core import AgentManager

    # Constructor
    try:
        manager = AgentManager()
        assert manager._agent is None
        assert manager._exit_stack is None
        assert manager._context_provider is None
        report("AgentManager() constructor", True)
    except Exception as e:
        report("AgentManager constructor", False, str(e))

    # is_mem0_enabled before init
    try:
        assert manager.is_mem0_enabled is False
        report("is_mem0_enabled before init = False", True)
    except Exception as e:
        report("is_mem0_enabled before init", False, str(e))

    # run() before init raises
    try:
        from maf_openclaw_mini.agent import AgentContext
        ctx = AgentContext(user_id="U1", channel_id="C1")
        await manager.run("test", ctx)
        report("run() before init should raise", False, "Did not raise RuntimeError")
    except RuntimeError as e:
        assert "not initialized" in str(e).lower()
        report("run() before init raises RuntimeError", True)
    except Exception as e:
        report("run() before init", False, str(e))

    # shutdown() before init is safe
    try:
        await manager.shutdown()
        report("shutdown() before init is no-op", True)
    except Exception as e:
        report("shutdown() before init", False, str(e))


# ================================================================
# 12. UTILS MODULE
# ================================================================

def test_utils_module():
    print("\n" + "=" * 60)
    print("12. UTILS MODULE")
    print("=" * 60)

    try:
        from maf_openclaw_mini.utils import format_error_for_user
        result = format_error_for_user(Exception("Test error"))
        assert isinstance(result, str)
        assert len(result) > 0
        report("format_error_for_user", True)
    except Exception as e:
        report("format_error_for_user", False, str(e))

    try:
        from maf_openclaw_mini.utils import setup_logging, get_logger
        setup_logging()
        logger = get_logger("test")
        assert logger is not None
        report("setup_logging + get_logger", True)
    except Exception as e:
        report("setup_logging", False, str(e))


# ================================================================
# 13. SLACK BOT MAIN MODULE IMPORTS
# ================================================================

def test_slack_bot_imports():
    print("\n" + "=" * 60)
    print("13. SLACK BOT MAIN MODULE (import check)")
    print("=" * 60)

    # Verify the main bot can import all its dependencies
    try:
        from maf_openclaw_mini.agent import AgentManager, AgentContext
        from maf_openclaw_mini.agent.tools import set_web_client
        from maf_openclaw_mini.storage import init_database, get_stats
        from maf_openclaw_mini.rag.vectorstore import init_vectorstore, get_document_count
        from maf_openclaw_mini.rag.indexer import start_background_indexer, stop_background_indexer
        from maf_openclaw_mini.scheduler import start_task_scheduler, stop_task_scheduler
        from maf_openclaw_mini.utils import setup_logging, get_logger, format_error_for_user
        report("All slack_bot_with_tools.py imports resolve", True)
    except Exception as e:
        report("slack_bot imports", False, str(e))


# ================================================================
# 14. NO CIRCULAR IMPORTS
# ================================================================

def test_no_circular_imports():
    print("\n" + "=" * 60)
    print("14. NO CIRCULAR IMPORTS")
    print("=" * 60)

    # If we got here without ImportError, all modules loaded fine.
    # Do an explicit fresh import chain test
    modules_to_check = [
        "maf_openclaw_mini.agent",
        "maf_openclaw_mini.agent.core",
        "maf_openclaw_mini.agent.tools",
        "maf_openclaw_mini.agent.prompts",
        "maf_openclaw_mini.agent.context",
        "maf_openclaw_mini.storage",
        "maf_openclaw_mini.storage.session_provider",
        "maf_openclaw_mini.storage.message_store",
        "maf_openclaw_mini.storage.database",
        "maf_openclaw_mini.memory",
        "maf_openclaw_mini.memory.composite_provider",
        "maf_openclaw_mini.mcp",
        "maf_openclaw_mini.mcp.config",
        "maf_openclaw_mini.rag.search_tool",
        "maf_openclaw_mini.rag.vectorstore",
        "maf_openclaw_mini.scheduler",
        "maf_openclaw_mini.scheduler.task_scheduler",
        "maf_openclaw_mini.scheduler.reminder_tools",
        "maf_openclaw_mini.tools",
        "maf_openclaw_mini.tools.web_search",
        "maf_openclaw_mini.utils",
    ]

    all_good = True
    for mod_name in modules_to_check:
        try:
            __import__(mod_name)
        except ImportError as e:
            report(f"Import {mod_name}", False, str(e))
            all_good = False

    if all_good:
        report(f"All {len(modules_to_check)} modules import without circular dependency errors", True)


# ================================================================
# MAIN
# ================================================================

async def main():
    print("=" * 60)
    print("MAF-OPENCLAW-MINI: COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 60)

    # Sync tests
    test_imports()
    test_maf_type_compliance()
    test_mcp_config()
    test_agent_module()
    test_tools_module()
    test_utils_module()
    test_slack_bot_imports()
    test_no_circular_imports()

    # Async tests
    await test_storage_layer()
    await test_session_context_provider()
    await test_composite_context_provider()
    await test_rag_module()
    await test_scheduler_module()
    await test_agent_manager_structure()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    total = passed + failed
    print(f"Total:  {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if errors:
        print(f"\nFailed tests:")
        for name, detail in errors:
            print(f"  FAIL  {name}: {detail}")

    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
