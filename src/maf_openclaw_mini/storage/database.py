"""
SQLite Database Module

Following Microsoft Agent Framework patterns for state persistence.
Reference: https://learn.microsoft.com/en-us/agent-framework/tutorials/agents/memory

This module provides:
- Session management (conversation threads)
- Message history storage
- User data persistence
"""

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/assistant.db")

# Ensure data directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)


@dataclass
class Session:
    """Session data class."""
    id: str
    user_id: str
    channel_id: str
    thread_ts: Optional[str]
    created_at: int
    last_active: int


@dataclass
class Message:
    """Message data class."""
    id: int
    session_id: str
    role: str
    content: str
    timestamp: int


def init_database() -> None:
    """
    Initialize the SQLite database schema.

    Creates tables for sessions, messages, and approved users.
    Should be called once at application startup.
    """
    with get_connection() as conn:
        conn.executescript("""
            -- Sessions table: tracks conversation threads
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                thread_ts TEXT,
                created_at INTEGER NOT NULL,
                last_active INTEGER NOT NULL
            );

            -- Messages table: conversation history
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Approved users table: for DM access control
            CREATE TABLE IF NOT EXISTS approved_users (
                user_id TEXT PRIMARY KEY,
                approved_by TEXT,
                approved_at INTEGER NOT NULL
            );

            -- Pairing codes table: for DM approval workflow
            CREATE TABLE IF NOT EXISTS pairing_codes (
                code TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(channel_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
        """)

    print(f"Database: Initialized at {DATABASE_PATH}")


@contextmanager
def get_connection():
    """
    Get a database connection with automatic commit and cleanup.

    Usage:
        with get_connection() as conn:
            conn.execute("SELECT * FROM sessions")
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ======================
# SESSION MANAGEMENT
# ======================

def get_or_create_session(
    user_id: str,
    channel_id: str,
    thread_ts: Optional[str] = None,
) -> Session:
    """
    Get or create a session for a user in a channel.

    Following MAF pattern: Sessions are created per user/channel/thread combination.
    This allows tracking conversation context across messages.

    Args:
        user_id: Slack user ID
        channel_id: Slack channel ID
        thread_ts: Thread timestamp (for threaded conversations)

    Returns:
        Session object with session details
    """
    now = int(time.time())

    with get_connection() as conn:
        # Try to find existing session
        if thread_ts:
            row = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND channel_id = ? AND thread_ts = ?",
                (user_id, channel_id, thread_ts),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND channel_id = ? AND thread_ts IS NULL",
                (user_id, channel_id),
            ).fetchone()

        if row:
            # Update last_active timestamp
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (now, row["id"]),
            )
            return Session(
                id=row["id"],
                user_id=row["user_id"],
                channel_id=row["channel_id"],
                thread_ts=row["thread_ts"],
                created_at=row["created_at"],
                last_active=now,
            )

        # Create new session
        session_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO sessions (id, user_id, channel_id, thread_ts, created_at, last_active)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, channel_id, thread_ts, now, now),
        )

        return Session(
            id=session_id,
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            created_at=now,
            last_active=now,
        )


def get_session(session_id: str) -> Optional[Session]:
    """Get a session by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

        if row:
            return Session(
                id=row["id"],
                user_id=row["user_id"],
                channel_id=row["channel_id"],
                thread_ts=row["thread_ts"],
                created_at=row["created_at"],
                last_active=row["last_active"],
            )
        return None


def cleanup_old_sessions(max_age_hours: int = 24) -> int:
    """
    Delete sessions older than max_age_hours.

    Returns:
        Number of sessions deleted
    """
    cutoff = int(time.time()) - (max_age_hours * 3600)

    with get_connection() as conn:
        # Delete messages first (foreign key)
        conn.execute(
            """DELETE FROM messages WHERE session_id IN
               (SELECT id FROM sessions WHERE last_active < ?)""",
            (cutoff,),
        )

        # Delete sessions
        cursor = conn.execute(
            "DELETE FROM sessions WHERE last_active < ?",
            (cutoff,),
        )
        return cursor.rowcount


# ======================
# MESSAGE HISTORY
# ======================

def add_message(session_id: str, role: str, content: str) -> Message:
    """
    Add a message to session history.

    Following MAF pattern: Messages are stored per session for context retrieval.

    Args:
        session_id: Session ID
        role: Message role ('user' or 'assistant')
        content: Message content

    Returns:
        Message object with message details
    """
    now = int(time.time())

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        return Message(
            id=cursor.lastrowid,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=now,
        )


def get_session_messages(
    session_id: str,
    limit: int = 50,
) -> list[Message]:
    """
    Get message history for a session.

    Following MAF pattern: Retrieved messages are used for conversation context.

    Args:
        session_id: Session ID
        limit: Maximum number of messages to retrieve

    Returns:
        List of Message objects (oldest first)
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM messages WHERE session_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()

        # Reverse to get chronological order
        messages = [
            Message(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
            )
            for row in reversed(rows)
        ]

        return messages


def get_recent_messages(
    user_id: str,
    limit: int = 10,
) -> list[Message]:
    """
    Get recent messages for a user across all sessions.

    Useful for building user context.

    Args:
        user_id: Slack user ID
        limit: Maximum number of messages

    Returns:
        List of Message objects (newest first)
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT m.* FROM messages m
               JOIN sessions s ON m.session_id = s.id
               WHERE s.user_id = ?
               ORDER BY m.timestamp DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()

        return [
            Message(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]


# ======================
# USER MANAGEMENT
# ======================

def is_user_approved(user_id: str) -> bool:
    """Check if a user is approved for DM access."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM approved_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row is not None


def approve_user(user_id: str, approved_by: str) -> None:
    """Approve a user for DM access."""
    now = int(time.time())

    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO approved_users (user_id, approved_by, approved_at)
               VALUES (?, ?, ?)""",
            (user_id, approved_by, now),
        )


def generate_pairing_code(user_id: str, expires_in_hours: int = 1) -> str:
    """
    Generate a pairing code for DM approval.

    Returns:
        6-character hex code
    """
    import secrets

    code = secrets.token_hex(3).upper()
    now = int(time.time())
    expires = now + (expires_in_hours * 3600)

    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO pairing_codes (code, user_id, created_at, expires_at)
               VALUES (?, ?, ?, ?)""",
            (code, user_id, now, expires),
        )

    return code


def approve_pairing(code: str, approved_by: str) -> Optional[str]:
    """
    Approve a pairing code.

    Returns:
        User ID if successful, None if code invalid/expired
    """
    now = int(time.time())

    with get_connection() as conn:
        row = conn.execute(
            "SELECT user_id FROM pairing_codes WHERE code = ? AND expires_at > ?",
            (code.upper(), now),
        ).fetchone()

        if not row:
            return None

        user_id = row["user_id"]

        # Approve the user
        approve_user(user_id, approved_by)

        # Delete the pairing code
        conn.execute("DELETE FROM pairing_codes WHERE code = ?", (code.upper(),))

        return user_id


# ======================
# STATISTICS
# ======================

def get_stats() -> dict:
    """Get database statistics."""
    with get_connection() as conn:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM sessions").fetchone()[0]
        approved = conn.execute("SELECT COUNT(*) FROM approved_users").fetchone()[0]

        return {
            "sessions": sessions,
            "messages": messages,
            "unique_users": users,
            "approved_users": approved,
        }
