"""
utils/storage.py

Thin SQLite data-access layer shared across modules. Centralized here
so every milestone reuses the same connection/schema handling instead
of each rolling its own.

This layer is intentionally milestone-agnostic: it only knows about
"messages" (session_id, role, content, timestamp). Milestone-specific
metadata (e.g. a sentiment score, a citation list, a detected
language) belongs to that milestone's own module and should be added
via an explicit, additive schema migration when that milestone's PRD
is implemented -- not baked in here ahead of time.

All queries are parameterized -- never string-interpolate user input
into SQL.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from config import paths_config
from utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with WAL mode enabled.

    Ensures the connection is always closed, and rolled back on error.
    """
    paths_config.database_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(paths_config.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    try:
        with get_connection() as conn:
            conn.executescript(_SCHEMA)
    except sqlite3.Error as exc:
        logger.error("Failed to initialize database: %s", exc)
        raise


def save_message(session_id: str, role: str, content: str) -> None:
    """Persist a single chat message.

    Raises ValueError for an invalid role or missing session rather
    than silently storing bad data.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid role: {role!r}")
    if not session_id:
        raise ValueError("session_id is required")

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )


def get_session_messages(session_id: str) -> list[sqlite3.Row]:
    """Return all messages for a session, oldest first."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return cursor.fetchall()


def get_message_count() -> int:
    """Total stored messages -- used by the foundation's placeholder dashboard tile."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
        row = cursor.fetchone()
        return int(row["count"]) if row else 0


def get_session_count() -> int:
    """Total distinct chat sessions recorded."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(DISTINCT session_id) as count FROM messages")
        row = cursor.fetchone()
        return int(row["count"]) if row else 0
