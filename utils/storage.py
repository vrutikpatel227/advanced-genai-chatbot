"""
utils/storage.py

Thin SQLite data-access layer shared across modules. Centralized here
so every milestone reuses the same connection/schema handling instead
of each rolling its own.

Base schema: "messages" (session_id, role, content, timestamp).
Milestone-specific metadata is added additively via migration rather
than baked into the base schema ahead of time. Milestone 1 (Sentiment
Analysis) adds two nullable columns -- sentiment_label,
sentiment_confidence -- via _migrate_add_sentiment_columns(), so
existing rows and the base message API are unaffected.

Milestone 2 (Medical Knowledge Assistant) adds a new, separate table
-- "medical_queries" -- rather than columns on "messages", since its
shape (question, answer, retrieved sources) doesn't fit the generic
chat message model. This keeps the base messages table untouched.

All queries are parameterized -- never string-interpolate user input
into SQL.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

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

CREATE TABLE IF NOT EXISTS medical_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_medical_queries_session ON medical_queries(session_id);
CREATE INDEX IF NOT EXISTS idx_medical_queries_created ON medical_queries(created_at);
"""

# Additive columns for Milestone 1 (Sentiment Analysis). Kept as a
# separate migration step -- rather than in _SCHEMA -- so it's clear
# this was added by a specific milestone on top of the base schema.
_SENTIMENT_COLUMNS = {
    "sentiment_label": "TEXT",
    "sentiment_confidence": "REAL",
}


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


def _migrate_add_sentiment_columns(conn: sqlite3.Connection) -> None:
    """Add sentiment columns to an existing messages table if missing.

    Safe to call repeatedly: checks PRAGMA table_info first so it never
    tries to add a column twice (which SQLite would reject).
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    for column, sql_type in _SENTIMENT_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {column} {sql_type}")


def init_db() -> None:
    """Create tables if they don't exist yet, and apply additive
    migrations. Safe to call on every startup."""
    try:
        with get_connection() as conn:
            conn.executescript(_SCHEMA)
            _migrate_add_sentiment_columns(conn)
    except sqlite3.Error as exc:
        logger.error("Failed to initialize database: %s", exc)
        raise


def save_message(
    session_id: str,
    role: str,
    content: str,
    sentiment_label: Optional[str] = None,
    sentiment_confidence: Optional[float] = None,
) -> None:
    """Persist a single chat message, optionally with its detected sentiment.

    sentiment_label/sentiment_confidence are only meaningful for user
    messages (Milestone 1); assistant messages simply omit them.
    Raises ValueError for an invalid role or missing session rather
    than silently storing bad data.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid role: {role!r}")
    if not session_id:
        raise ValueError("session_id is required")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                session_id, role, content, created_at,
                sentiment_label, sentiment_confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                datetime.now(timezone.utc).isoformat(),
                sentiment_label,
                sentiment_confidence,
            ),
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
    """Total stored messages."""
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


def get_sentiment_summary() -> dict[str, int]:
    """Counts of user messages per sentiment label, for the Analytics
    dashboard. Only counts messages that have been sentiment-analyzed."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT sentiment_label, COUNT(*) as count
            FROM messages
            WHERE role = 'user' AND sentiment_label IS NOT NULL
            GROUP BY sentiment_label
            """
        ).fetchall()
    return {row["sentiment_label"]: row["count"] for row in rows}


def get_total_conversations() -> int:
    """Total analyzed user turns -- the "Total Conversations" metric on
    the Analytics dashboard (one user message = one conversation turn)."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM messages WHERE role = 'user' AND sentiment_label IS NOT NULL"
        )
        row = cursor.fetchone()
        return int(row["count"]) if row else 0


# --- Milestone 2: Medical Knowledge Assistant -------------------------------------


def save_medical_query(
    session_id: str,
    question: str,
    answer: str,
    sources: Optional[list[dict]] = None,
) -> None:
    """Persist a medical Q&A exchange: the question, the generated
    answer, and the retrieved sources (stored as JSON, since it's a
    list of {source, question, url, focus, score} dicts, not a single
    scalar column)."""
    if not session_id:
        raise ValueError("session_id is required")
    if not question or not question.strip():
        raise ValueError("question is required")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO medical_queries (session_id, question, answer, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                json.dumps(sources or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_medical_queries(session_id: str) -> list[sqlite3.Row]:
    """Return all medical Q&A exchanges for a session, oldest first."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM medical_queries WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return cursor.fetchall()


def get_medical_query_count() -> int:
    """Total medical questions answered, across all sessions."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) as count FROM medical_queries")
        row = cursor.fetchone()
        return int(row["count"]) if row else 0
