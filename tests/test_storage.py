"""
tests/test_storage.py

Tests the milestone-agnostic message storage layer. Uses a temporary
sqlite path so tests never touch the real database/app.db.
"""

import dataclasses
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from utils import storage  # noqa: E402


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the sqlite path at a temp file for every test in this module.

    paths_config is a frozen dataclass (by design, so nothing mutates
    config at runtime) -- so tests swap in a whole replacement instance
    via dataclasses.replace() rather than assigning a single field.
    """
    test_db_path = tmp_path / "test_app.db"
    patched_paths = dataclasses.replace(config.paths_config, sqlite_path=test_db_path)
    monkeypatch.setattr(config, "paths_config", patched_paths)
    monkeypatch.setattr(storage, "paths_config", patched_paths)
    storage.init_db()
    yield


def test_init_db_is_idempotent():
    storage.init_db()
    storage.init_db()  # should not raise on repeated calls


def test_save_and_retrieve_message():
    session_id = str(uuid.uuid4())
    storage.save_message(session_id, "user", "Hello there")
    storage.save_message(session_id, "assistant", "Hi! How can I help?")

    messages = storage.get_session_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello there"
    assert messages[1]["role"] == "assistant"


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        storage.save_message(str(uuid.uuid4()), "bot", "invalid role test")


def test_missing_session_id_raises():
    with pytest.raises(ValueError):
        storage.save_message("", "user", "no session id")


def test_message_and_session_counts():
    s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
    storage.save_message(s1, "user", "hi")
    storage.save_message(s1, "assistant", "hello")
    storage.save_message(s2, "user", "hey")

    assert storage.get_message_count() == 3
    assert storage.get_session_count() == 2


def test_sessions_are_isolated():
    s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
    storage.save_message(s1, "user", "message for session one")
    storage.save_message(s2, "user", "message for session two")

    assert len(storage.get_session_messages(s1)) == 1
    assert storage.get_session_messages(s1)[0]["content"] == "message for session one"


# --- Milestone 1: Sentiment Analysis storage additions ---

def test_save_message_with_sentiment_metadata():
    session_id = str(uuid.uuid4())
    storage.save_message(session_id, "user", "I love this!", sentiment_label="positive", sentiment_confidence=0.97)
    storage.save_message(session_id, "assistant", "Glad to hear it!")

    messages = storage.get_session_messages(session_id)
    assert messages[0]["sentiment_label"] == "positive"
    assert messages[0]["sentiment_confidence"] == pytest.approx(0.97)
    # Assistant messages don't carry sentiment metadata.
    assert messages[1]["sentiment_label"] is None


def test_get_sentiment_summary_counts_by_label():
    session_id = str(uuid.uuid4())
    storage.save_message(session_id, "user", "great!", sentiment_label="positive", sentiment_confidence=0.9)
    storage.save_message(session_id, "user", "terrible.", sentiment_label="negative", sentiment_confidence=0.8)
    storage.save_message(session_id, "user", "what time do you open?", sentiment_label="neutral", sentiment_confidence=0.6)
    storage.save_message(session_id, "user", "amazing service!", sentiment_label="positive", sentiment_confidence=0.95)

    summary = storage.get_sentiment_summary()
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["neutral"] == 1


def test_get_total_conversations_counts_analyzed_user_messages():
    session_id = str(uuid.uuid4())
    storage.save_message(session_id, "user", "hi", sentiment_label="neutral", sentiment_confidence=0.5)
    storage.save_message(session_id, "assistant", "hello!")  # not counted: assistant role
    storage.save_message(session_id, "user", "unanalyzed message")  # not counted: no sentiment_label

    assert storage.get_total_conversations() == 1


def test_messages_without_sentiment_are_excluded_from_summary():
    session_id = str(uuid.uuid4())
    storage.save_message(session_id, "user", "message with no sentiment recorded")
    summary = storage.get_sentiment_summary()
    assert summary == {}


# --- Milestone 2: Medical Knowledge Assistant storage additions ---

def test_save_and_retrieve_medical_query():
    session_id = str(uuid.uuid4())
    sources = [{"source": "NIDDK", "question": "What are the symptoms of diabetes?", "url": "", "focus": "Diabetes", "score": 0.8}]
    storage.save_medical_query(session_id, "What are the symptoms of diabetes?", "Thirst and fatigue.", sources)

    queries = storage.get_medical_queries(session_id)
    assert len(queries) == 1
    assert queries[0]["question"] == "What are the symptoms of diabetes?"
    assert queries[0]["answer"] == "Thirst and fatigue."
    assert "NIDDK" in queries[0]["sources_json"]


def test_save_medical_query_requires_session_id():
    with pytest.raises(ValueError):
        storage.save_medical_query("", "A question?", "An answer.")


def test_save_medical_query_requires_question():
    with pytest.raises(ValueError):
        storage.save_medical_query(str(uuid.uuid4()), "", "An answer.")


def test_medical_query_count():
    s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
    storage.save_medical_query(s1, "Q1?", "A1")
    storage.save_medical_query(s1, "Q2?", "A2")
    storage.save_medical_query(s2, "Q3?", "A3")
    assert storage.get_medical_query_count() == 3


def test_medical_queries_do_not_affect_base_messages_table():
    session_id = str(uuid.uuid4())
    storage.save_medical_query(session_id, "Q?", "A")
    # The generic messages table is untouched by medical queries.
    assert storage.get_message_count() == 0
    assert storage.get_medical_query_count() == 1
