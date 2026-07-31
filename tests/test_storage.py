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
