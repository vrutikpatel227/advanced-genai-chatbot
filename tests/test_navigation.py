"""
tests/test_navigation.py

Tests the shared page registry in components/navigation.py -- this is
the single source of truth the sidebar and app.py route from, so it's
worth guarding directly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.navigation import PAGES, get_page  # noqa: E402


def test_chat_page_is_implemented():
    chat = get_page("chat")
    assert chat.implemented is True


def test_milestone1_pages_are_implemented():
    for key in ("chat", "analytics", "about_sentiment"):
        page = get_page(key)
        assert page.implemented is True, f"{key} should be implemented after Milestone 1"


def test_milestone2_page_is_implemented():
    medical = get_page("medical")
    assert medical.implemented is True


def test_milestone3_page_is_implemented():
    kb = get_page("knowledge_base")
    assert kb.implemented is True


def test_remaining_milestone_pages_are_not_yet_implemented():
    pending_keys = {
        "research", "multimodal", "multilingual", "memory",
    }
    for key in pending_keys:
        page = get_page(key)
        assert page.implemented is False, f"{key} should not be marked implemented yet"


def test_page_keys_are_unique():
    keys = [p.key for p in PAGES]
    assert len(keys) == len(set(keys))


def test_get_page_raises_for_unknown_key():
    with pytest.raises(KeyError):
        get_page("does-not-exist")
