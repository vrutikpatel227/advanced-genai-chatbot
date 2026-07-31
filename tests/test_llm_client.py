"""
tests/test_llm_client.py

Tests the LLM client wrapper's error handling without making any real
network calls -- we only exercise the "not configured" path here,
since hitting the real API isn't appropriate for unit tests.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import llm_client  # noqa: E402


def test_is_configured_reflects_api_key(monkeypatch):
    # llm_config is a frozen dataclass -- swap the whole instance rather
    # than assigning a single field.
    monkeypatch.setattr(llm_client, "llm_config", dataclasses.replace(llm_client.llm_config, api_key=""))
    assert llm_client.is_configured() is False

    monkeypatch.setattr(
        llm_client, "llm_config", dataclasses.replace(llm_client.llm_config, api_key="sk-fake-key-for-test")
    )
    assert llm_client.is_configured() is True


def test_get_chat_completion_raises_configuration_error_without_key(monkeypatch):
    monkeypatch.setattr(llm_client, "llm_config", dataclasses.replace(llm_client.llm_config, api_key=""))
    with pytest.raises(llm_client.LLMConfigurationError):
        llm_client.get_chat_completion([llm_client.ChatMessage("user", "hello")])
