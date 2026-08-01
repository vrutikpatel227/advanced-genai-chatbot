"""
tests/test_llm_client.py

Tests the unified LLM client's configuration validation and error
handling across providers, without making any real network calls.

Note: config.py, utils/llm_client.py, and utils/providers/__init__.py
each hold their own `from config import llm_config` binding (Python's
"import a name" semantics), so tests patch all three references to a
replacement LLMConfig, and clear the cached provider afterward.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from utils import llm_client  # noqa: E402
from utils import providers as providers_pkg  # noqa: E402


def _patch_llm_config(monkeypatch, **overrides):
    new_config = dataclasses.replace(config.llm_config, **overrides)
    monkeypatch.setattr(config, "llm_config", new_config)
    monkeypatch.setattr(llm_client, "llm_config", new_config)
    monkeypatch.setattr(providers_pkg, "llm_config", new_config)
    return new_config


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    """Ensure the cached provider never leaks between tests."""
    llm_client._get_provider.cache_clear()
    yield
    llm_client._get_provider.cache_clear()


def test_is_configured_true_for_groq_with_key(monkeypatch):
    _patch_llm_config(monkeypatch, provider="groq", groq_api_key="fake-groq-key")
    assert llm_client.is_configured() is True


def test_is_configured_false_for_groq_without_key(monkeypatch):
    _patch_llm_config(monkeypatch, provider="groq", groq_api_key="")
    assert llm_client.is_configured() is False


def test_is_configured_true_for_openai_with_key(monkeypatch):
    _patch_llm_config(monkeypatch, provider="openai", openai_api_key="sk-fake-key")
    assert llm_client.is_configured() is True


def test_is_configured_false_for_invalid_provider(monkeypatch):
    _patch_llm_config(monkeypatch, provider="not-a-real-provider")
    assert llm_client.is_configured() is False


def test_validate_configuration_message_for_invalid_provider(monkeypatch):
    _patch_llm_config(monkeypatch, provider="not-a-real-provider")
    is_valid, message = llm_client.validate_configuration()
    assert is_valid is False
    assert "Invalid LLM_PROVIDER" in message


def test_validate_configuration_message_for_missing_key(monkeypatch):
    _patch_llm_config(monkeypatch, provider="groq", groq_api_key="")
    is_valid, message = llm_client.validate_configuration()
    assert is_valid is False
    assert "GROQ_API_KEY" in message


def test_validate_configuration_success_message(monkeypatch):
    _patch_llm_config(monkeypatch, provider="groq", groq_api_key="fake-groq-key")
    is_valid, message = llm_client.validate_configuration()
    assert is_valid is True
    assert "groq" in message.lower()


def test_get_chat_completion_raises_configuration_error_without_key(monkeypatch):
    _patch_llm_config(monkeypatch, provider="groq", groq_api_key="")
    with pytest.raises(llm_client.LLMConfigurationError):
        llm_client.get_chat_completion([llm_client.ChatMessage("user", "hello")])


def test_get_chat_completion_raises_configuration_error_for_invalid_provider(monkeypatch):
    _patch_llm_config(monkeypatch, provider="not-a-real-provider")
    with pytest.raises(llm_client.LLMConfigurationError):
        llm_client.get_chat_completion([llm_client.ChatMessage("user", "hi")])


def test_switching_provider_changes_active_provider(monkeypatch):
    """Confirms provider selection is driven purely by config, with no
    hardcoded provider anywhere in the client."""
    _patch_llm_config(monkeypatch, provider="groq", groq_api_key="fake-groq-key")
    assert llm_client._get_provider().name == "groq"

    llm_client._get_provider.cache_clear()
    _patch_llm_config(monkeypatch, provider="openai", openai_api_key="sk-fake-key")
    assert llm_client._get_provider().name == "openai"
