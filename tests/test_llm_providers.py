"""
tests/test_llm_providers.py

Tests for the provider registry (utils/providers) and individual
provider implementations, without making any real network calls.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.providers import (  # noqa: E402
    SUPPORTED_PROVIDERS,
    ProviderAuthError,
    ProviderError,
    build_provider,
)
from utils.providers.gemini_provider import GeminiProvider  # noqa: E402
from utils.providers.groq_provider import GroqProvider  # noqa: E402
from utils.providers.openai_provider import OpenAIProvider  # noqa: E402


def test_supported_providers_list():
    assert set(SUPPORTED_PROVIDERS) == {"openai", "groq", "gemini"}


def test_build_provider_returns_correct_type():
    assert isinstance(build_provider("openai"), OpenAIProvider)
    assert isinstance(build_provider("groq"), GroqProvider)
    assert isinstance(build_provider("gemini"), GeminiProvider)


def test_build_provider_is_case_insensitive():
    assert isinstance(build_provider("OpenAI"), OpenAIProvider)
    assert isinstance(build_provider("GROQ"), GroqProvider)


def test_build_provider_raises_for_unknown_name():
    with pytest.raises(ValueError):
        build_provider("not-a-real-provider")


def test_openai_provider_is_configured():
    assert OpenAIProvider(api_key="", base_url="https://api.openai.com/v1").is_configured() is False
    assert OpenAIProvider(api_key="sk-fake", base_url="https://api.openai.com/v1").is_configured() is True


def test_groq_provider_is_configured():
    assert GroqProvider(api_key="").is_configured() is False
    assert GroqProvider(api_key="fake-key").is_configured() is True


def test_gemini_provider_is_configured():
    assert GeminiProvider(api_key="").is_configured() is False
    assert GeminiProvider(api_key="fake-key").is_configured() is True


def test_openai_provider_generate_raises_auth_error_when_not_configured():
    provider = OpenAIProvider(api_key="", base_url="https://api.openai.com/v1")
    with pytest.raises(ProviderAuthError):
        provider.generate(
            [{"role": "user", "content": "hi"}],
            model="", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_groq_provider_generate_raises_auth_error_when_not_configured():
    provider = GroqProvider(api_key="")
    with pytest.raises(ProviderAuthError):
        provider.generate(
            [{"role": "user", "content": "hi"}],
            model="", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_gemini_provider_generate_raises_auth_error_when_not_configured():
    provider = GeminiProvider(api_key="")
    with pytest.raises(ProviderAuthError):
        provider.generate(
            [{"role": "user", "content": "hi"}],
            model="", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_gemini_provider_generate_raises_provider_error_when_package_missing():
    """google-generativeai is intentionally not installed yet (this is a
    future-ready placeholder per the PRD) -- this exercises the real,
    unmocked 'package not installed' path."""
    provider = GeminiProvider(api_key="fake-key")
    with pytest.raises(ProviderError):
        provider.generate(
            [{"role": "user", "content": "hi"}],
            model="", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_each_provider_has_a_default_model():
    for provider in (
        build_provider("openai"),
        build_provider("groq"),
        build_provider("gemini"),
    ):
        assert isinstance(provider.default_model, str) and len(provider.default_model) > 0
