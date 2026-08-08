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


def test_gemini_provider_generate_raises_provider_error_when_package_missing(monkeypatch):
    """Simulates google-genai not being installed (regardless of whether
    it's actually installed in this environment), without making a real
    network call -- deterministic and fast."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "google.genai" or name.startswith("google.genai"):
            raise ImportError("simulated: google-genai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

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


# --- Milestone 5: vision support -------------------------------------------------


def test_openai_supports_vision_for_gpt4o_models():
    provider = OpenAIProvider(api_key="sk-fake", base_url="https://api.openai.com/v1")
    assert provider.supports_vision("gpt-4o") is True
    assert provider.supports_vision("gpt-4o-mini") is True
    assert provider.supports_vision("gpt-4o-2024-08-06") is True  # dated snapshot


def test_openai_default_model_supports_vision():
    provider = OpenAIProvider(api_key="sk-fake", base_url="https://api.openai.com/v1")
    assert provider.supports_vision(provider.default_model) is True


def test_openai_does_not_support_vision_for_text_only_models():
    provider = OpenAIProvider(api_key="sk-fake", base_url="https://api.openai.com/v1")
    assert provider.supports_vision("gpt-3.5-turbo") is False


def test_groq_default_model_does_not_support_vision():
    provider = GroqProvider(api_key="fake-key")
    assert provider.supports_vision(provider.default_model) is False


def test_groq_supports_vision_for_vision_named_models():
    provider = GroqProvider(api_key="fake-key")
    assert provider.supports_vision("llama-3.2-11b-vision-preview") is True


def test_gemini_default_model_supports_vision():
    provider = GeminiProvider(api_key="fake-key")
    assert provider.supports_vision(provider.default_model) is True


def test_gemini_older_models_do_not_support_vision():
    provider = GeminiProvider(api_key="fake-key")
    assert provider.supports_vision("gemini-pro") is False


def test_base_provider_generate_with_image_raises_by_default():
    """A provider that doesn't override generate_with_image() should
    raise VisionNotSupportedError, not crash with a missing-method error."""
    from utils.providers.base_provider import LLMProvider, VisionNotSupportedError

    class _StubProvider(LLMProvider):
        name = "stub"
        default_model = "stub-model"

        def is_configured(self) -> bool:
            return True

        def generate(self, messages, *, model, temperature, max_tokens, timeout) -> str:
            return "stub reply"

    stub = _StubProvider()
    assert stub.supports_vision("anything") is False
    with pytest.raises(VisionNotSupportedError):
        stub.generate_with_image(
            "describe this", b"fake bytes", "image/png",
            model="stub-model", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_openai_provider_generate_with_image_raises_auth_error_when_not_configured():
    provider = OpenAIProvider(api_key="", base_url="https://api.openai.com/v1")
    with pytest.raises(ProviderAuthError):
        provider.generate_with_image(
            "describe this", b"fake bytes", "image/png",
            model="gpt-4o-mini", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_groq_provider_generate_with_image_raises_auth_error_when_not_configured():
    provider = GroqProvider(api_key="")
    with pytest.raises(ProviderAuthError):
        provider.generate_with_image(
            "describe this", b"fake bytes", "image/png",
            model="llama-3.2-11b-vision-preview", temperature=0.3, max_tokens=100, timeout=10,
        )
