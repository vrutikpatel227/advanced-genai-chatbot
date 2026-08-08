"""
tests/test_gemini_provider.py

Unit tests for the production Gemini provider implementation
(bug-fix: replaced the previous placeholder). All SDK calls are
mocked -- this sandbox has no network access to Google's Gemini API
endpoint, so these tests verify request construction and error mapping
without making real API calls. See docs/daily_report_gemini_fix.md for
notes on what should additionally be verified against the live API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.providers.base_provider import (  # noqa: E402
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    VisionNotSupportedError,
)
from utils.providers.gemini_provider import GeminiProvider  # noqa: E402


def _mock_client_with_response(response_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# --- configuration / capability checks ------------------------------------------


def test_is_configured():
    assert GeminiProvider(api_key="").is_configured() is False
    assert GeminiProvider(api_key="fake-key").is_configured() is True


def test_supports_vision_for_1_5_and_later():
    provider = GeminiProvider(api_key="fake-key")
    assert provider.supports_vision("gemini-1.5-flash") is True
    assert provider.supports_vision("gemini-1.5-pro") is True
    assert provider.supports_vision("gemini-2.0-flash") is True


def test_does_not_support_vision_for_legacy_models():
    provider = GeminiProvider(api_key="fake-key")
    assert provider.supports_vision("gemini-pro") is False


def test_generate_raises_auth_error_when_not_configured():
    provider = GeminiProvider(api_key="")
    with pytest.raises(ProviderAuthError):
        provider.generate(
            [{"role": "user", "content": "hi"}],
            model="", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_generate_with_image_raises_auth_error_when_not_configured():
    provider = GeminiProvider(api_key="")
    with pytest.raises(ProviderAuthError):
        provider.generate_with_image(
            "describe this", b"fake image bytes", "image/png",
            model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
        )


def test_generate_with_image_raises_vision_not_supported_for_legacy_model():
    provider = GeminiProvider(api_key="fake-key")
    with pytest.raises(VisionNotSupportedError):
        provider.generate_with_image(
            "describe this", b"fake image bytes", "image/png",
            model="gemini-pro", temperature=0.3, max_tokens=100, timeout=10,
        )


# --- generate(): request construction (mocked client) ----------------------------


def test_generate_returns_response_text():
    provider = GeminiProvider(api_key="fake-key")
    mock_client = _mock_client_with_response("Hello there!")

    with patch.object(provider, "_get_client_and_types") as mock_get_client:
        from google.genai import types
        mock_get_client.return_value = (mock_client, types)

        result = provider.generate(
            [{"role": "user", "content": "Hi"}],
            model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
        )

    assert result == "Hello there!"
    mock_client.models.generate_content.assert_called_once()


def test_generate_separates_system_instruction_from_conversation():
    provider = GeminiProvider(api_key="fake-key")
    mock_client = _mock_client_with_response("ok")

    with patch.object(provider, "_get_client_and_types") as mock_get_client:
        from google.genai import types
        mock_get_client.return_value = (mock_client, types)

        provider.generate(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
            model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
        )

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == "Be concise."
    assert len(call_kwargs["contents"]) == 1


def test_generate_maps_assistant_role_to_model_role():
    provider = GeminiProvider(api_key="fake-key")
    mock_client = _mock_client_with_response("ok")

    with patch.object(provider, "_get_client_and_types") as mock_get_client:
        from google.genai import types
        mock_get_client.return_value = (mock_client, types)

        provider.generate(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
            model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
        )

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    roles = [c.role for c in call_kwargs["contents"]]
    assert roles == ["user", "model", "user"]


def test_generate_raises_provider_error_for_no_conversation_messages():
    provider = GeminiProvider(api_key="fake-key")
    mock_client = _mock_client_with_response("ok")

    with patch.object(provider, "_get_client_and_types") as mock_get_client:
        from google.genai import types
        mock_get_client.return_value = (mock_client, types)

        with pytest.raises(ProviderError):
            provider.generate(
                [{"role": "system", "content": "Be concise."}],
                model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
            )


# --- generate_with_image(): request construction (mocked client) -----------------


def test_generate_with_image_sends_text_and_image_parts():
    provider = GeminiProvider(api_key="fake-key")
    mock_client = _mock_client_with_response("A red square.")

    with patch.object(provider, "_get_client_and_types") as mock_get_client:
        from google.genai import types
        mock_get_client.return_value = (mock_client, types)

        result = provider.generate_with_image(
            "Describe this image.", b"fake-image-bytes", "image/png",
            model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
        )

    assert result == "A red square."
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    parts = call_kwargs["contents"][0].parts
    assert len(parts) == 2


# --- error mapping ------------------------------------------------------------------


def test_map_error_translates_401_to_auth_error():
    from google.genai import errors

    provider = GeminiProvider(api_key="fake-key")
    fake_error = MagicMock(spec=errors.ClientError)
    fake_error.code = 401
    mapped = provider._map_error(fake_error)
    assert isinstance(mapped, ProviderAuthError)


def test_map_error_translates_429_to_rate_limit_error():
    from google.genai import errors

    provider = GeminiProvider(api_key="fake-key")
    fake_error = MagicMock(spec=errors.ClientError)
    fake_error.code = 429
    mapped = provider._map_error(fake_error)
    assert isinstance(mapped, ProviderRateLimitError)


def test_map_error_translates_500_to_connection_error():
    from google.genai import errors

    provider = GeminiProvider(api_key="fake-key")
    fake_error = MagicMock(spec=errors.ServerError)
    fake_error.code = 503
    mapped = provider._map_error(fake_error)
    assert isinstance(mapped, ProviderConnectionError)


def test_map_error_translates_httpx_timeout():
    import httpx

    provider = GeminiProvider(api_key="fake-key")
    mapped = provider._map_error(httpx.TimeoutException("timed out"))
    assert isinstance(mapped, ProviderTimeoutError)


def test_map_error_translates_httpx_connect_error():
    import httpx

    provider = GeminiProvider(api_key="fake-key")
    mapped = provider._map_error(httpx.ConnectError("connection failed"))
    assert isinstance(mapped, ProviderConnectionError)


def test_map_error_falls_back_to_generic_provider_error():
    provider = GeminiProvider(api_key="fake-key")
    mapped = provider._map_error(RuntimeError("something unexpected"))
    assert isinstance(mapped, ProviderError)


def test_generate_maps_sdk_error_through_map_error():
    provider = GeminiProvider(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("simulated failure")

    with patch.object(provider, "_get_client_and_types") as mock_get_client:
        from google.genai import types
        mock_get_client.return_value = (mock_client, types)

        with pytest.raises(ProviderError):
            provider.generate(
                [{"role": "user", "content": "hi"}],
                model="gemini-1.5-flash", temperature=0.3, max_tokens=100, timeout=10,
            )


# --- package-not-installed path (simulated, deterministic) -----------------------


def test_get_client_raises_provider_error_when_sdk_not_installed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "google.genai" or name.startswith("google.genai"):
            raise ImportError("simulated: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    provider = GeminiProvider(api_key="fake-key")
    with pytest.raises(ProviderError, match="google-genai"):
        provider._get_client_and_types(timeout=10)
