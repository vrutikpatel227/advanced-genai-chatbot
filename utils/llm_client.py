"""
utils/llm_client.py

Unified, provider-agnostic LLM client. Every module (the base chat
page, sentiment-adaptive replies, and every future milestone) calls
get_chat_completion() -- none of them know or care which provider
(OpenAI, Groq, Gemini, ...) is actually selected. The active provider
is chosen purely by the LLM_PROVIDER environment variable; see
config.py for the setting and utils/providers/ for the implementations.

Public API is intentionally unchanged from before this refactor
(ChatMessage, is_configured(), get_chat_completion(),
LLMConfigurationError, LLMRequestError) so no calling code
(app.py, components/) needed to change.

Milestone 5 (Multimodal AI) adds is_vision_supported() and
get_vision_completion() -- vision-capable equivalents of
is_configured()/get_chat_completion() that check whether the currently
selected provider *and model* support image input before calling it,
raising a friendly VisionNotSupportedError rather than a confusing
provider-level failure if not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from config import llm_config
from utils.logger import get_logger
from utils.providers import (
    SUPPORTED_PROVIDERS,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    VisionNotSupportedError as _ProviderVisionNotSupportedError,
    build_provider,
)

logger = get_logger(__name__)

_KEY_HINT_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM cannot be called due to missing/invalid
    configuration: an unknown LLM_PROVIDER value, or the selected
    provider's API key isn't set."""


class LLMRequestError(RuntimeError):
    """Raised when the LLM API call itself fails: network failure,
    rate limit, timeout, or any other provider-side error."""


class VisionNotSupportedError(RuntimeError):
    """Raised when the currently selected provider/model doesn't
    support image input. Callers (the Multimodal AI page) show this as
    a friendly "please select a vision-capable model" message."""


@dataclass(frozen=True)
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@lru_cache(maxsize=1)
def _get_provider():
    """Build (and cache) the provider selected by LLM_PROVIDER.

    Cached because provider selection only changes when the process is
    restarted with a different .env -- never mid-run. Raises ValueError
    for an unknown provider name; callers below translate that into
    LLMConfigurationError.
    """
    return build_provider(llm_config.provider)


def validate_configuration() -> tuple[bool, str]:
    """Startup/runtime validation: is the selected provider recognized,
    and does it have the API key it needs? Never raises -- always
    returns (is_valid, message) so callers can show a friendly message
    instead of crashing the app.
    """
    provider_name = (llm_config.provider or "").strip().lower()

    if provider_name not in SUPPORTED_PROVIDERS:
        return False, (
            f"Invalid LLM_PROVIDER: '{llm_config.provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    provider = _get_provider()

    if not provider.is_configured():
        key_hint = _KEY_HINT_BY_PROVIDER.get(provider_name, f"{provider_name.upper()}_API_KEY")
        return False, (
            f"No API key set for provider '{provider_name}'. "
            f"Set {key_hint} in your .env file."
        )

    model_name = llm_config.model or provider.default_model
    return True, f"LLM provider '{provider_name}' configured (model: {model_name})."


def is_configured() -> bool:
    """Whether the selected provider is valid and has its API key set.
    UI should check this before calling get_chat_completion()."""
    is_valid, _ = validate_configuration()
    return is_valid


def is_vision_supported() -> bool:
    """Whether the currently selected provider AND model support image
    input. UI should check this before calling get_vision_completion()
    to show a friendly message instead of an error."""
    is_valid, _ = validate_configuration()
    if not is_valid:
        return False
    provider = _get_provider()
    model_name = llm_config.model or provider.default_model
    return provider.supports_vision(model_name)


def get_chat_completion(messages: list[ChatMessage]) -> str:
    """Send a chat completion request through the currently selected
    provider and return the assistant's text reply.

    Raises LLMConfigurationError if the provider is invalid or missing
    its API key, and LLMRequestError if the call itself fails (network,
    rate limit, timeout, or any other provider error). Callers (the
    Streamlit UI) are expected to catch these and show a friendly
    message rather than letting the app crash.
    """
    is_valid, message = validate_configuration()
    if not is_valid:
        raise LLMConfigurationError(message)

    provider = _get_provider()
    provider_messages = [{"role": m.role, "content": m.content} for m in messages]
    model_name = llm_config.model or provider.default_model

    start = time.monotonic()
    try:
        reply = provider.generate(
            provider_messages,
            model=model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.request_timeout,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "LLM request succeeded | provider=%s | model=%s | elapsed=%.2fs",
            provider.name, model_name, elapsed,
        )
        return reply
    except ProviderAuthError as exc:
        logger.error("LLM configuration error | provider=%s | %s", provider.name, exc)
        raise LLMConfigurationError(str(exc)) from exc
    except (ProviderRateLimitError, ProviderTimeoutError, ProviderConnectionError, ProviderError) as exc:
        elapsed = time.monotonic() - start
        logger.error(
            "LLM request failed | provider=%s | elapsed=%.2fs | %s",
            provider.name, elapsed, exc,
        )
        raise LLMRequestError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - final safety net: never let a raw error escape
        elapsed = time.monotonic() - start
        logger.error(
            "LLM request failed with an unexpected error | provider=%s | elapsed=%.2fs | %s",
            provider.name, elapsed, exc,
        )
        raise LLMRequestError(f"Unexpected error: {exc}") from exc


def get_vision_completion(prompt: str, image_bytes: bytes, image_mime_type: str) -> str:
    """Send a vision request (text prompt + image) through the
    currently selected provider and return the assistant's text reply.

    Raises LLMConfigurationError if the provider is invalid or missing
    its API key, VisionNotSupportedError if the selected provider/model
    doesn't support image input, and LLMRequestError if the call itself
    fails. Callers (the Multimodal AI page) catch these and show a
    friendly message rather than letting the app crash.
    """
    is_valid, message = validate_configuration()
    if not is_valid:
        raise LLMConfigurationError(message)

    provider = _get_provider()
    model_name = llm_config.model or provider.default_model

    if not provider.supports_vision(model_name):
        raise VisionNotSupportedError(
            f"The currently selected model ('{model_name}' on provider "
            f"'{provider.name}') does not support image analysis. Please "
            f"select a vision-capable model."
        )

    start = time.monotonic()
    try:
        reply = provider.generate_with_image(
            prompt,
            image_bytes,
            image_mime_type,
            model=model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.request_timeout,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "Vision request succeeded | provider=%s | model=%s | elapsed=%.2fs",
            provider.name, model_name, elapsed,
        )
        return reply
    except _ProviderVisionNotSupportedError as exc:
        raise VisionNotSupportedError(str(exc)) from exc
    except ProviderAuthError as exc:
        logger.error("Vision configuration error | provider=%s | %s", provider.name, exc)
        raise LLMConfigurationError(str(exc)) from exc
    except (ProviderRateLimitError, ProviderTimeoutError, ProviderConnectionError, ProviderError) as exc:
        elapsed = time.monotonic() - start
        logger.error(
            "Vision request failed | provider=%s | elapsed=%.2fs | %s",
            provider.name, elapsed, exc,
        )
        raise LLMRequestError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - final safety net: never let a raw error escape
        elapsed = time.monotonic() - start
        logger.error(
            "Vision request failed with an unexpected error | provider=%s | elapsed=%.2fs | %s",
            provider.name, elapsed, exc,
        )
        raise LLMRequestError(f"Unexpected error: {exc}") from exc


# Startup validation + logging (selected provider, model, and whether
# configuration is valid) -- runs once when this module is first
# imported, which happens at application startup (app.py imports this
# module at the top level). Never raises: only logs.
_startup_valid, _startup_message = validate_configuration()
if _startup_valid:
    logger.info("Startup LLM configuration check: %s", _startup_message)
else:
    logger.warning("Startup LLM configuration check failed: %s", _startup_message)
