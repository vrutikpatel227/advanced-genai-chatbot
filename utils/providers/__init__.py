"""
utils/providers

Provider registry: maps a provider name (from LLM_PROVIDER in .env) to
its concrete implementation. Adding a new provider later means adding
one file here plus one branch in build_provider() -- nothing else in
the application changes.
"""

from __future__ import annotations

from config import llm_config

from .base_provider import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .openai_provider import OpenAIProvider

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "groq", "gemini")


def build_provider(name: str) -> LLMProvider:
    """Instantiate the provider for the given name using current config.

    Raises ValueError for an unrecognized provider name -- callers
    (utils/llm_client.py) translate this into the public
    LLMConfigurationError so the app never crashes on a bad .env value.
    """
    normalized = (name or "").strip().lower()

    if normalized == "openai":
        return OpenAIProvider(api_key=llm_config.openai_api_key, base_url=llm_config.openai_base_url)
    if normalized == "groq":
        return GroqProvider(api_key=llm_config.groq_api_key)
    if normalized == "gemini":
        return GeminiProvider(api_key=llm_config.gemini_api_key)

    raise ValueError(
        f"Unknown LLM_PROVIDER: {name!r}. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )


__all__ = [
    "LLMProvider",
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderConnectionError",
    "build_provider",
    "SUPPORTED_PROVIDERS",
]
