"""
utils/providers/base_provider.py

Abstract base class every LLM provider must implement, plus the
shared provider-level exception hierarchy. utils/llm_client.py
translates these into its own public exceptions
(LLMConfigurationError, LLMRequestError) so the rest of the
application only ever depends on that one stable interface, never on
provider-specific SDK details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Base class for all provider-level errors."""


class ProviderAuthError(ProviderError):
    """Missing or invalid API key for this provider."""


class ProviderRateLimitError(ProviderError):
    """The provider's rate limit was exceeded."""


class ProviderTimeoutError(ProviderError):
    """The request to the provider timed out."""


class ProviderConnectionError(ProviderError):
    """A network/connection failure occurred talking to the provider."""


class LLMProvider(ABC):
    """Common interface every concrete provider (OpenAI, Groq, Gemini,
    ...) must implement. Callers depend only on this interface, never
    on a specific provider's SDK -- this is what lets the app switch
    providers via a single .env variable with zero code changes."""

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has the API key it needs to run."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        """Send a chat completion request and return the reply text.

        messages is a list of {"role": ..., "content": ...} dicts.
        Implementations must translate SDK-specific failures into
        ProviderAuthError, ProviderRateLimitError, ProviderTimeoutError,
        ProviderConnectionError, or a generic ProviderError -- never
        let a raw SDK exception escape uncaught.
        """
        raise NotImplementedError
