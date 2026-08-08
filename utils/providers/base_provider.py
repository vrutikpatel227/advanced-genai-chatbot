"""
utils/providers/base_provider.py

Abstract base class every LLM provider must implement, plus the
shared provider-level exception hierarchy. utils/llm_client.py
translates these into its own public exceptions
(LLMConfigurationError, LLMRequestError) so the rest of the
application only ever depends on that one stable interface, never on
provider-specific SDK details.

Milestone 5 (Multimodal AI) adds optional vision support to this
interface: supports_vision() and generate_with_image() have concrete
default implementations here (not abstract), so every existing
provider keeps working completely unchanged unless it explicitly
overrides them to add real image support.
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


class VisionNotSupportedError(ProviderError):
    """Raised when the currently selected provider/model doesn't
    support image input. Callers show this as a friendly
    "please select a vision-capable model" message, per the PRD --
    never a crash."""


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

    def supports_vision(self, model: str) -> bool:
        """Whether the given model (for this provider) accepts image
        input. Default: no vision support -- providers that do support
        it override this with their own known-vision-model check."""
        return False

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime_type: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        """Send a vision request (text prompt + image) and return the
        reply text. Default implementation always raises
        VisionNotSupportedError -- providers that support vision must
        override this."""
        raise VisionNotSupportedError(
            f"The '{self.name}' provider's vision support is not implemented."
        )
