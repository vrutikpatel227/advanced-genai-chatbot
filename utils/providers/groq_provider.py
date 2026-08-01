"""
utils/providers/groq_provider.py

Groq provider implementation of the LLMProvider interface. Groq's
Python SDK mirrors the OpenAI SDK's chat completion interface and
exception types, since Groq's API is OpenAI-compatible.
"""

from __future__ import annotations

from .base_provider import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


class GroqProvider(LLMProvider):
    name = "groq"
    default_model = "llama-3.1-8b-instant"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        if not self.is_configured():
            raise ProviderAuthError("GROQ_API_KEY is not set.")

        try:
            from groq import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                Groq,
                RateLimitError,
            )
        except ImportError as exc:
            raise ProviderError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from exc

        try:
            client = Groq(api_key=self._api_key)
            response = client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError as exc:
            raise ProviderAuthError(f"Groq authentication failed: {exc}") from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError(f"Groq rate limit exceeded: {exc}") from exc
        except APITimeoutError as exc:
            raise ProviderTimeoutError(f"Groq request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise ProviderConnectionError(f"Groq connection failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - final safety net for any other SDK error
            raise ProviderError(f"Groq request failed: {exc}") from exc
