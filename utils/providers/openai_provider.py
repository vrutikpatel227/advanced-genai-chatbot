"""
utils/providers/openai_provider.py

OpenAI provider implementation of the LLMProvider interface.
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


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url

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
            raise ProviderAuthError("OPENAI_API_KEY is not set.")

        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as exc:
            raise ProviderError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc

        try:
            client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            response = client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError as exc:
            raise ProviderAuthError(f"OpenAI authentication failed: {exc}") from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc
        except APITimeoutError as exc:
            raise ProviderTimeoutError(f"OpenAI request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise ProviderConnectionError(f"OpenAI connection failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - final safety net for any other SDK error
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
