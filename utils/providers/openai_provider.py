"""
utils/providers/openai_provider.py

OpenAI provider implementation of the LLMProvider interface. Also
implements vision support (Milestone 5) via OpenAI's image_url content
part format, supported by the gpt-4o / gpt-4-turbo model families.
"""

from __future__ import annotations

import base64

from .base_provider import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Known vision-capable OpenAI model name fragments. Checked via
# substring match so future dated snapshots (e.g. "gpt-4o-2024-08-06")
# are still recognized without needing an update here.
_VISION_MODEL_FRAGMENTS = ("gpt-4o", "gpt-4-turbo", "gpt-4-vision")


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def supports_vision(self, model: str) -> bool:
        model_name = (model or self.default_model).lower()
        return any(fragment in model_name for fragment in _VISION_MODEL_FRAGMENTS)

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

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image_mime_type};base64,{encoded}"

        try:
            client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            response = client.chat.completions.create(
                model=model or self.default_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
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
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI vision request failed: {exc}") from exc
