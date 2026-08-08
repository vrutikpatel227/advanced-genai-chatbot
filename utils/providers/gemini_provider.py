"""
utils/providers/gemini_provider.py

Google Gemini provider implementation of the LLMProvider interface.

Root cause of the previous placeholder never working (even after
`pip install google-generativeai`): that package -- Google's older,
now-legacy Gemini SDK -- was never added as an active dependency (only
commented out in requirements.txt), so the ImportError branch always
fired and raised the "future-ready placeholder" message. But even
installing it wouldn't have been the right fix: Google has since
released `google-genai`, a newer, actively maintained, unified SDK
that supersedes `google-generativeai`. This implementation uses that
current SDK.

Uses genai.Client(...).models.generate_content(...) for both text and
vision requests -- images are just an additional Part in the same
`contents` list, so no separate vision-specific client/method is
needed at the SDK level (this provider still exposes generate() and
generate_with_image() separately to match the existing LLMProvider
interface used by every other provider).
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

# Known vision-capable Gemini model name fragments. The Gemini 1.5+
# family accepts image input natively; kept as a simple, fast,
# no-network-call heuristic rather than querying the API for model
# capabilities on every check.
_VISION_MODEL_FRAGMENTS = ("gemini-1.5", "gemini-2", "gemini-3")


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = "gemini-1.5-flash"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def supports_vision(self, model: str) -> bool:
        model_name = (model or self.default_model).lower()
        return any(fragment in model_name for fragment in _VISION_MODEL_FRAGMENTS)

    # --- shared helpers ------------------------------------------------------------

    def _get_client_and_types(self, timeout: int):
        """Import the SDK and build a configured Client. Raises
        ProviderError with a clear install instruction if the SDK
        isn't installed -- this is the only place that ImportError
        should ever surface, and only when the package genuinely isn't
        installed (not as a permanent placeholder)."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderError(
                "The 'google-genai' package is not installed. Run: pip install google-genai"
            ) from exc

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=timeout * 1000,  # SDK expects milliseconds
                retry_options=types.HttpRetryOptions(attempts=3),  # PRD: retry handling where appropriate
            ),
        )
        return client, types

    def _map_error(self, exc: Exception) -> Exception:
        """Translate google-genai's exceptions into our shared provider
        exception hierarchy, so callers never need to know which SDK
        is behind the currently selected provider."""
        try:
            from google.genai import errors
        except ImportError:
            errors = None  # type: ignore[assignment]
        try:
            import httpx
        except ImportError:
            httpx = None  # type: ignore[assignment]

        if errors is not None and isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            if code in (401, 403):
                return ProviderAuthError(f"Gemini authentication failed: {exc}")
            if code == 429:
                return ProviderRateLimitError(f"Gemini rate limit exceeded: {exc}")
            if isinstance(code, int) and code >= 500:
                return ProviderConnectionError(f"Gemini server error: {exc}")
            return ProviderError(f"Gemini request failed: {exc}")

        if httpx is not None:
            if isinstance(exc, httpx.TimeoutException):
                return ProviderTimeoutError(f"Gemini request timed out: {exc}")
            if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
                return ProviderConnectionError(f"Gemini connection failed: {exc}")

        return ProviderError(f"Gemini request failed: {exc}")

    # --- text generation -----------------------------------------------------------

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
            raise ProviderAuthError("GEMINI_API_KEY is not set.")

        client, types = self._get_client_and_types(timeout)

        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
        if not conversation:
            raise ProviderError("Gemini request failed: no user/assistant messages to send.")

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in conversation
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction="\n".join(system_texts) if system_texts else None,
        )

        try:
            response = client.models.generate_content(
                model=model or self.default_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - mapped uniformly below
            raise self._map_error(exc) from exc

        return response.text or ""

    # --- vision generation ----------------------------------------------------------

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
            raise ProviderAuthError("GEMINI_API_KEY is not set.")

        model_name = model or self.default_model
        if not self.supports_vision(model_name):
            from .base_provider import VisionNotSupportedError

            raise VisionNotSupportedError(
                f"Gemini model '{model_name}' does not support image input."
            )

        client, types = self._get_client_and_types(timeout)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
                ],
            )
        ]
        config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - mapped uniformly below
            raise self._map_error(exc) from exc

        return response.text or ""
