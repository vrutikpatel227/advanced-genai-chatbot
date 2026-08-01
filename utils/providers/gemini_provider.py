"""
utils/providers/gemini_provider.py

Google Gemini provider -- a future-ready placeholder implementation,
per the PRD. It's fully wired into the LLMProvider interface
(is_configured / generate) and can be selected today via
LLM_PROVIDER=gemini, but `google-generativeai` is not yet installed as
a hard dependency (see requirements.txt) since this path hasn't been
verified against the live API. If actually invoked without the
package installed, it raises a clear, friendly ProviderError instead
of crashing -- exactly the "Missing API key / invalid provider"
style error handling the rest of the app already expects.

To activate for real: `pip install google-generativeai` and uncomment
it in requirements.txt.
"""

from __future__ import annotations

from .base_provider import LLMProvider, ProviderAuthError, ProviderError


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = "gemini-1.5-flash"

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
            raise ProviderAuthError("GEMINI_API_KEY is not set.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderError(
                "Gemini support is a future-ready placeholder: the "
                "'google-generativeai' package isn't installed yet. "
                "Run: pip install google-generativeai"
            ) from exc

        try:
            genai.configure(api_key=self._api_key)

            system_instruction = "\n".join(
                m["content"] for m in messages if m["role"] == "system"
            ) or None
            conversation = [m for m in messages if m["role"] != "system"]

            gen_model = genai.GenerativeModel(
                model_name=model or self.default_model,
                system_instruction=system_instruction,
            )
            history = [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [m["content"]],
                }
                for m in conversation[:-1]
            ]
            chat = gen_model.start_chat(history=history)
            response = chat.send_message(
                conversation[-1]["content"],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                request_options={"timeout": timeout},
            )
            return response.text or ""
        except Exception as exc:  # noqa: BLE001 - placeholder: not yet verified against the live API
            raise ProviderError(f"Gemini request failed: {exc}") from exc
