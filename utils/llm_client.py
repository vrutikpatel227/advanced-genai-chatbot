"""
utils/llm_client.py

Thin wrapper around an OpenAI-compatible chat completion API.
Centralized so every module (chat, RAG, research assistant, etc. in
later milestones) calls the LLM through the same, error-handled path
instead of each writing its own HTTP/SDK code.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import llm_config
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM cannot be called due to missing configuration."""


class LLMRequestError(RuntimeError):
    """Raised when the LLM API call itself fails (network, API error, timeout)."""


@dataclass(frozen=True)
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


def is_configured() -> bool:
    """Whether an API key is present. UI should check this before calling."""
    return bool(llm_config.api_key)


def get_chat_completion(messages: list[ChatMessage]) -> str:
    """Send a chat completion request and return the assistant's text reply.

    Raises LLMConfigurationError if no API key is set, and
    LLMRequestError if the call fails for any other reason. Callers
    (the Streamlit UI) are expected to catch these and show a friendly
    message rather than letting the app crash.
    """
    if not is_configured():
        raise LLMConfigurationError(
            "No LLM API key configured. Set OPENAI_API_KEY in your .env file."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMRequestError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from exc

    try:
        client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        response = client.chat.completions.create(
            model=llm_config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.request_timeout,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - any SDK/network failure surfaces uniformly
        logger.error("LLM request failed: %s", exc)
        raise LLMRequestError(f"LLM request failed: {exc}") from exc
