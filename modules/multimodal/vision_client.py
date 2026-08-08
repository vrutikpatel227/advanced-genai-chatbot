"""
modules/multimodal/vision_client.py

Thin wrapper around utils/llm_client.py's vision functions. This
module never imports Groq/OpenAI/Gemini SDKs directly, per the PRD's
"Do not hardcode Groq, OpenAI, or Gemini" requirement -- it only calls
the existing configurable LLM client.
"""

from __future__ import annotations

from utils.llm_client import (
    LLMConfigurationError,
    LLMRequestError,
    VisionNotSupportedError,
    get_vision_completion,
    is_vision_supported,
)

__all__ = [
    "LLMConfigurationError",
    "LLMRequestError",
    "VisionNotSupportedError",
    "analyze_image",
    "is_vision_available",
]


def is_vision_available() -> bool:
    """Whether the currently configured LLM provider/model supports
    image input."""
    return is_vision_supported()


def analyze_image(prompt: str, image_bytes: bytes, image_mime_type: str) -> str:
    """Send an image + text prompt to the currently configured
    vision-capable LLM and return its text response."""
    return get_vision_completion(prompt, image_bytes, image_mime_type)
