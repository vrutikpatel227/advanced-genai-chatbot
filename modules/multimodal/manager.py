"""
modules/multimodal/manager.py

Top-level orchestration for the Multimodal AI Assistant: validate
image -> vision-capability check -> analyze via the existing
configurable LLM provider -> save conversation. Never crashes -- every
failure mode maps to a friendly AnalysisResult instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger
from utils.storage import get_multimodal_conversations, save_multimodal_conversation

from .config import MultimodalConfig, ensure_multimodal_directories, multimodal_config
from .image_loader import InvalidImageError, ValidatedImage, validate_image
from .vision_client import (
    LLMConfigurationError,
    LLMRequestError,
    VisionNotSupportedError,
    analyze_image,
    is_vision_available,
)

logger = get_logger(__name__)

__all__ = ["AnalysisResult", "MultimodalManager", "InvalidImageError", "ValidatedImage"]


@dataclass(frozen=True)
class AnalysisResult:
    status: str          # "success" | "error"
    message: str
    response: Optional[str] = None


class MultimodalManager:
    def __init__(self, config: Optional[MultimodalConfig] = None) -> None:
        self._config = config or multimodal_config

    @property
    def vision_available(self) -> bool:
        return is_vision_available()

    def validate(self, filename: str, file_bytes: bytes) -> ValidatedImage:
        """Validate an uploaded image. Raises InvalidImageError with a
        friendly message on any problem."""
        ensure_multimodal_directories()
        return validate_image(filename, file_bytes, self._config)

    def analyze(
        self,
        session_id: str,
        filename: str,
        file_bytes: bytes,
        prompt: str,
        image: ValidatedImage,
    ) -> AnalysisResult:
        """Full analysis pipeline. Never raises -- every failure mode
        (empty prompt, unsupported vision model, missing API key,
        request failure, database failure) maps to a friendly
        AnalysisResult instead."""
        if not prompt or not prompt.strip():
            return AnalysisResult(status="error", message="Please enter a question about the image.")

        try:
            response_text = analyze_image(prompt, file_bytes, image.mime_type)
        except VisionNotSupportedError as exc:
            return AnalysisResult(status="error", message=str(exc))
        except LLMConfigurationError as exc:
            return AnalysisResult(status="error", message=str(exc))
        except LLMRequestError as exc:
            logger.error("Vision analysis failed for '%s': %s", filename, exc)
            return AnalysisResult(
                status="error",
                message="Sorry, I ran into a problem analyzing this image. Please try again shortly.",
            )
        except Exception as exc:  # noqa: BLE001 - never let an unexpected error crash the page
            logger.error("Unexpected error analyzing '%s': %s", filename, exc)
            return AnalysisResult(
                status="error",
                message="Something went wrong analyzing this image. Please try again.",
            )

        logger.info("Image analysis succeeded for '%s'.", filename)

        try:
            save_multimodal_conversation(session_id, filename, prompt, response_text)
        except Exception as exc:  # noqa: BLE001 - PRD error handling: database failure
            logger.error("Failed to save multimodal conversation: %s", exc)
            return AnalysisResult(
                status="success",
                message="Analysis complete, but saving it to history failed.",
                response=response_text,
            )

        return AnalysisResult(status="success", message="Analysis complete.", response=response_text)

    def get_history(self, session_id: str):
        return get_multimodal_conversations(session_id)
