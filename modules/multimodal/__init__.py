"""
modules/multimodal

Milestone 5: Multimodal AI.

Public API:
    MultimodalManager       - validate/analyze/history orchestration
    AnalysisResult            - result of manager.analyze()
    ValidatedImage              - result of manager.validate()
    render_multimodal_page        - the Streamlit page
    InvalidImageError               - unsupported/oversized/corrupted image
"""

from .image_loader import InvalidImageError, ValidatedImage
from .manager import AnalysisResult, MultimodalManager
from .multimodal_chat import render_multimodal_page

__all__ = [
    "MultimodalManager",
    "AnalysisResult",
    "ValidatedImage",
    "render_multimodal_page",
    "InvalidImageError",
]
