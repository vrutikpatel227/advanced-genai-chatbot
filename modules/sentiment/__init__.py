"""
modules/sentiment

Milestone 1: Sentiment Analysis.

Public API:
    SentimentAnalyzer   - analyze(text) -> SentimentResult
    SentimentLabel       - POSITIVE / NEGATIVE / NEUTRAL
    SentimentResult      - dataclass with label, score, confidence, backend
    get_tone_instructions - maps a SentimentLabel to LLM tone guidance
"""

from .analyzer import SentimentAnalyzer, SentimentLabel, SentimentResult
from .response_style import get_tone_instructions

__all__ = [
    "SentimentAnalyzer",
    "SentimentLabel",
    "SentimentResult",
    "get_tone_instructions",
]
