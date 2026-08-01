"""
modules/sentiment/analyzer.py

Milestone 1: Sentiment Analysis.

Public interface: SentimentAnalyzer.analyze(text) -> SentimentResult

Design notes (per PRD "Technical Requirements"):
- Primary backend: HuggingFace transformers text-classification pipeline,
  cardiffnlp/twitter-roberta-base-sentiment-latest by default. Requires
  the `transformers`/`torch` packages and, on first run, network access
  to download model weights.
- Fallback backend: a dependency-free rule-based lexicon scorer
  (modules/sentiment/lexicon.py) that always works offline. The
  analyzer transparently falls back to this if the transformer backend
  can't be loaded, so sentiment analysis (and therefore the whole
  chatbot, since every message is analyzed before a reply is
  generated) never breaks due to model/network unavailability.
- Every result records which backend actually produced it, so this
  degradation is visible in the UI and logs rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

from config import sentiment_config
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentResult:
    text: str
    label: SentimentLabel
    score: float          # signed score, roughly -1 (very negative) to +1 (very positive)
    confidence: float      # 0..1, how confident the backend is
    backend: str           # "transformer" or "lexicon"


def _label_from_score(score: float, confidence: float) -> SentimentLabel:
    if confidence < sentiment_config.confidence_floor:
        return SentimentLabel.NEUTRAL
    if score > 0.15:
        return SentimentLabel.POSITIVE
    if score < -0.15:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


@lru_cache(maxsize=1)
def _load_transformer_pipeline():
    """Lazily load the HF pipeline. Cached so we only pay the load cost once.

    Raises if transformers/torch aren't installed or the model can't be
    fetched -- callers must catch this and fall back to the lexicon.
    """
    from transformers import pipeline  # local import: optional heavy dependency

    return pipeline(
        task="sentiment-analysis",
        model=sentiment_config.model_name,
        top_k=None,
    )


class SentimentAnalyzer:
    """Analyzes customer message sentiment with automatic fallback."""

    def __init__(self, force_lexicon: bool = False) -> None:
        self._force_lexicon = force_lexicon or not sentiment_config.use_transformer_model
        self._transformer_available: Optional[bool] = None

    def _transformer_ready(self) -> bool:
        if self._force_lexicon:
            return False
        if self._transformer_available is not None:
            return self._transformer_available

        try:
            _load_transformer_pipeline()
            self._transformer_available = True
        except Exception as exc:  # noqa: BLE001 - any load failure -> fallback
            logger.warning(
                "Transformer sentiment backend unavailable (%s); falling back to rule-based lexicon scorer.",
                exc,
            )
            self._transformer_available = False

        return self._transformer_available

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a message. Empty/whitespace-only input is handled
        gracefully and returns a neutral result rather than raising."""
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                backend="none",
            )

        if self._transformer_ready():
            try:
                return self._analyze_with_transformer(text)
            except Exception as exc:  # noqa: BLE001
                logger.error("Transformer inference failed (%s); falling back to lexicon.", exc)
                self._transformer_available = False

        return self._analyze_with_lexicon(text)

    def _analyze_with_transformer(self, text: str) -> SentimentResult:
        pipe = _load_transformer_pipeline()
        raw = pipe(text[:512])  # guard against pathologically long input
        predictions = raw[0] if isinstance(raw[0], list) else raw
        best = max(predictions, key=lambda p: p["score"])

        label_map = {
            "positive": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
            "label_0": SentimentLabel.NEGATIVE,
            "label_1": SentimentLabel.NEUTRAL,
            "label_2": SentimentLabel.POSITIVE,
        }
        label = label_map.get(best["label"].lower(), SentimentLabel.NEUTRAL)
        confidence = float(best["score"])
        signed_score = confidence if label == SentimentLabel.POSITIVE else (
            -confidence if label == SentimentLabel.NEGATIVE else 0.0
        )

        return SentimentResult(
            text=text,
            label=label,
            score=signed_score,
            confidence=confidence,
            backend="transformer",
        )

    def _analyze_with_lexicon(self, text: str) -> SentimentResult:
        from .lexicon import score_text

        score, confidence = score_text(text)
        label = _label_from_score(score, confidence)

        return SentimentResult(
            text=text,
            label=label,
            score=score,
            confidence=confidence,
            backend="lexicon",
        )
