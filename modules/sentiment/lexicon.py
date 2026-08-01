"""
modules/sentiment/lexicon.py

A small, self-contained, weighted-lexicon (rule-based) sentiment
scorer. Serves as the PRD-required fallback when the HuggingFace
transformer model can't be loaded (no network access to download
weights, or the `transformers`/`torch` packages aren't installed).

This is intentionally simple -- not a replacement for the transformer
backend, just a dependency-free safety net so the chatbot never breaks
because sentiment analysis is temporarily unavailable.
"""

from __future__ import annotations

import re

_POSITIVE_WORDS = {
    "good": 1.0, "great": 1.6, "excellent": 2.0, "amazing": 2.0, "awesome": 1.8,
    "love": 1.8, "loved": 1.8, "happy": 1.4, "pleased": 1.3, "satisfied": 1.4,
    "thanks": 1.0, "thank": 1.0, "helpful": 1.3, "perfect": 1.8, "fast": 0.8,
    "easy": 0.8, "wonderful": 1.8, "fantastic": 1.9, "resolved": 1.2, "works": 0.7,
    "nice": 1.0, "best": 1.5, "recommend": 1.3, "smooth": 1.0, "quick": 0.8,
}

_NEGATIVE_WORDS = {
    "bad": -1.2, "terrible": -2.0, "awful": -2.0, "horrible": -2.0, "worst": -2.0,
    "hate": -1.9, "hated": -1.9, "angry": -1.6, "frustrated": -1.6, "frustrating": -1.6,
    "annoyed": -1.3, "annoying": -1.3, "broken": -1.5, "bug": -1.0, "issue": -0.8,
    "problem": -0.9, "disappointed": -1.6, "disappointing": -1.6, "slow": -0.9,
    "useless": -1.8, "refund": -0.8, "cancel": -0.9, "cancelled": -0.9, "waited": -0.7,
    "waiting": -0.6, "unacceptable": -2.0, "scam": -2.2, "never": -0.6, "worse": -1.5,
    "complaint": -1.0, "rude": -1.7, "delay": -0.8, "delayed": -0.8, "error": -0.9,
    "crash": -1.4, "crashed": -1.4, "fail": -1.3, "failed": -1.3,
}

_NEGATORS = {"not", "no", "never", "n't", "cannot", "can't", "won't", "didn't", "isn't"}
_INTENSIFIERS = {"very": 1.4, "extremely": 1.7, "really": 1.3, "so": 1.2, "absolutely": 1.6}

_TOKEN_RE = re.compile(r"[a-z']+")


def score_text(text: str) -> tuple[float, float]:
    """Return (compound_score, confidence) for the given text.

    compound_score is roughly in [-1, 1]; confidence is derived from
    how many sentiment-bearing tokens were found relative to length.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0, 0.0

    raw_total = 0.0
    hits = 0
    negate_next = False
    intensity = 1.0

    for tok in tokens:
        if tok in _NEGATORS:
            negate_next = True
            continue
        if tok in _INTENSIFIERS:
            intensity = _INTENSIFIERS[tok]
            continue

        weight = _POSITIVE_WORDS.get(tok) or _NEGATIVE_WORDS.get(tok)
        if weight is not None:
            hits += 1
            adjusted = weight * intensity
            if negate_next:
                adjusted *= -0.8
            raw_total += adjusted

        negate_next = False
        intensity = 1.0

    if hits == 0:
        return 0.0, 0.1

    # Normalize into roughly [-1, 1] using a soft cap.
    compound = max(-1.0, min(1.0, raw_total / (hits * 2.0) + raw_total / len(tokens)))
    confidence = min(1.0, 0.35 + 0.12 * hits)
    return compound, confidence
