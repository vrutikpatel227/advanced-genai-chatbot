"""
modules/sentiment/response_style.py

Maps a detected SentimentLabel to tone instructions for the LLM, per
PRD section 3 ("Adaptive Responses"):
  - Positive  -> friendly, encouraging
  - Negative  -> supportive, empathetic
  - Neutral   -> professional, informative

Kept inside modules/sentiment/ (rather than hardcoded in app.py) so
the tone-mapping is owned by this milestone's module and can be
extended/tuned without touching the app shell.
"""

from __future__ import annotations

from .analyzer import SentimentLabel

_TONE_INSTRUCTIONS: dict[SentimentLabel, str] = {
    SentimentLabel.POSITIVE: (
        "The customer's message is positive. Respond in a friendly, warm, "
        "and encouraging tone. Match their good energy while staying helpful."
    ),
    SentimentLabel.NEGATIVE: (
        "The customer's message is negative -- they may be frustrated or "
        "upset. Respond with a supportive, empathetic tone: acknowledge "
        "their frustration first, then offer clear, concrete help."
    ),
    SentimentLabel.NEUTRAL: (
        "The customer's message is neutral. Respond in a professional, "
        "clear, and informative tone."
    ),
}


def get_tone_instructions(label: SentimentLabel) -> str:
    """Return the tone guidance to append to the LLM system prompt for
    the given detected sentiment label."""
    return _TONE_INSTRUCTIONS.get(label, _TONE_INSTRUCTIONS[SentimentLabel.NEUTRAL])
