"""
tests/test_sentiment.py

Unit tests for modules.sentiment. Forces the lexicon (rule-based)
backend so tests are deterministic and run offline -- no model
download or GPU required, matching the PRD's required fallback path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.sentiment import SentimentAnalyzer, SentimentLabel, get_tone_instructions  # noqa: E402


def make_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer(force_lexicon=True)


def test_positive_sentence():
    analyzer = make_analyzer()
    result = analyzer.analyze("This is amazing, thank you so much for the fast help!")
    assert result.label == SentimentLabel.POSITIVE
    assert result.backend == "lexicon"
    assert result.score > 0
    assert 0.0 <= result.confidence <= 1.0


def test_negative_sentence():
    analyzer = make_analyzer()
    result = analyzer.analyze("This is terrible, the product is broken and I hate it.")
    assert result.label == SentimentLabel.NEGATIVE
    assert result.score < 0


def test_neutral_sentence():
    analyzer = make_analyzer()
    result = analyzer.analyze("What are your business hours?")
    assert result.label == SentimentLabel.NEUTRAL


def test_empty_input_is_handled_gracefully():
    analyzer = make_analyzer()
    result = analyzer.analyze("")
    assert result.label == SentimentLabel.NEUTRAL
    assert result.confidence == 0.0
    assert result.backend == "none"


def test_whitespace_only_input():
    analyzer = make_analyzer()
    result = analyzer.analyze("   \n\t  ")
    assert result.label == SentimentLabel.NEUTRAL
    assert result.backend == "none"


def test_negation_flips_sentiment_direction():
    analyzer = make_analyzer()
    positive = analyzer.analyze("This is good.")
    negated = analyzer.analyze("This is not good.")
    assert negated.score < positive.score


def test_long_input_is_not_rejected():
    analyzer = make_analyzer()
    long_text = "This is great. " * 200
    result = analyzer.analyze(long_text)
    assert result.label in (SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL)


def test_analyzer_falls_back_when_transformer_unavailable(monkeypatch):
    """Simulates the model-loading-failure path required by the PRD's
    error handling section, without needing network access."""
    analyzer = SentimentAnalyzer(force_lexicon=False)

    def _boom():
        raise RuntimeError("simulated: no network access to download model weights")

    monkeypatch.setattr(
        "modules.sentiment.analyzer._load_transformer_pipeline",
        _boom,
    )
    result = analyzer.analyze("This is great!")
    assert result.backend == "lexicon"
    assert result.label == SentimentLabel.POSITIVE


def test_tone_instructions_cover_every_label():
    for label in SentimentLabel:
        tone = get_tone_instructions(label)
        assert isinstance(tone, str) and len(tone) > 0


def test_tone_instructions_differ_by_label():
    positive_tone = get_tone_instructions(SentimentLabel.POSITIVE)
    negative_tone = get_tone_instructions(SentimentLabel.NEGATIVE)
    neutral_tone = get_tone_instructions(SentimentLabel.NEUTRAL)
    assert len({positive_tone, negative_tone, neutral_tone}) == 3
