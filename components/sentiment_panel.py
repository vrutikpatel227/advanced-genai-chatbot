"""
components/sentiment_panel.py

Small side-panel showing the most recently detected sentiment +
confidence, per PRD section "UI Requirements -> Right panel or
sidebar: Current sentiment, Confidence".

Accepts plain primitives (label/confidence/backend strings) rather
than importing modules.sentiment types directly, so this shared
component doesn't couple components/ to a specific milestone module.
"""

from __future__ import annotations

import streamlit as st

_COLOR_BY_LABEL = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "🟡",
}


def render_sentiment_panel(label: str | None, confidence: float | None, backend: str | None = None) -> None:
    """Render the current-sentiment panel. Shows a neutral placeholder
    if no message has been analyzed yet this session."""
    st.subheader("Current Sentiment")

    if label is None:
        st.caption("Send a message to see its sentiment here.")
        return

    emoji = _COLOR_BY_LABEL.get(label, "⚪")
    st.metric(label="Sentiment", value=f"{emoji} {label.title()}")
    if confidence is not None:
        st.metric(label="Confidence", value=f"{confidence:.0%}")
    if backend:
        st.caption(f"via {backend} backend")
