"""
components/dashboard.py

Analytics dashboard, per PRD section 5:
  - Total Conversations
  - Positive / Negative / Neutral counts
  - Pie chart
  - Bar chart

Accepts a plain summary dict ({"positive": n, "negative": n,
"neutral": n}) rather than importing modules.sentiment types, keeping
this shared component decoupled from any one milestone's data model.
"""

from __future__ import annotations

import streamlit as st

_COLOR_MAP = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#f1c40f"}
_ALL_LABELS = ["Positive", "Negative", "Neutral"]


def render_sentiment_dashboard(total_conversations: int, summary: dict[str, int]) -> None:
    """Render the sentiment analytics dashboard.

    summary keys are expected to be lowercase sentiment labels
    ("positive"/"negative"/"neutral"); missing labels are shown as 0
    rather than omitted, so the chart shape stays consistent.
    """
    counts = {label: summary.get(label.lower(), 0) for label in _ALL_LABELS}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Conversations", total_conversations)
    col2.metric("🟢 Positive", counts["Positive"])
    col3.metric("🔴 Negative", counts["Negative"])
    col4.metric("🟡 Neutral", counts["Neutral"])

    if total_conversations == 0:
        st.info("No conversations recorded yet. Chat with the bot to generate analytics.")
        return

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame({"Sentiment": list(counts.keys()), "Count": list(counts.values())})

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_pie = px.pie(
            df, names="Sentiment", values="Count", color="Sentiment",
            color_discrete_map=_COLOR_MAP, title="Sentiment Distribution (Pie)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with chart_col2:
        fig_bar = px.bar(
            df, x="Sentiment", y="Count", color="Sentiment",
            color_discrete_map=_COLOR_MAP, title="Sentiment Distribution (Bar)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
