"""
components/layout.py

Shared page-header and sidebar components. Kept separate from
navigation.py's page registry so the *data* (what pages exist) and the
*rendering* (how they're drawn) can change independently.
"""

from __future__ import annotations

import streamlit as st

from components.navigation import PAGES, NavPage
from utils.llm_client import is_configured


def render_page_header(title: str, subtitle: str | None = None) -> None:
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def render_sidebar(app_title: str) -> NavPage:
    """Render the sidebar navigation and return the selected NavPage.

    Reads the page list from components.navigation.PAGES, so adding a
    milestone there automatically shows up here with no other changes.
    """
    st.sidebar.title(app_title)

    labels = [p.label for p in PAGES]
    selected_label = st.sidebar.radio("Navigate", labels, label_visibility="collapsed")
    selected = next(p for p in PAGES if p.label == selected_label)

    st.sidebar.divider()
    st.sidebar.caption("Milestone status:")
    st.sidebar.markdown(
        "\n".join(
            f"- {'✅' if p.implemented else '⬜'} {p.label}" for p in PAGES
        )
    )

    if not is_configured():
        st.sidebar.divider()
        st.sidebar.warning(
            "No LLM API key set. Chat replies will show a configuration notice "
            "until `OPENAI_API_KEY` is set in `.env`."
        )

    return selected
