"""
components/about.py

Generic "About" page renderer. Any implemented milestone can use this
to show a details page (feature overview, model used, etc.) by
passing its own title/description/details -- kept generic rather than
sentiment-specific so future milestones reuse it too.
"""

from __future__ import annotations

import streamlit as st


def render_about_page(title: str, description: str, details: list[str]) -> None:
    st.header(title)
    st.write(description)
    if details:
        st.subheader("Details")
        for item in details:
            st.markdown(f"- {item}")
