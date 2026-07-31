"""
components/placeholder.py

Generic "not implemented yet" page, shown for any milestone in the
navigation registry whose `implemented` flag is still False. Keeps
app.py from needing a bunch of near-duplicate "coming soon" blocks.
"""

from __future__ import annotations

import streamlit as st

from components.navigation import NavPage


def render_placeholder_page(page: NavPage) -> None:
    st.header(page.label)
    st.info(
        f"**Not implemented yet.** {page.description}\n\n"
        "This milestone will be built out once its corresponding PRD is provided."
    )
