"""Shared, reusable Streamlit UI components.

Every milestone should build its pages out of these components (or add
new ones here) rather than writing raw Streamlit calls scattered across
the app, so the look, feel, and navigation stay consistent as the
project grows across milestones.
"""

from .layout import render_page_header, render_sidebar
from .chat import render_chat_history, render_chat_input
from .placeholder import render_placeholder_page
from .navigation import PAGES, NavPage, get_page

__all__ = [
    "render_page_header",
    "render_sidebar",
    "render_chat_history",
    "render_chat_input",
    "render_placeholder_page",
    "PAGES",
    "NavPage",
    "get_page",
]
