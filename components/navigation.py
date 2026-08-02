"""
components/navigation.py

Single source of truth for the app's pages. Add a new NavPage entry
here when a milestone is implemented (or to register its placeholder
ahead of time) -- app.py and the sidebar both read from this list, so
there's only one place to update.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavPage:
    key: str            # stable identifier, used in st.session_state / routing
    label: str          # shown in the sidebar, may include an emoji
    implemented: bool    # False = renders as a "coming soon" placeholder
    description: str = ""  # shown on the placeholder page when not implemented


PAGES: list[NavPage] = [
    NavPage(
        key="chat",
        label="💬 Chat",
        implemented=True,
        description="Conversational interface with real-time sentiment analysis.",
    ),
    NavPage(
        key="analytics",
        label="📊 Analytics",
        implemented=True,
        description="Sentiment analytics: conversation totals and sentiment distribution.",
    ),
    NavPage(
        key="about_sentiment",
        label="ℹ️ About Module",
        implemented=True,
        description="Details about the Sentiment Analysis module (Milestone 1).",
    ),
    NavPage(
        key="medical",
        label="🏥 Medical Knowledge Assistant",
        implemented=True,
        description="Answers medical questions grounded in the MedQuAD knowledge base via RAG.",
    ),
    NavPage(
        key="knowledge_base",
        label="📚 Dynamic Knowledge Base",
        implemented=False,
        description="Will let the bot ingest and query documents supplied at runtime.",
    ),
    NavPage(
        key="research",
        label="🔬 Research Paper Assistant",
        implemented=False,
        description="Will summarize and answer questions about uploaded research papers.",
    ),
    NavPage(
        key="multimodal",
        label="🖼️ Multimodal AI",
        implemented=False,
        description="Will accept image input alongside text.",
    ),
    NavPage(
        key="multilingual",
        label="🌐 Multilingual Support",
        implemented=False,
        description="Will detect and respond in the customer's language.",
    ),
    NavPage(
        key="memory",
        label="🧠 Conversation Memory",
        implemented=False,
        description="Will let the bot recall context across sessions.",
    ),
]


def get_page(key: str) -> NavPage:
    for page in PAGES:
        if page.key == key:
            return page
    raise KeyError(f"Unknown page key: {key!r}")
