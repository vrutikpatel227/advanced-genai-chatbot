"""
app.py
Entry point for the Advanced GenAI Customer Service Bot (Streamlit).

This is the *foundation* shell: navigation, a base chat interface
(plain LLM call, no milestone-specific intelligence), and placeholder
pages for every future milestone. Each milestone will be implemented
in its own module under modules/ and wired into render_page() below
only when its PRD is provided -- this file should need only a small,
additive change per milestone, never a rewrite.
"""

from __future__ import annotations

import uuid

import streamlit as st

from components import (
    NavPage,
    render_chat_history,
    render_chat_input,
    render_page_header,
    render_placeholder_page,
    render_sidebar,
)
from config import app_config, ensure_directories
from utils.llm_client import (
    ChatMessage,
    LLMConfigurationError,
    LLMRequestError,
    get_chat_completion,
)
from utils.logger import get_logger
from utils.storage import init_db, save_message

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful, professional customer service assistant. "
    "Be concise, polite, and solution-oriented."
)


def bootstrap() -> None:
    """One-time app setup: directories, database, session state."""
    ensure_directories()
    init_db()
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content"}


def render_chat_page() -> None:
    """Base chatbot interface: plain LLM Q&A, no milestone-specific logic."""
    render_page_header(
        "💬 Customer Service Chat",
        "Base chat interface. Milestone-specific intelligence (sentiment, RAG, "
        "multimodal, etc.) will be layered in once each milestone's PRD is provided.",
    )

    render_chat_history(st.session_state.messages)

    user_input = render_chat_input()
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(session_id=st.session_state.session_id, role="user", content=user_input)

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            history = [ChatMessage("system", SYSTEM_PROMPT)]
            history.extend(
                ChatMessage(m["role"], m["content"]) for m in st.session_state.messages
            )
            reply = get_chat_completion(history)
        except LLMConfigurationError:
            reply = (
                "⚠️ I can't reach the language model yet because no API key is "
                "configured. Set `OPENAI_API_KEY` in your `.env` file to enable "
                "live replies."
            )
        except LLMRequestError as exc:
            logger.error("Chat completion failed: %s", exc)
            reply = "⚠️ Sorry, I ran into a problem reaching the assistant. Please try again shortly."

        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    save_message(session_id=st.session_state.session_id, role="assistant", content=reply)


def render_page(page: NavPage) -> None:
    """Route to the right page renderer. Add a branch here per milestone
    once it's implemented -- until then its NavPage.implemented stays
    False and it falls through to the shared placeholder renderer."""
    if page.key == "chat":
        render_chat_page()
    else:
        render_placeholder_page(page)


def main() -> None:
    st.set_page_config(page_title=app_config.app_title, page_icon="🤖", layout="wide")
    bootstrap()
    selected_page = render_sidebar(app_config.app_title)
    render_page(selected_page)


if __name__ == "__main__":
    main()
