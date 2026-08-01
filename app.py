"""
app.py
Entry point for the Advanced GenAI Customer Service Bot (Streamlit).

Foundation: navigation, base chat shell, placeholder pages for future
milestones. Milestone 1 (Sentiment Analysis) is wired in below:
  - Every user message is analyzed before a reply is generated.
  - The chatbot's tone adapts to the detected sentiment.
  - Sentiment + confidence are shown per message and in a side panel.
  - An Analytics page visualizes sentiment distribution.
  - An About Module page documents this milestone.

Future milestones plug into render_page() the same way, only once
their own PRD is provided.
"""

from __future__ import annotations

import uuid

import streamlit as st

from components import (
    NavPage,
    render_about_page,
    render_chat_history,
    render_chat_input,
    render_page_header,
    render_placeholder_page,
    render_sentiment_dashboard,
    render_sentiment_panel,
    render_sidebar,
)
from config import app_config, ensure_directories, sentiment_config
from modules.sentiment import SentimentAnalyzer, get_tone_instructions
from utils.llm_client import (
    ChatMessage,
    LLMConfigurationError,
    LLMRequestError,
    get_chat_completion,
)
from utils.logger import get_logger
from utils.storage import (
    get_sentiment_summary,
    get_total_conversations,
    init_db,
    save_message,
)

logger = get_logger(__name__)

BASE_SYSTEM_PROMPT = (
    "You are a helpful, professional customer service assistant. "
    "Be concise and solution-oriented."
)


@st.cache_resource
def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Cached across reruns so the (possibly heavy) transformer model
    loads only once per server process, not once per user interaction."""
    return SentimentAnalyzer()


def bootstrap() -> None:
    """One-time app setup: directories, database, session state."""
    ensure_directories()
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        logger.error("Database initialization failed: %s", exc)
        st.error(
            "⚠️ Could not initialize the database. Chat will still work this "
            "session, but message history won't be saved. Please check the logs."
        )

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of ChatTurn (role, content, caption)
    if "current_sentiment" not in st.session_state:
        st.session_state.current_sentiment = None  # most recent SentimentResult


def render_chat_page() -> None:
    """Chat interface with real-time sentiment analysis (Milestone 1)."""
    chat_col, panel_col = st.columns([3, 1])

    with chat_col:
        render_page_header(
            "💬 Customer Service Chat",
            "Every message is analyzed for sentiment in real time; the "
            "assistant's tone adapts accordingly.",
        )
        render_chat_history(st.session_state.messages)

        user_input = render_chat_input()

    with panel_col:
        current = st.session_state.current_sentiment
        render_sentiment_panel(
            label=current.label.value if current else None,
            confidence=current.confidence if current else None,
            backend=current.backend if current else None,
        )

    if not user_input:
        return

    analyzer = get_sentiment_analyzer()

    # PRD error handling: empty input. render_chat_input already only
    # returns a non-empty string on submit, but guard defensively too.
    if not user_input.strip():
        st.warning("Please enter a message.")
        return

    try:
        sentiment = analyzer.analyze(user_input)
    except Exception as exc:  # noqa: BLE001 - analyzer should self-fallback, but never crash the chat
        logger.error("Sentiment analysis failed unexpectedly: %s", exc)
        sentiment = None

    sentiment_caption = None
    if sentiment is not None:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(sentiment.label.value, "⚪")
        sentiment_caption = (
            f"{emoji} {sentiment.label.value.title()} "
            f"(confidence {sentiment.confidence:.0%}, via {sentiment.backend})"
        )
        st.session_state.current_sentiment = sentiment

    user_turn = {"role": "user", "content": user_input}
    if sentiment_caption:
        user_turn["caption"] = sentiment_caption
    st.session_state.messages.append(user_turn)

    try:
        save_message(
            session_id=st.session_state.session_id,
            role="user",
            content=user_input,
            sentiment_label=sentiment.label.value if sentiment else None,
            sentiment_confidence=sentiment.confidence if sentiment else None,
        )
    except Exception as exc:  # noqa: BLE001 - PRD error handling: database failure
        logger.error("Failed to save user message: %s", exc)
        st.warning("⚠️ Couldn't save this message to history, but the conversation will continue.")

    with chat_col:
        with st.chat_message("user"):
            st.write(user_input)
            if sentiment_caption:
                st.caption(sentiment_caption)

        with st.chat_message("assistant"):
            try:
                tone_instructions = (
                    get_tone_instructions(sentiment.label) if sentiment else ""
                )
                system_prompt = f"{BASE_SYSTEM_PROMPT} {tone_instructions}".strip()

                history = [ChatMessage("system", system_prompt)]
                history.extend(
                    ChatMessage(m["role"], m["content"]) for m in st.session_state.messages
                )
                reply = get_chat_completion(history)
            except LLMConfigurationError:
                reply = (
                    "⚠️ I can't reach the language model yet because no API key is "
                    "configured. Set `OPENAI_API_KEY` in your `.env` file to enable "
                    "live replies. (Sentiment analysis above still works.)"
                )
            except LLMRequestError as exc:
                logger.error("Chat completion failed: %s", exc)
                reply = "⚠️ Sorry, I ran into a problem reaching the assistant. Please try again shortly."

            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    try:
        save_message(session_id=st.session_state.session_id, role="assistant", content=reply)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save assistant message: %s", exc)


def render_analytics_page() -> None:
    """Sentiment analytics dashboard (Milestone 1, PRD section 5)."""
    render_page_header("📊 Analytics", "Sentiment distribution across all recorded conversations.")
    try:
        total = get_total_conversations()
        summary = get_sentiment_summary()
    except Exception as exc:  # noqa: BLE001 - PRD error handling: database failure
        logger.error("Failed to load analytics data: %s", exc)
        st.error("⚠️ Couldn't load analytics data right now. Please try again shortly.")
        return

    render_sentiment_dashboard(total, summary)


def render_about_sentiment_page() -> None:
    """About Module page for the Sentiment Analysis milestone."""
    render_about_page(
        title="ℹ️ About: Sentiment Analysis Module",
        description=(
            "This milestone analyzes every customer message in real time and "
            "classifies it as Positive, Negative, or Neutral, then adapts the "
            "assistant's tone to match."
        ),
        details=[
            f"Primary model: `{sentiment_config.model_name}` (HuggingFace transformers)",
            "Fallback: dependency-free rule-based lexicon scorer (used automatically "
            "if the transformer model can't be loaded)",
            "Adaptive tone: friendly/encouraging (positive), supportive/empathetic "
            "(negative), professional/informative (neutral)",
            "Every analyzed message and its sentiment/confidence are stored in SQLite",
            "See the Analytics page for aggregate sentiment statistics",
        ],
    )


def render_page(page: NavPage) -> None:
    """Route to the right page renderer. Add a branch here per milestone
    once it's implemented -- until then its NavPage.implemented stays
    False and it falls through to the shared placeholder renderer."""
    if page.key == "chat":
        render_chat_page()
    elif page.key == "analytics":
        render_analytics_page()
    elif page.key == "about_sentiment":
        render_about_sentiment_page()
    else:
        render_placeholder_page(page)


def main() -> None:
    st.set_page_config(page_title=app_config.app_title, page_icon="🤖", layout="wide")
    bootstrap()
    selected_page = render_sidebar(app_config.app_title)
    render_page(selected_page)


if __name__ == "__main__":
    main()
