"""
components/chat.py

Reusable chat-rendering pieces. Deliberately has no knowledge of
sentiment, RAG, or any other milestone-specific intelligence -- it
just renders a list of {"role", "content"} messages and collects new
input. Milestone modules can wrap these to add their own annotations
(e.g. a sentiment badge) without modifying this file.
"""

from __future__ import annotations

from typing import TypedDict

import streamlit as st


class ChatTurn(TypedDict, total=False):
    role: str      # "user" | "assistant"
    content: str
    caption: str    # optional small annotation shown under the message
                    # (e.g. a sentiment badge) -- generic on purpose so
                    # this component stays decoupled from any one milestone


def render_chat_history(messages: list[ChatTurn]) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            caption = msg.get("caption")
            if caption:
                st.caption(caption)


def render_chat_input(placeholder: str = "Type your message...") -> str | None:
    return st.chat_input(placeholder)
