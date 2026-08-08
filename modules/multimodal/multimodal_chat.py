"""
modules/multimodal/multimodal_chat.py

Streamlit page for the Multimodal AI Assistant (Milestone 5): upload
an image, preview it, ask questions about it, and view past analyses.
Reuses the shared page-header component and the same "cache the heavy
resource, catch specific errors, never crash" pattern established in
Milestones 1-4.
"""

from __future__ import annotations

import uuid

import streamlit as st

from components import render_page_header
from utils.logger import get_logger

from .image_loader import InvalidImageError
from .manager import MultimodalManager

logger = get_logger(__name__)

_SUGGESTED_QUESTIONS = [
    "Describe this image.",
    "What objects are visible?",
    "Explain what is happening.",
    "Summarize this image.",
]


@st.cache_resource(show_spinner=False)
def _get_manager() -> MultimodalManager:
    return MultimodalManager()


def render_multimodal_page() -> None:
    render_page_header(
        "🖼️ Multimodal AI",
        "Upload an image and ask questions about it using a vision-capable LLM.",
    )

    manager = _get_manager()

    if "multimodal_session_id" not in st.session_state:
        st.session_state.multimodal_session_id = str(uuid.uuid4())
    if "multimodal_prompt" not in st.session_state:
        st.session_state.multimodal_prompt = ""

    if not manager.vision_available:
        st.warning(
            "⚠️ The currently selected model does not support image analysis. "
            "Please select a vision-capable model (e.g. an OpenAI `gpt-4o` model, "
            "a Gemini `1.5`+ model, or a Groq Llama-Vision model) via `LLM_PROVIDER` "
            "/ `LLM_MODEL` in your `.env` file."
        )

    st.subheader("📤 Upload Image")
    st.caption("Supported formats: PNG, JPG, JPEG")
    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
    )

    if uploaded_file is None:
        _render_history(manager)
        return

    file_bytes = uploaded_file.getvalue()
    try:
        image_info = manager.validate(uploaded_file.name, file_bytes)
    except InvalidImageError as exc:
        st.error(f"⚠️ {exc}")
        _render_history(manager)
        return
    except Exception as exc:  # noqa: BLE001 - never crash the page
        logger.error("Unexpected error validating image '%s': %s", uploaded_file.name, exc)
        st.error("⚠️ Something went wrong reading this image. Please try again.")
        _render_history(manager)
        return

    st.image(
        file_bytes,
        caption=f"{uploaded_file.name} ({image_info.width}×{image_info.height}, {image_info.format_name})",
        use_container_width=True,
    )

    st.subheader("❓ Ask About This Image")

    suggestion_cols = st.columns(len(_SUGGESTED_QUESTIONS))
    for col, question in zip(suggestion_cols, _SUGGESTED_QUESTIONS):
        if col.button(question, key=f"suggest_{question}", use_container_width=True):
            st.session_state.multimodal_prompt = question

    prompt = st.text_input(
        "What would you like to know about this image?",
        value=st.session_state.multimodal_prompt,
        placeholder="e.g. 'Describe this image' or 'What objects are visible?'",
    )
    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if not prompt or not prompt.strip():
            st.warning("Please enter a question about the image.")
        else:
            with st.spinner("Analyzing image..."):
                result = manager.analyze(
                    st.session_state.multimodal_session_id,
                    uploaded_file.name, file_bytes, prompt, image_info,
                )

            if result.status == "success":
                st.markdown("**AI Response**")
                st.write(result.response)
            else:
                st.error(f"⚠️ {result.message}")

    _render_history(manager)


def _render_history(manager: MultimodalManager) -> None:
    history = manager.get_history(st.session_state.multimodal_session_id)
    if not history:
        return

    st.divider()
    st.subheader("🕘 Previous Analyses (this session)")
    for entry in reversed(history):
        with st.expander(f"📷 {entry['image_filename']} — {entry['user_prompt'][:60]}"):
            st.caption(entry["created_at"][:19].replace("T", " "))
            st.markdown(f"**Question:** {entry['user_prompt']}")
            st.markdown(f"**Response:** {entry['ai_response']}")
