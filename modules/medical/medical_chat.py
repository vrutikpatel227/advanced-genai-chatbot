"""
modules/medical/medical_chat.py

Streamlit page for the Medical Knowledge Assistant (Milestone 2).
Reuses the shared page-header component from components/, and the
same "cache the heavy resource, catch specific errors, never crash"
pattern already established by Milestone 1's sentiment analyzer.
"""

from __future__ import annotations

import uuid

import streamlit as st

from components import render_page_header
from utils.llm_client import LLMConfigurationError, LLMRequestError
from utils.logger import get_logger
from utils.storage import save_medical_query

from .loader import DatasetLoadError
from .rag_pipeline import MedicalRAGPipeline
from .vector_store import VectorStoreError

logger = get_logger(__name__)

DISCLAIMER = (
    "⚠️ **Educational purposes only.** This assistant does not provide "
    "professional medical advice, diagnosis, or treatment. Always consult "
    "a qualified healthcare provider for medical concerns."
)


@st.cache_resource(show_spinner=False)
def _get_pipeline() -> MedicalRAGPipeline:
    """Cached across reruns: dataset loading, embedding generation, and
    index building happen once per server process, not once per question."""
    pipeline = MedicalRAGPipeline()
    pipeline.initialize()
    return pipeline


def render_medical_chat_page() -> None:
    render_page_header(
        "🏥 Medical Knowledge Assistant",
        "Ask a medical question; answers are grounded in the MedQuAD medical "
        "knowledge base via retrieval-augmented generation (RAG).",
    )

    if "medical_session_id" not in st.session_state:
        st.session_state.medical_session_id = str(uuid.uuid4())
    if "medical_history" not in st.session_state:
        st.session_state.medical_history = []

    try:
        with st.spinner("Loading medical knowledge base (first run may take a moment)..."):
            pipeline = _get_pipeline()
    except DatasetLoadError as exc:
        st.error(f"⚠️ Medical dataset unavailable: {exc}")
        st.caption(DISCLAIMER)
        return
    except VectorStoreError as exc:
        st.error(f"⚠️ Medical knowledge base could not be prepared: {exc}")
        st.caption(DISCLAIMER)
        return
    except Exception as exc:  # noqa: BLE001 - never let the page crash
        logger.error("Unexpected error initializing medical RAG pipeline: %s", exc)
        st.error("⚠️ The medical assistant is temporarily unavailable. Please try again later.")
        st.caption(DISCLAIMER)
        return

    st.caption(
        f"Knowledge base ready: {pipeline.index_size:,} indexed passages "
        f"(embedding backend: {pipeline.embedding_backend})."
    )

    for turn in st.session_state.medical_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander(f"📚 Retrieved sources ({len(turn['sources'])})"):
                    for src in turn["sources"]:
                        _render_source(src)

    question = st.chat_input("Ask a medical question, e.g. 'What are the symptoms of diabetes?'")
    if not question:
        st.caption(DISCLAIMER)
        return
    if not question.strip():
        st.warning("Please enter a medical question.")
        st.caption(DISCLAIMER)
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        answer_text = ""
        sources_payload: list[dict] = []
        try:
            with st.spinner("Searching medical knowledge base and generating an answer..."):
                result = pipeline.answer(question)
            answer_text = result.answer
            sources_payload = [
                {
                    "question": s.question, "source": s.source,
                    "url": s.url, "focus": s.focus, "score": s.score,
                }
                for s in result.sources
            ]
            st.write(answer_text)
            if not result.had_context:
                st.info("No closely matching information was found in the knowledge base for this question.")
            elif sources_payload:
                with st.expander(f"📚 Retrieved sources ({len(sources_payload)})"):
                    for src in sources_payload:
                        _render_source(src)
        except LLMConfigurationError:
            answer_text = (
                "⚠️ I can't reach the language model yet because no API key is "
                "configured for the selected provider. Set the appropriate key "
                "in your .env file to enable answers."
            )
            st.write(answer_text)
        except LLMRequestError as exc:
            logger.error("Medical RAG LLM request failed: %s", exc)
            answer_text = "⚠️ Sorry, I ran into a problem generating an answer. Please try again shortly."
            st.write(answer_text)
        except ValueError as exc:
            st.warning(str(exc))
            st.caption(DISCLAIMER)
            return

    st.session_state.medical_history.append(
        {"question": question, "answer": answer_text, "sources": sources_payload}
    )

    try:
        save_medical_query(
            session_id=st.session_state.medical_session_id,
            question=question,
            answer=answer_text,
            sources=sources_payload,
        )
    except Exception as exc:  # noqa: BLE001 - PRD error handling: database failure
        logger.error("Failed to save medical query: %s", exc)
        st.warning("⚠️ Couldn't save this exchange to history, but you can keep asking questions.")

    st.caption(DISCLAIMER)


def _render_source(src: dict) -> None:
    focus = f" ({src['focus']})" if src.get("focus") else ""
    st.markdown(f"**{src['source']}{focus}** — _{src['question']}_")
    if src.get("url"):
        st.caption(src["url"])
    st.caption(f"Relevance score: {src['score']:.2f}")
