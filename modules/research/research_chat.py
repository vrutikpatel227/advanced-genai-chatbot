"""
modules/research/research_chat.py

Streamlit page for the Research Assistant (Milestone 4): upload
papers, ask grounded questions with citations, generate structured
summaries, and manage indexed papers (list/delete/re-index). Reuses
the shared page-header component and the same "cache the heavy
resource, catch specific errors, never crash" pattern established in
Milestones 1-3.
"""

from __future__ import annotations

import streamlit as st

from components import render_page_header
from utils.llm_client import LLMConfigurationError, LLMRequestError
from utils.logger import get_logger

from .manager import ResearchManager
from .research_pipeline import ResearchRAGPipeline

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def _get_manager() -> ResearchManager:
    return ResearchManager()


def render_research_assistant_page() -> None:
    render_page_header(
        "📄 Research Assistant",
        "Upload research papers (PDF); ask grounded questions with "
        "citations, generate structured summaries, and manage your "
        "indexed papers.",
    )

    manager = _get_manager()
    pipeline = ResearchRAGPipeline(manager)

    _render_upload_section(manager)
    st.divider()
    _render_qa_section(pipeline)
    st.divider()
    _render_summary_section(manager)
    st.divider()
    _render_management_section(manager)


def _render_upload_section(manager: ResearchManager) -> None:
    st.subheader("📤 Upload Research Paper")
    st.caption("Supported format: PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"], label_visibility="collapsed",
    )
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    with st.spinner(f"Processing '{uploaded_file.name}'..."):
        try:
            result = manager.process_upload(uploaded_file.name, file_bytes)
        except Exception as exc:  # noqa: BLE001 - never crash the page
            logger.error("Unexpected error processing upload '%s': %s", uploaded_file.name, exc)
            st.error(f"⚠️ Something went wrong processing '{uploaded_file.name}'. Please try again.")
            return

    if result.status == "success":
        st.success(f"✅ {result.message}")
    elif result.status == "duplicate":
        st.info(f"ℹ️ {result.message}")
    else:
        st.error(f"⚠️ {result.message}")


def _render_qa_section(pipeline: ResearchRAGPipeline) -> None:
    st.subheader("❓ Question Answering")
    question = st.text_input(
        "Ask a question about your uploaded papers",
        placeholder="e.g. 'What methodology did the authors use?'",
    )
    ask_clicked = st.button("Ask")

    if not ask_clicked:
        return
    if not question or not question.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("Retrieving relevant passages and generating an answer..."):
        try:
            result = pipeline.answer(question)
        except LLMConfigurationError:
            st.warning(
                "⚠️ No API key is configured for the selected LLM provider. "
                "Set it in your .env file to enable answers."
            )
            return
        except LLMRequestError as exc:
            logger.error("Research QA failed: %s", exc)
            st.error("⚠️ Sorry, I ran into a problem generating an answer. Please try again shortly.")
            return
        except ValueError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never crash the page
            logger.error("Unexpected error during research QA: %s", exc)
            st.error("⚠️ Something went wrong answering that question. Please try again.")
            return

    st.markdown("**Answer**")
    st.write(result.answer)

    if not result.had_context:
        st.info("No relevant passages were found in the indexed papers for this question.")
    elif result.citations:
        st.markdown("**Sources**")
        for citation in result.citations:
            with st.expander(f"📄 {citation.filename} (similarity {citation.similarity_score:.2f})"):
                st.write(citation.chunk_text)


def _render_summary_section(manager: ResearchManager) -> None:
    st.subheader("📝 Paper Summary")
    papers = manager.list_papers()
    if not papers:
        st.caption("No papers indexed yet. Upload one above to summarize it.")
        return

    options = {paper.filename: paper.doc_id for paper in papers}
    selected_label = st.selectbox("Select a paper to summarize", list(options.keys()))

    if not st.button("Generate Summary"):
        return

    doc_id = options[selected_label]
    with st.spinner(f"Summarizing '{selected_label}'..."):
        try:
            summary = manager.summarize(doc_id)
        except LLMConfigurationError:
            st.warning(
                "⚠️ No API key is configured for the selected LLM provider. "
                "Set it in your .env file to enable summaries."
            )
            return
        except LLMRequestError as exc:
            logger.error("Summary generation failed: %s", exc)
            st.error("⚠️ Sorry, I ran into a problem generating the summary. Please try again shortly.")
            return
        except ValueError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never crash the page
            logger.error("Unexpected error generating summary: %s", exc)
            st.error("⚠️ Something went wrong generating the summary. Please try again.")
            return

    st.markdown(summary.text)


def _render_management_section(manager: ResearchManager) -> None:
    st.subheader("📚 Paper Management")

    try:
        stats = manager.get_stats()
        st.caption(f"{stats.total_papers} paper(s), {stats.total_chunks} chunk(s) indexed.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load research stats: %s", exc)

    papers = manager.list_papers()
    if not papers:
        st.caption("No papers indexed yet.")
        return

    header_cols = st.columns([3, 2, 1, 1, 1, 1])
    for col, label in zip(header_cols, ["File Name", "Upload Date", "Chunks", "Status", "", ""]):
        col.markdown(f"**{label}**")

    for paper in papers:
        col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 1, 1, 1, 1])
        col1.write(paper.filename)
        col2.write(paper.created_at[:19].replace("T", " "))
        col3.write(str(paper.chunk_count))
        col4.write(paper.status)

        if col5.button("🗑️", key=f"delete_{paper.doc_id}", help="Delete this paper"):
            try:
                manager.delete_paper(paper.doc_id)
                st.success(f"Deleted '{paper.filename}'.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to delete paper %s: %s", paper.doc_id, exc)
                st.error(f"⚠️ Failed to delete '{paper.filename}'. Please try again.")

        if col6.button("🔄", key=f"reindex_{paper.doc_id}", help="Re-index this paper"):
            with st.spinner(f"Re-indexing '{paper.filename}'..."):
                try:
                    new_count = manager.reindex_paper(paper.doc_id)
                    st.success(f"Re-indexed '{paper.filename}' ({new_count} chunks).")
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to reindex paper %s: %s", paper.doc_id, exc)
                    st.error(f"⚠️ Failed to re-index '{paper.filename}'. Please try again.")

    if st.button("🔃 Refresh"):
        st.rerun()
