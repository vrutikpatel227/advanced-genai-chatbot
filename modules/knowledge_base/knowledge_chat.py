"""
modules/knowledge_base/knowledge_chat.py

Streamlit page for the Dynamic Knowledge Base (Milestone 3): upload
documents, view stats, search what's indexed, and manage the index.
Reuses the shared page-header component and the same
"cache the heavy resource, catch specific errors, never crash" pattern
established in Milestones 1-2.
"""

from __future__ import annotations

import streamlit as st

from components import render_page_header
from utils.logger import get_logger

from .manager import KnowledgeBaseManager

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def _get_manager() -> KnowledgeBaseManager:
    return KnowledgeBaseManager()


def render_knowledge_base_page() -> None:
    render_page_header(
        "📚 Dynamic Knowledge Base",
        "Upload documents (PDF, TXT, Markdown) to expand the assistant's "
        "knowledge. New documents are indexed incrementally -- no full "
        "rebuild needed.",
    )

    manager = _get_manager()

    _render_upload_section(manager)
    st.divider()
    _render_stats_section(manager)
    st.divider()
    _render_search_section(manager)
    st.divider()
    _render_document_list_section(manager)
    st.divider()
    _render_index_controls_section(manager)


def _render_upload_section(manager: KnowledgeBaseManager) -> None:
    st.subheader("📤 Upload Document")
    st.caption("Supported formats: PDF, TXT, Markdown (.md)")

    uploaded_file = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "md"], label_visibility="collapsed",
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


def _render_stats_section(manager: KnowledgeBaseManager) -> None:
    st.subheader("📊 Knowledge Statistics")
    try:
        stats = manager.get_stats()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load knowledge base stats: %s", exc)
        st.error("⚠️ Couldn't load knowledge base statistics right now.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", stats.total_documents)
    col2.metric("Total Chunks", stats.total_chunks)
    last_update = stats.last_update[:19].replace("T", " ") if stats.last_update else "Never"
    col3.metric("Last Update", last_update)
    col4.metric("Vector DB Status", "🟢 Ready" if stats.vector_store_ready else "⚪ Empty")
    if stats.vector_store_ready:
        st.caption(f"Embedding backend: {manager.embedding_backend}")


def _render_search_section(manager: KnowledgeBaseManager) -> None:
    st.subheader("🔍 Search Knowledge Base")
    st.caption("Every newly indexed document becomes searchable immediately.")
    query = st.text_input(
        "Search indexed documents",
        placeholder="e.g. 'refund policy', 'API rate limits'...",
        label_visibility="collapsed",
    )
    if not query:
        return

    try:
        results = manager.search(query)
    except Exception as exc:  # noqa: BLE001
        logger.error("Knowledge base search failed: %s", exc)
        st.error("⚠️ Search failed. Please try again.")
        return

    if not results:
        st.info("No closely matching content found in the knowledge base.")
        return

    for item in results:
        with st.expander(f"📄 {item.chunk.filename} (relevance {item.score:.2f})"):
            st.write(item.chunk.text)


def _render_document_list_section(manager: KnowledgeBaseManager) -> None:
    st.subheader("📄 Indexed Documents")
    documents = manager.list_documents()
    if not documents:
        st.caption("No documents indexed yet. Upload one above to get started.")
        return

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "File Name": doc.filename,
                "Type": doc.file_type,
                "Upload Date": doc.created_at[:19].replace("T", " "),
                "Chunks": doc.chunk_count,
                "Status": doc.status,
            }
            for doc in documents
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_index_controls_section(manager: KnowledgeBaseManager) -> None:
    st.subheader("⚙️ Index Controls")
    col1, col2, col3 = st.columns(3)

    if col1.button("🔄 Update Index", use_container_width=True):
        with st.spinner("Checking for pending documents..."):
            try:
                updated = manager.update_index()
            except Exception as exc:  # noqa: BLE001
                logger.error("Update index failed: %s", exc)
                st.error("⚠️ Failed to update the index. Please try again.")
            else:
                if updated:
                    st.success(f"✅ Updated {updated} pending document(s).")
                else:
                    st.info("Nothing to update -- everything is already indexed.")

    if col2.button("🏗️ Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding the entire index... this may take a moment."):
            try:
                rebuilt = manager.rebuild_index()
            except Exception as exc:  # noqa: BLE001
                logger.error("Rebuild index failed: %s", exc)
                st.error("⚠️ Failed to rebuild the index. Please try again.")
            else:
                st.success(f"✅ Rebuilt the index from {rebuilt} document(s).")

    if col3.button("🔃 Refresh Statistics", use_container_width=True):
        st.rerun()
