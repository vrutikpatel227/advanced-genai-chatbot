"""
tests/test_research.py

Unit tests for the Research Assistant (Milestone 4). Uses small
synthetic PDFs (generated with reportlab) and a temporary SQLite
database/vector store path per test -- no network access or heavy
model downloads required (the hashing fallback embedder is stateless
and always available offline).
"""

from __future__ import annotations

import dataclasses
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as root_config  # noqa: E402
from modules.knowledge_base.config import KnowledgeBaseConfig  # noqa: E402
from modules.research.chunker import chunk_paper  # noqa: E402
from modules.research.citation import build_citations  # noqa: E402
from modules.research.config import ResearchConfig  # noqa: E402
from modules.research.manager import ResearchManager  # noqa: E402
from modules.research.parser import InvalidPaperError, validate_paper  # noqa: E402
from modules.research.research_pipeline import ResearchRAGPipeline, build_research_prompt  # noqa: E402
from modules.research.retriever import ResearchRetriever, RetrievedPassage  # noqa: E402
from modules.research.summarizer import summarize_paper  # noqa: E402
from modules.research.vector_store import ResearchVectorStore  # noqa: E402
from utils import storage  # noqa: E402
from utils.llm_client import LLMConfigurationError  # noqa: E402


def _make_pdf(title: str, body: str) -> bytes:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, title)
    c.drawString(100, 720, body)
    c.save()
    return buf.getvalue()


def _make_research_config(tmp_path: Path, **kb_overrides) -> ResearchConfig:
    kb = KnowledgeBaseConfig(
        documents_dir=tmp_path / "documents",
        vector_store_dir=tmp_path / "vector_store",
        max_file_size_mb=1,
        hashing_dimension=64,
    )
    kb = dataclasses.replace(kb, **kb_overrides)
    return ResearchConfig(kb=kb, max_summary_chunks=10, max_summary_context_chars=4000)


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Point the shared SQLite DB at a temp file for every test here."""
    test_db_path = tmp_path / "research_test.db"
    patched_paths = dataclasses.replace(root_config.paths_config, sqlite_path=test_db_path)
    monkeypatch.setattr(root_config, "paths_config", patched_paths)
    monkeypatch.setattr(storage, "paths_config", patched_paths)
    storage.init_db()
    yield


# --- parser.py -----------------------------------------------------------------


def test_validate_paper_accepts_pdf(tmp_path):
    cfg = _make_research_config(tmp_path)
    assert validate_paper("paper.pdf", b"fake bytes", cfg) == ".pdf"


def test_validate_paper_rejects_non_pdf(tmp_path):
    cfg = _make_research_config(tmp_path)
    with pytest.raises(InvalidPaperError):
        validate_paper("paper.txt", b"fake bytes", cfg)


def test_validate_paper_rejects_empty_file(tmp_path):
    cfg = _make_research_config(tmp_path)
    with pytest.raises(InvalidPaperError):
        validate_paper("paper.pdf", b"", cfg)


def test_validate_paper_rejects_oversized_file(tmp_path):
    kb = KnowledgeBaseConfig(
        documents_dir=tmp_path / "documents", vector_store_dir=tmp_path / "vs", max_file_size_mb=1,
    )
    cfg = ResearchConfig(kb=kb, max_summary_chunks=10, max_summary_context_chars=4000)
    too_big = b"x" * (2 * 1024 * 1024)
    with pytest.raises(InvalidPaperError):
        validate_paper("paper.pdf", too_big, cfg)


# --- chunker.py --------------------------------------------------------------


def test_chunk_paper_preserves_metadata(tmp_path):
    cfg = _make_research_config(tmp_path)
    chunks = chunk_paper("paper-1", "study.pdf", "This paper studies neural networks.", cfg)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "paper-1"
    assert chunks[0].filename == "study.pdf"


def test_chunk_paper_handles_empty_text(tmp_path):
    cfg = _make_research_config(tmp_path)
    assert chunk_paper("paper-1", "empty.pdf", "", cfg) == []


# --- vector_store.py (the new capability: per-paper deletion) -----------------------


def _sample_chunks():
    from modules.knowledge_base.chunker import KnowledgeChunk

    return [
        KnowledgeChunk(chunk_id="p1::0", text="Neural networks and deep learning research.", doc_id="p1", filename="paper1.pdf", chunk_index=0),
        KnowledgeChunk(chunk_id="p2::0", text="Reinforcement learning and game theory research.", doc_id="p2", filename="paper2.pdf", chunk_index=0),
    ]


def test_research_vector_store_inherits_incremental_add(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    added = store.add_chunks(_sample_chunks())
    assert added == 2
    assert store.manifest.chunk_count == 2


def test_research_vector_store_delete_document_removes_only_that_papers_chunks(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())

    removed = store.delete_document("p1")
    assert removed == 1
    assert store.manifest.chunk_count == 1
    remaining_doc_ids = {chunk.doc_id for chunk in store.chunks.values()}
    assert remaining_doc_ids == {"p2"}


def test_research_vector_store_delete_document_is_safe_when_nothing_to_delete(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())
    removed = store.delete_document("nonexistent-doc")
    assert removed == 0
    assert store.manifest.chunk_count == 2  # untouched


def test_research_vector_store_delete_then_search_excludes_deleted_paper(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())
    store.delete_document("p1")

    results = store.search("neural networks deep learning", top_k=5)
    result_doc_ids = {chunk.doc_id for chunk, _score in results}
    assert "p1" not in result_doc_ids


def test_get_chunks_for_document_returns_only_that_papers_chunks(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())

    chunks = store.get_chunks_for_document("p2")
    assert len(chunks) == 1
    assert chunks[0].doc_id == "p2"


# --- retriever.py --------------------------------------------------------------


def test_retriever_empty_query_returns_empty(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())
    retriever = ResearchRetriever(store, cfg)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_retrieve_from_paper_returns_all_chunks(tmp_path):
    cfg = _make_research_config(tmp_path)
    store = ResearchVectorStore(cfg.kb)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())
    retriever = ResearchRetriever(store, cfg)

    passages = retriever.retrieve_from_paper("p2")
    assert len(passages) == 1
    assert passages[0].chunk.doc_id == "p2"


# --- citation.py --------------------------------------------------------------


def test_build_citations_deduplicates_by_chunk_id():
    chunk = _sample_chunks()[0]
    retrieved = [RetrievedPassage(chunk=chunk, score=0.5), RetrievedPassage(chunk=chunk, score=0.5)]
    citations = build_citations(retrieved)
    assert len(citations) == 1


def test_build_citations_never_fabricates():
    retrieved = [RetrievedPassage(chunk=c, score=0.9) for c in _sample_chunks()]
    citations = build_citations(retrieved)
    assert len(citations) == 2
    assert {c.filename for c in citations} == {"paper1.pdf", "paper2.pdf"}


# --- summarizer.py -------------------------------------------------------------


def test_summarize_paper_raises_value_error_for_no_chunks(tmp_path):
    cfg = _make_research_config(tmp_path)
    with pytest.raises(ValueError):
        summarize_paper("empty.pdf", [], cfg)


def test_summarize_paper_returns_structured_summary(tmp_path):
    cfg = _make_research_config(tmp_path)
    chunks = _sample_chunks()
    with patch(
        "modules.research.summarizer.get_chat_completion",
        return_value="Executive Summary: ...\nResearch Objective: ...",
    ):
        summary = summarize_paper("paper1.pdf", chunks, cfg)
    assert "Executive Summary" in summary.text
    assert summary.chunk_count_used == len(chunks)


# --- manager.py: full workflow including delete/reindex -----------------------------


def test_manager_process_upload_success(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    result = manager.process_upload("paper.pdf", _make_pdf("Title", "Some research content about AI."))
    assert result.status == "success"
    assert result.chunk_count == 1


def test_manager_process_upload_duplicate_detection(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    pdf_bytes = _make_pdf("Title", "Some research content about AI.")
    first = manager.process_upload("paper.pdf", pdf_bytes)
    second = manager.process_upload("paper_copy.pdf", pdf_bytes)
    assert first.status == "success"
    assert second.status == "duplicate"


def test_manager_process_upload_rejects_non_pdf(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    result = manager.process_upload("notes.txt", b"some text")
    assert result.status == "error"
    assert "PDF" in result.message


def test_manager_delete_paper_does_not_affect_other_papers(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    r1 = manager.process_upload("a.pdf", _make_pdf("Paper A", "Content about neural networks."))
    r2 = manager.process_upload("b.pdf", _make_pdf("Paper B", "Content about reinforcement learning."))

    deleted = manager.delete_paper(r1.doc_id)
    assert deleted is True

    stats = manager.get_stats()
    assert stats.total_papers == 1
    assert stats.total_chunks == 1

    remaining = manager.list_papers()
    assert len(remaining) == 1
    assert remaining[0].doc_id == r2.doc_id


def test_manager_delete_nonexistent_paper_returns_false(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    assert manager.delete_paper("nonexistent-id") is False


def test_manager_reindex_paper_updates_chunk_count(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    result = manager.process_upload("a.pdf", _make_pdf("Paper A", "Content about neural networks."))

    new_count = manager.reindex_paper(result.doc_id)
    assert new_count >= 1

    papers = manager.list_papers()
    assert papers[0].chunk_count == new_count


def test_manager_reindex_nonexistent_paper_raises(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    with pytest.raises(ValueError):
        manager.reindex_paper("nonexistent-id")


def test_manager_summarize_nonexistent_paper_raises(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    with pytest.raises(ValueError):
        manager.summarize("nonexistent-id")


def test_manager_summarize_uses_all_papers_chunks(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    result = manager.process_upload("a.pdf", _make_pdf("Paper A", "Content about neural networks."))

    with patch("modules.research.summarizer.get_chat_completion", return_value="Executive Summary: test."):
        summary = manager.summarize(result.doc_id)
    assert summary.chunk_count_used >= 1


def test_manager_retrieve_after_upload_is_immediate(tmp_path):
    base = _make_research_config(tmp_path)
    cfg = ResearchConfig(
        kb=dataclasses.replace(base.kb, min_similarity=0.0),
        max_summary_chunks=base.max_summary_chunks,
        max_summary_context_chars=base.max_summary_context_chars,
    )
    manager = ResearchManager(cfg)
    manager.process_upload("a.pdf", _make_pdf("Paper A", "Content about neural networks and deep learning."))
    results = manager.retrieve("neural networks")
    assert len(results) >= 1


# --- research_pipeline.py -----------------------------------------------------------


def test_build_research_prompt_includes_grounding_instruction():
    chunk = _sample_chunks()[0]
    retrieved = [RetrievedPassage(chunk=chunk, score=0.9)]
    messages = build_research_prompt("What did the paper find?", retrieved)
    assert messages[0].role == "system"
    assert "ONLY" in messages[0].content
    assert messages[1].role == "user"


def test_build_research_prompt_handles_no_context():
    messages = build_research_prompt("Some question", [])
    assert "No relevant passages" in messages[1].content


def test_research_pipeline_raises_value_error_for_empty_question(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    pipeline = ResearchRAGPipeline(manager)
    with pytest.raises(ValueError):
        pipeline.answer("   ")


def test_research_pipeline_propagates_llm_configuration_error(tmp_path):
    cfg = _make_research_config(tmp_path)
    manager = ResearchManager(cfg)
    manager.process_upload("a.pdf", _make_pdf("Paper A", "Content about neural networks."))
    pipeline = ResearchRAGPipeline(manager)

    with patch(
        "modules.research.research_pipeline.get_chat_completion",
        side_effect=LLMConfigurationError("no key set"),
    ):
        with pytest.raises(LLMConfigurationError):
            pipeline.answer("What is this paper about?")


def test_research_pipeline_returns_grounded_answer_with_citations(tmp_path):
    base = _make_research_config(tmp_path)
    cfg = ResearchConfig(
        kb=dataclasses.replace(base.kb, min_similarity=0.0),
        max_summary_chunks=10, max_summary_context_chars=4000,
    )
    manager = ResearchManager(cfg)
    manager.process_upload("a.pdf", _make_pdf("Paper A", "Content about neural networks and deep learning."))
    pipeline = ResearchRAGPipeline(manager)

    with patch(
        "modules.research.research_pipeline.get_chat_completion",
        return_value="The paper discusses neural networks.",
    ):
        result = pipeline.answer("What is this paper about?")

    assert result.answer == "The paper discusses neural networks."
    assert result.had_context is True
    assert len(result.citations) >= 1
