"""
tests/test_knowledge_base.py

Unit tests for the Dynamic Knowledge Base (Milestone 3). Uses small
synthetic documents and a temporary SQLite database/vector store path
per test -- no network access or heavy model downloads required (the
hashing fallback embedder is stateless and always available offline).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as root_config  # noqa: E402
from modules.knowledge_base.config import KnowledgeBaseConfig  # noqa: E402
from modules.knowledge_base.embeddings import (  # noqa: E402
    HashingEmbedder,
    KnowledgeEmbeddingGenerator,
)
from modules.knowledge_base.chunker import chunk_document  # noqa: E402
from modules.knowledge_base.manager import KnowledgeBaseManager  # noqa: E402
from modules.knowledge_base.parser import (  # noqa: E402
    InvalidFileError,
    ParsingError,
    extract_text,
    validate_file,
)
from modules.knowledge_base.vector_store import KnowledgeVectorStore  # noqa: E402
from utils import storage  # noqa: E402


def _make_kb_config(tmp_path: Path, **overrides) -> KnowledgeBaseConfig:
    base = KnowledgeBaseConfig(
        documents_dir=tmp_path / "documents",
        vector_store_dir=tmp_path / "vector_store",
        max_file_size_mb=1,
    )
    return dataclasses.replace(base, **overrides)


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Point the shared SQLite DB at a temp file for every test here."""
    test_db_path = tmp_path / "kb_test.db"
    patched_paths = dataclasses.replace(root_config.paths_config, sqlite_path=test_db_path)
    monkeypatch.setattr(root_config, "paths_config", patched_paths)
    monkeypatch.setattr(storage, "paths_config", patched_paths)
    storage.init_db()
    yield


# --- parser.py -----------------------------------------------------------------


def test_validate_file_accepts_supported_extensions(tmp_path):
    cfg = _make_kb_config(tmp_path)
    assert validate_file("notes.txt", b"hello", cfg) == ".txt"
    assert validate_file("notes.md", b"hello", cfg) == ".md"
    assert validate_file("notes.pdf", b"hello", cfg) == ".pdf"


def test_validate_file_rejects_unsupported_extension(tmp_path):
    cfg = _make_kb_config(tmp_path)
    with pytest.raises(InvalidFileError):
        validate_file("image.png", b"fake", cfg)


def test_validate_file_rejects_empty_file(tmp_path):
    cfg = _make_kb_config(tmp_path)
    with pytest.raises(InvalidFileError):
        validate_file("empty.txt", b"", cfg)


def test_validate_file_rejects_oversized_file(tmp_path):
    cfg = _make_kb_config(tmp_path, max_file_size_mb=1)
    too_big = b"x" * (2 * 1024 * 1024)
    with pytest.raises(InvalidFileError):
        validate_file("big.txt", too_big, cfg)


def test_extract_text_txt():
    assert extract_text("a.txt", b"hello world", ".txt") == "hello world"


def test_extract_text_markdown():
    text = extract_text("a.md", b"# Title\n\nSome content.", ".md")
    assert "Title" in text and "Some content" in text


def test_extract_text_corrupted_pdf_raises_parsing_error():
    with pytest.raises(ParsingError):
        extract_text("bad.pdf", b"not a real pdf", ".pdf")


# --- chunker.py --------------------------------------------------------------


def test_chunk_document_preserves_metadata(tmp_path):
    cfg = _make_kb_config(tmp_path)
    chunks = chunk_document("doc-1", "policy.txt", "This is a short policy document.", cfg)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].filename == "policy.txt"
    assert chunks[0].chunk_index == 0


def test_chunk_document_splits_long_text(tmp_path):
    cfg = _make_kb_config(tmp_path, chunk_size=100, chunk_overlap=10)
    long_text = "This is a sentence about company policy. " * 20
    chunks = chunk_document("doc-1", "policy.txt", long_text, cfg)
    assert len(chunks) > 1
    assert all(c.doc_id == "doc-1" for c in chunks)


def test_chunk_document_handles_empty_text(tmp_path):
    cfg = _make_kb_config(tmp_path)
    assert chunk_document("doc-1", "empty.txt", "", cfg) == []


# --- embeddings.py -----------------------------------------------------------


def test_hashing_embedder_is_stateless_and_incremental():
    embedder = HashingEmbedder(n_features=64)
    embedder.fit([])  # no corpus needed
    v1 = embedder.embed(["first document about refunds"])
    v2 = embedder.embed(["a completely different second document about shipping"])
    assert v1.shape == (1, 64)
    assert v2.shape == (1, 64)


def test_knowledge_embedding_generator_falls_back_to_hashing():
    generator = KnowledgeEmbeddingGenerator(_make_kb_config(Path("/tmp")))
    with patch(
        "modules.knowledge_base.embeddings.SentenceTransformerEmbedder.fit",
        side_effect=RuntimeError("simulated: no network access"),
    ):
        embedder = generator.select_backend(["some text"])
    assert embedder.name == "hashing-fallback"


# --- vector_store.py ---------------------------------------------------------


def _sample_chunks():
    from modules.knowledge_base.chunker import KnowledgeChunk

    return [
        KnowledgeChunk(chunk_id="d1::0", text="Refunds are accepted within 30 days.", doc_id="d1", filename="refunds.txt", chunk_index=0),
        KnowledgeChunk(chunk_id="d2::0", text="API rate limit is 100 requests per minute.", doc_id="d2", filename="api.md", chunk_index=0),
    ]


def test_vector_store_incremental_add_and_search(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    store = KnowledgeVectorStore(cfg)
    store.ensure_ready()
    added = store.add_chunks(_sample_chunks())
    assert added == 2

    results = store.search("refund policy", top_k=2)
    assert len(results) >= 1


def test_vector_store_add_chunks_does_not_touch_existing_vectors(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    store = KnowledgeVectorStore(cfg)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())
    assert store.manifest.chunk_count == 2

    from modules.knowledge_base.chunker import KnowledgeChunk
    new_chunk = [KnowledgeChunk(chunk_id="d3::0", text="Shipping takes 3-5 days.", doc_id="d3", filename="shipping.txt", chunk_index=0)]
    store.add_chunks(new_chunk)
    assert store.manifest.chunk_count == 3  # incremental, not a rebuild


def test_vector_store_persists_and_reloads(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    store1 = KnowledgeVectorStore(cfg)
    store1.ensure_ready()
    store1.add_chunks(_sample_chunks())

    store2 = KnowledgeVectorStore(cfg)
    store2.load()
    assert store2.manifest.chunk_count == 2
    results = store2.search("api rate limit", top_k=2)
    assert len(results) >= 1


def test_vector_store_rebuild_replaces_everything(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    store = KnowledgeVectorStore(cfg)
    store.ensure_ready()
    store.add_chunks(_sample_chunks())

    from modules.knowledge_base.chunker import KnowledgeChunk
    rebuild_chunks = [KnowledgeChunk(chunk_id="d9::0", text="Only one chunk after rebuild.", doc_id="d9", filename="only.txt", chunk_index=0)]
    store.rebuild(rebuild_chunks)
    assert store.manifest.chunk_count == 1


def test_vector_store_search_returns_empty_before_ready(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    store = KnowledgeVectorStore(cfg)
    assert store.search("anything", top_k=2) == []


# --- manager.py: full upload workflow -----------------------------------------


def test_manager_process_upload_success(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    result = manager.process_upload("refunds.txt", b"Refunds are accepted within 30 days of purchase.")
    assert result.status == "success"
    assert result.chunk_count == 1


def test_manager_process_upload_duplicate_detection(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    content = b"Refunds are accepted within 30 days of purchase."
    first = manager.process_upload("refunds.txt", content)
    second = manager.process_upload("refunds_copy.txt", content)
    assert first.status == "success"
    assert second.status == "duplicate"


def test_manager_process_upload_invalid_format(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    result = manager.process_upload("image.png", b"fake bytes")
    assert result.status == "error"
    assert "Unsupported" in result.message


def test_manager_process_upload_empty_file(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    result = manager.process_upload("empty.txt", b"")
    assert result.status == "error"


def test_manager_search_after_upload_is_immediate(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64, min_similarity=0.0)
    manager = KnowledgeBaseManager(cfg)
    manager.process_upload("refunds.txt", b"Refunds are accepted within 30 days of purchase.")
    results = manager.search("refund policy")
    assert len(results) >= 1


def test_manager_search_empty_query_returns_empty(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    assert manager.search("") == []
    assert manager.search("   ") == []


def test_manager_get_stats_reflects_uploads(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    manager.process_upload("a.txt", b"Document A content about refunds.")
    manager.process_upload("b.txt", b"Document B content about shipping times.")
    stats = manager.get_stats()
    assert stats.total_documents == 2
    assert stats.total_chunks == 2
    assert stats.vector_store_ready is True


def test_manager_list_documents(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    manager.process_upload("a.txt", b"Document A content.")
    docs = manager.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "a.txt"
    assert docs[0].status == "indexed"


def test_manager_update_index_noop_when_nothing_pending(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    manager.process_upload("a.txt", b"Document A content.")
    updated = manager.update_index()
    assert updated == 0  # already indexed on upload


def test_manager_rebuild_index_preserves_searchability(tmp_path):
    cfg = _make_kb_config(tmp_path, hashing_dimension=64, min_similarity=0.0)
    manager = KnowledgeBaseManager(cfg)
    manager.process_upload("refunds.txt", b"Refunds are accepted within 30 days of purchase.")
    rebuilt = manager.rebuild_index()
    assert rebuilt == 1
    results = manager.search("refund policy")
    assert len(results) >= 1


def test_manager_never_crashes_on_unexpected_chunking_error(tmp_path):
    """Even if chunking raises unexpectedly, process_upload must not
    propagate the exception (PRD: never crash the app)."""
    cfg = _make_kb_config(tmp_path, hashing_dimension=64)
    manager = KnowledgeBaseManager(cfg)
    with patch(
        "modules.knowledge_base.manager.chunk_document",
        side_effect=RuntimeError("simulated unexpected failure"),
    ):
        result = manager.process_upload("a.txt", b"some content")
    assert result.status == "error"
