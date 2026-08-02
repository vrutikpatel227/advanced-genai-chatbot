"""
tests/test_medical.py

Unit tests for the Medical Knowledge Assistant (Milestone 2). All
tests use small synthetic data (synthetic XML files, synthetic
MedicalQAPair/MedicalChunk instances) -- none require downloading the
full MedQuAD dataset or any network access, so the suite stays fast
and deterministic in CI.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.medical.config import MedicalConfig  # noqa: E402
from modules.medical.embeddings import (  # noqa: E402
    EmbeddingGenerator,
    TfidfFallbackEmbedder,
)
from modules.medical.loader import DatasetLoadError, MedQuADLoader, MedicalQAPair  # noqa: E402
from modules.medical.preprocess import chunk_qa_pairs, clean_text  # noqa: E402
from modules.medical.prompts import build_medical_prompt  # noqa: E402
from modules.medical.rag_pipeline import MedicalRAGPipeline  # noqa: E402
from modules.medical.retriever import MedicalRetriever, RetrievedChunk  # noqa: E402
from modules.medical.vector_store import MedicalVectorStore, VectorStoreError  # noqa: E402
from utils.llm_client import LLMConfigurationError  # noqa: E402

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000099" source="TestSource" url="http://example.com/test">
<Focus>Test Condition</Focus>
<QAPairs>
    <QAPair pid="1">
        <Question qid="0000099-1" qtype="symptoms">What are the symptoms of the test condition ?</Question>
        <Answer>Common symptoms include fatigue and mild fever.</Answer>
    </QAPair>
    <QAPair pid="2">
        <Question qid="0000099-2" qtype="treatment">What are the treatments for the test condition ?</Question>
        <Answer></Answer>
    </QAPair>
</QAPairs>
</Document>
"""

CORRUPTED_XML = "<Document><Unclosed>"


def _make_config(tmp_path: Path, **overrides) -> MedicalConfig:
    base = MedicalConfig(
        dataset_dir=tmp_path / "raw",
        dataset_cache_path=tmp_path / "cache.json",
        vector_store_dir=tmp_path / "vector_store",
        max_source_files=0,
    )
    return dataclasses.replace(base, **overrides)


# --- preprocess.py -----------------------------------------------------------------


def test_clean_text_normalizes_whitespace():
    assert clean_text("  a   b\n\nc  ") == "a b c"


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


def test_chunk_qa_pairs_preserves_metadata():
    pairs = [
        MedicalQAPair(
            doc_id="d1", pair_id="d1-1", question="What is X?", answer="X is a condition.",
            focus="X", source="TestSource", url="http://example.com",
        )
    ]
    chunks = chunk_qa_pairs(pairs, config=_make_config(Path("/tmp"), chunk_size=800, chunk_overlap=100))
    assert len(chunks) == 1
    assert chunks[0].source == "TestSource"
    assert chunks[0].focus == "X"
    assert chunks[0].url == "http://example.com"
    assert "What is X?" in chunks[0].text


def test_chunk_qa_pairs_splits_long_answers():
    long_answer = "This is a sentence about the condition. " * 100
    pairs = [
        MedicalQAPair(doc_id="d1", pair_id="d1-1", question="Q?", answer=long_answer,
                       focus="F", source="S", url="")
    ]
    cfg = _make_config(Path("/tmp"), chunk_size=200, chunk_overlap=20)
    chunks = chunk_qa_pairs(pairs, config=cfg)
    assert len(chunks) > 1
    assert all(c.pair_id == "d1-1" for c in chunks)


def test_chunk_qa_pairs_skips_empty_answers():
    pairs = [MedicalQAPair(doc_id="d1", pair_id="d1-1", question="Q?", answer="", focus="", source="S", url="")]
    chunks = chunk_qa_pairs(pairs, config=_make_config(Path("/tmp")))
    assert chunks == []


# --- loader.py ---------------------------------------------------------------------


def test_loader_parses_real_medquad_schema(tmp_path):
    (tmp_path / "raw" / "Folder1").mkdir(parents=True)
    (tmp_path / "raw" / "Folder1" / "0000099.xml").write_text(SAMPLE_XML)

    loader = MedQuADLoader(_make_config(tmp_path))
    pairs = loader.load()

    # The second QAPair has an empty <Answer/> (mirrors real MedQuAD's
    # copyright-redacted entries) and must be skipped.
    assert len(pairs) == 1
    assert pairs[0].question == "What are the symptoms of the test condition ?"
    assert pairs[0].source == "TestSource"
    assert pairs[0].focus == "Test Condition"


def test_loader_skips_corrupted_files_without_crashing(tmp_path):
    raw = tmp_path / "raw" / "Folder1"
    raw.mkdir(parents=True)
    (raw / "good.xml").write_text(SAMPLE_XML)
    (raw / "bad.xml").write_text(CORRUPTED_XML)

    loader = MedQuADLoader(_make_config(tmp_path))
    pairs = loader.load()  # must not raise despite the corrupted file
    assert len(pairs) == 1


def test_loader_raises_when_dataset_missing_and_download_fails(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, dataset_download_url="http://invalid.invalid/nonexistent.zip")
    loader = MedQuADLoader(cfg)
    with pytest.raises(DatasetLoadError):
        loader.load()


def test_loader_uses_cache_on_second_call(tmp_path):
    raw = tmp_path / "raw" / "Folder1"
    raw.mkdir(parents=True)
    (raw / "good.xml").write_text(SAMPLE_XML)

    cfg = _make_config(tmp_path)
    loader = MedQuADLoader(cfg)
    first = loader.load()
    assert cfg.dataset_cache_path.exists()

    # Even if the raw directory disappears, the cache should still work.
    import shutil
    shutil.rmtree(cfg.dataset_dir)
    second = MedQuADLoader(cfg).load()
    assert len(second) == len(first) == 1


# --- embeddings.py -----------------------------------------------------------------


def test_tfidf_embedder_produces_fixed_dimension_vectors():
    embedder = TfidfFallbackEmbedder(max_features=64)
    corpus = ["diabetes symptoms include thirst", "asthma causes airway inflammation", "unrelated text here"]
    embedder.fit(corpus)
    vectors = embedder.embed(corpus)
    assert vectors.shape == (3, 64)


def test_embedding_generator_falls_back_to_tfidf_when_transformer_unavailable():
    generator = EmbeddingGenerator(_make_config(Path("/tmp"), tfidf_max_features=32))
    with patch(
        "modules.medical.embeddings.SentenceTransformerEmbedder.fit",
        side_effect=RuntimeError("simulated: no network access to download model"),
    ):
        vectors, embedder = generator.build(["some text", "more text"])
    assert embedder.name == "tfidf-fallback"
    assert vectors.shape[0] == 2


# --- vector_store.py + retriever.py -------------------------------------------------


def _synthetic_chunks():
    from modules.medical.preprocess import MedicalChunk

    return [
        MedicalChunk(chunk_id="1::0", text="Question: diabetes symptoms\nAnswer: thirst and fatigue are common.",
                     question="diabetes symptoms", focus="Diabetes", source="TestSource", url="", doc_id="1", pair_id="1"),
        MedicalChunk(chunk_id="2::0", text="Question: asthma causes\nAnswer: airway inflammation triggers asthma.",
                     question="asthma causes", focus="Asthma", source="TestSource", url="", doc_id="2", pair_id="2"),
    ]


def test_vector_store_build_and_search_roundtrip(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    store = MedicalVectorStore(cfg)
    store.build(_synthetic_chunks())

    retriever = MedicalRetriever(store, cfg)
    results = retriever.retrieve("What causes diabetes thirst?", top_k=2)
    assert len(results) >= 1
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_vector_store_skips_rebuild_when_unchanged(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    chunks = _synthetic_chunks()

    store1 = MedicalVectorStore(cfg)
    store1.build(chunks)
    fingerprint1 = store1.manifest.fingerprint

    store2 = MedicalVectorStore(cfg)
    store2.build(chunks)  # should load cached index, not rebuild
    assert store2.manifest.fingerprint == fingerprint1
    assert store2.manifest.chunk_count == 2


def test_vector_store_rebuilds_when_chunks_change(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    store = MedicalVectorStore(cfg)
    store.build(_synthetic_chunks())
    original_count = store.manifest.chunk_count

    more_chunks = _synthetic_chunks()
    from modules.medical.preprocess import MedicalChunk
    more_chunks.append(
        MedicalChunk(chunk_id="3::0", text="Question: flu symptoms\nAnswer: fever and cough.",
                     question="flu symptoms", focus="Flu", source="TestSource", url="", doc_id="3", pair_id="3")
    )
    store2 = MedicalVectorStore(cfg)
    store2.build(more_chunks)
    assert store2.manifest.chunk_count == original_count + 1


def test_retriever_returns_empty_list_for_empty_query(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    store = MedicalVectorStore(cfg)
    store.build(_synthetic_chunks())
    retriever = MedicalRetriever(store, cfg)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_raises_if_store_not_ready(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    store = MedicalVectorStore(cfg)  # never built or loaded
    retriever = MedicalRetriever(store, cfg)
    with pytest.raises(VectorStoreError):
        retriever.retrieve("some question")


def test_retriever_filters_by_min_similarity(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64, min_similarity=0.99)
    store = MedicalVectorStore(cfg)
    store.build(_synthetic_chunks())
    retriever = MedicalRetriever(store, cfg)
    # An unrelated, oddly-worded query is very unlikely to hit a 0.99 threshold.
    results = retriever.retrieve("completely unrelated gardening question", top_k=2)
    assert results == []


# --- prompts.py --------------------------------------------------------------------


def test_build_medical_prompt_includes_context_and_disclaimer_instruction():
    chunk = _synthetic_chunks()[0]
    retrieved = [RetrievedChunk(chunk=chunk, score=0.9)]
    messages = build_medical_prompt("What causes diabetes?", retrieved)
    assert messages[0].role == "system"
    assert "ONLY" in messages[0].content
    assert messages[1].role == "user"
    assert "diabetes" in messages[1].content.lower()


def test_build_medical_prompt_handles_no_retrieved_context():
    messages = build_medical_prompt("Some question", [])
    assert "No relevant reference passages" in messages[1].content


# --- rag_pipeline.py ---------------------------------------------------------------


def test_rag_pipeline_raises_value_error_for_empty_question(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    pipeline = MedicalRAGPipeline(cfg)
    pipeline._store.build(_synthetic_chunks())
    from modules.medical.retriever import MedicalRetriever as _R
    pipeline._retriever = _R(pipeline._store, cfg)
    pipeline._ready = True

    with pytest.raises(ValueError):
        pipeline.answer("   ")


def test_rag_pipeline_raises_runtime_error_if_not_initialized(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    pipeline = MedicalRAGPipeline(cfg)
    with pytest.raises(RuntimeError):
        pipeline.answer("What are the symptoms of diabetes?")


def test_rag_pipeline_propagates_llm_configuration_error(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    pipeline = MedicalRAGPipeline(cfg)
    pipeline._store.build(_synthetic_chunks())
    from modules.medical.retriever import MedicalRetriever as _R
    pipeline._retriever = _R(pipeline._store, cfg)
    pipeline._ready = True

    with patch(
        "modules.medical.rag_pipeline.get_chat_completion",
        side_effect=LLMConfigurationError("no key set"),
    ):
        with pytest.raises(LLMConfigurationError):
            pipeline.answer("What are the symptoms of diabetes?")


def test_rag_pipeline_returns_grounded_answer_with_sources(tmp_path):
    cfg = _make_config(tmp_path, tfidf_max_features=64)
    pipeline = MedicalRAGPipeline(cfg)
    pipeline._store.build(_synthetic_chunks())
    from modules.medical.retriever import MedicalRetriever as _R
    pipeline._retriever = _R(pipeline._store, cfg)
    pipeline._ready = True

    with patch("modules.medical.rag_pipeline.get_chat_completion", return_value="Thirst and fatigue are common."):
        result = pipeline.answer("What are the symptoms of diabetes?")

    assert result.answer == "Thirst and fatigue are common."
    assert result.had_context is True
    assert len(result.sources) >= 1
