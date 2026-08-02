"""
modules/medical/rag_pipeline.py

Top-level orchestration for the Medical Knowledge Assistant:

    question -> embed -> search FAISS -> top-K chunks -> LLM (via the
    existing configurable provider) -> grounded answer + sources

Deliberately reuses utils/llm_client.py's get_chat_completion() --
this module never imports Groq/OpenAI/Gemini SDKs directly, per the
PRD's "Use the existing configurable LLM Provider. Do NOT hardcode
Groq/OpenAI." requirement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from utils.llm_client import (
    LLMConfigurationError,
    LLMRequestError,
    get_chat_completion,
)
from utils.logger import get_logger

from .config import MedicalConfig, ensure_medical_directories, medical_config
from .loader import DatasetLoadError, MedQuADLoader
from .preprocess import chunk_qa_pairs
from .prompts import build_medical_prompt
from .retriever import MedicalRetriever, RetrievedChunk
from .vector_store import MedicalVectorStore, VectorStoreError

logger = get_logger(__name__)


@dataclass(frozen=True)
class MedicalSource:
    question: str
    source: str
    url: str
    focus: str
    score: float


@dataclass(frozen=True)
class MedicalAnswer:
    answer: str
    sources: list[MedicalSource]
    had_context: bool


class MedicalRAGPipeline:
    """Owns the full medical RAG lifecycle: initialize() loads/builds
    everything needed once; answer() is then cheap to call repeatedly."""

    def __init__(self, config: MedicalConfig | None = None) -> None:
        self._config = config or medical_config
        self._store = MedicalVectorStore(self._config)
        self._retriever: MedicalRetriever | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def index_size(self) -> int:
        return len(self._store.chunks) if self._store.chunks else 0

    @property
    def embedding_backend(self) -> str:
        return self._store.manifest.backend if self._store.manifest else "unknown"

    def initialize(self, force_rebuild: bool = False) -> None:
        """Load the dataset, chunk it, and build (or load a cached)
        vector index. Raises DatasetLoadError or VectorStoreError on
        failure -- callers (the UI) catch these and show a friendly
        message rather than crashing."""
        ensure_medical_directories()

        loader = MedQuADLoader(self._config)
        pairs = loader.load()  # may raise DatasetLoadError

        chunks = chunk_qa_pairs(pairs, self._config)
        if not chunks:
            raise DatasetLoadError(
                "The medical dataset loaded but produced no usable Q&A chunks."
            )

        self._store.build(chunks, force_rebuild=force_rebuild)  # may raise VectorStoreError
        self._retriever = MedicalRetriever(self._store, self._config)
        self._ready = True
        logger.info(
            "Medical RAG pipeline ready: %d chunks indexed, embedding backend=%s.",
            len(chunks), self._store.manifest.backend if self._store.manifest else "unknown",
        )

    def answer(self, question: str) -> MedicalAnswer:
        """Answer a medical question using retrieved context + the
        configured LLM provider. Raises ValueError for empty/invalid
        input, RuntimeError if the pipeline isn't initialized, and lets
        LLMConfigurationError/LLMRequestError propagate (the UI already
        knows how to handle these from Milestone 1)."""
        if not question or not question.strip():
            raise ValueError("Please enter a medical question.")
        if not self._ready or self._retriever is None:
            raise RuntimeError("Medical assistant is not initialized yet.")

        retrieved: list[RetrievedChunk] = self._retriever.retrieve(question)
        messages = build_medical_prompt(question, retrieved)

        start = time.monotonic()
        answer_text = get_chat_completion(messages)  # LLMConfigurationError/LLMRequestError propagate
        elapsed = time.monotonic() - start
        logger.info(
            "Medical RAG answer generated in %.2fs (%d retrieved chunks, had_context=%s).",
            elapsed, len(retrieved), bool(retrieved),
        )

        sources = self._deduplicate_sources(
            [
                MedicalSource(
                    question=item.chunk.question,
                    source=item.chunk.source,
                    url=item.chunk.url,
                    focus=item.chunk.focus,
                    score=item.score,
                )
                for item in retrieved
            ]
        )

        return MedicalAnswer(answer=answer_text, sources=sources, had_context=bool(retrieved))

    @staticmethod
    def _deduplicate_sources(sources: list[MedicalSource]) -> list[MedicalSource]:
        """Multiple retrieved chunks can come from the same underlying
        QA pair (split by chunking); collapse those for display."""
        seen: set[tuple[str, str]] = set()
        deduped: list[MedicalSource] = []
        for source in sources:
            key = (source.source, source.question)
            if key not in seen:
                seen.add(key)
                deduped.append(source)
        return deduped


__all__ = [
    "MedicalRAGPipeline",
    "MedicalAnswer",
    "MedicalSource",
    "DatasetLoadError",
    "VectorStoreError",
    "LLMConfigurationError",
    "LLMRequestError",
]
