"""
modules/medical/retriever.py

Retrieval step of the RAG pipeline:
  question -> embed -> search FAISS -> top-K relevant chunks

Kept separate from rag_pipeline.py so retrieval can be tested and
reused independently of the LLM generation step.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

from .config import MedicalConfig, medical_config
from .preprocess import MedicalChunk
from .vector_store import MedicalVectorStore, VectorStoreError

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: MedicalChunk
    score: float


class MedicalRetriever:
    def __init__(self, vector_store: MedicalVectorStore, config: MedicalConfig | None = None) -> None:
        self._store = vector_store
        self._config = config or medical_config

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the most relevant chunks for the query, filtered by
        the configured minimum similarity. Returns an empty list (not
        an error) for empty input or when nothing meets the similarity
        threshold -- the "empty search results" case from the PRD."""
        if not query or not query.strip():
            return []
        if self._store.embedder is None or self._store.index is None:
            raise VectorStoreError("Medical vector store is not ready -- build() or load() it first.")

        k = top_k or self._config.top_k
        query_vector = self._store.embedder.embed([query])[0]
        results = self._store.search(query_vector, k)

        filtered = [
            RetrievedChunk(chunk=chunk, score=score)
            for chunk, score in results
            if score >= self._config.min_similarity
        ]
        logger.info(
            "Retrieved %d/%d candidate chunks above similarity threshold (%.2f) for query.",
            len(filtered), len(results), self._config.min_similarity,
        )
        return filtered
