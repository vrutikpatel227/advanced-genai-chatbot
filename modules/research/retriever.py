"""
modules/research/retriever.py

Retrieval step of the research RAG pipeline: embeds the query and
searches the FAISS index for the most relevant chunks, filtered by a
minimum similarity threshold. search() itself is fully inherited from
KnowledgeVectorStore (embeds the query internally) -- this module only
adds the similarity-threshold filtering and a research-specific result
type, mirroring the same pattern already used in
modules.medical.retriever.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

from .chunker import ResearchChunk
from .config import ResearchConfig, research_config
from .vector_store import ResearchVectorStore, VectorStoreError

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedPassage:
    chunk: ResearchChunk
    score: float


class ResearchRetriever:
    def __init__(self, vector_store: ResearchVectorStore, config: ResearchConfig | None = None) -> None:
        self._store = vector_store
        self._config = config or research_config

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedPassage]:
        """Return the most relevant passages for the query, filtered by
        the configured minimum similarity. Empty input or nothing
        meeting the threshold both return an empty list, not an error."""
        if not query or not query.strip():
            return []
        if self._store.embedder is None or self._store.index is None:
            raise VectorStoreError("Research vector store is not ready -- build() or load() it first.")

        k = top_k or self._config.kb.top_k
        results = self._store.search(query, k)

        filtered = [
            RetrievedPassage(chunk=chunk, score=score)
            for chunk, score in results
            if score >= self._config.kb.min_similarity
        ]
        logger.info(
            "Retrieved %d/%d candidate passages above similarity threshold (%.2f) for query.",
            len(filtered), len(results), self._config.kb.min_similarity,
        )
        return filtered

    def retrieve_from_paper(self, doc_id: str) -> list[RetrievedPassage]:
        """All chunks for one paper (score=1.0, since these aren't
        similarity-ranked) -- used by the summarizer."""
        chunks = self._store.get_chunks_for_document(doc_id)
        return [RetrievedPassage(chunk=chunk, score=1.0) for chunk in chunks]
