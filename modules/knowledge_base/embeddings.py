"""
modules/knowledge_base/embeddings.py

Embedding generation for the Dynamic Knowledge Base.

Primary backend: reuses SentenceTransformerEmbedder and BaseEmbedder
directly from modules.medical.embeddings (Milestone 2) -- imported,
not reimplemented, per the PRD's "reuse the existing embedding
infrastructure" / "do not duplicate existing RAG functionality"
instruction. This class is stateless per call (encode() works on any
new text without needing to be fit on a corpus first), so it's
naturally incremental-friendly with zero changes needed.

Fallback backend: Milestone 2's TF-IDF fallback is fit once on a
static corpus, which doesn't suit the Knowledge Base's core
requirement -- new documents must become searchable incrementally,
potentially introducing vocabulary a pre-fit TF-IDF vectorizer has
never seen. A HashingVectorizer fallback is used here instead: it's
stateless (no fitting required, fixed-size output for any input),
which is the correct tool for genuinely incremental embedding -- this
is new code because it solves a different problem than Milestone 2's,
not a duplicate of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modules.medical.embeddings import BaseEmbedder, SentenceTransformerEmbedder  # reused, not duplicated
from utils.logger import get_logger

from .config import KnowledgeBaseConfig, kb_config

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when no embedding backend (primary or fallback) could be used."""


class HashingEmbedder(BaseEmbedder):
    """Stateless fallback embedder: a fixed-size feature hashing
    vectorizer. Unlike TF-IDF, it never needs to be fit on a corpus,
    so brand-new documents added later embed correctly without any
    refit step -- exactly what incremental indexing needs."""

    name = "hashing-fallback"

    def __init__(self, n_features: int) -> None:
        self._n_features = n_features
        self.dimension = n_features
        self._vectorizer = None

    def _ensure_vectorizer(self):
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import HashingVectorizer  # optional dependency

            self._vectorizer = HashingVectorizer(
                n_features=self._n_features, alternate_sign=False, norm="l2"
            )
        return self._vectorizer

    def fit(self, corpus: list[str]) -> None:
        # Stateless by design -- nothing to fit. Still validated here
        # so a missing scikit-learn install surfaces immediately.
        self._ensure_vectorizer()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectorizer = self._ensure_vectorizer()
        matrix = vectorizer.transform(texts)
        return matrix.toarray().astype("float32")

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps({"backend": self.name, "n_features": self._n_features}))

    def load_state(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self._n_features = data["n_features"]
        self.dimension = self._n_features
        self._vectorizer = None


BACKEND_REGISTRY: dict[str, type[BaseEmbedder]] = {
    SentenceTransformerEmbedder.name: SentenceTransformerEmbedder,
    HashingEmbedder.name: HashingEmbedder,
}


class KnowledgeEmbeddingGenerator:
    """Builds/selects the embedding backend for the knowledge base.
    Mirrors Milestone 2's try-primary-then-fallback pattern, but with
    a fallback suited to incremental use (see HashingEmbedder above)."""

    def __init__(self, config: KnowledgeBaseConfig | None = None) -> None:
        self._config = config or kb_config

    def select_backend(self, sample_texts: list[str]) -> BaseEmbedder:
        """Return a ready-to-use embedder, trying Sentence Transformers
        first and falling back to hashing on any failure. `sample_texts`
        is only used to smoke-test the primary backend actually works."""
        try:
            embedder: BaseEmbedder = SentenceTransformerEmbedder(self._config.embedding_model_name)
            embedder.fit(sample_texts)
            embedder.embed(sample_texts[:1] or ["test"])  # smoke test
            logger.info(
                "Knowledge base using sentence-transformer embedding backend (%s).",
                self._config.embedding_model_name,
            )
            return embedder
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            logger.warning(
                "Sentence-transformer embedding backend unavailable (%s); "
                "falling back to hashing embedder for incremental indexing.", exc,
            )

        try:
            embedder = HashingEmbedder(self._config.hashing_dimension)
            embedder.fit(sample_texts)
            logger.info("Knowledge base using hashing fallback embedding backend.")
            return embedder
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"All embedding backends failed: {exc}") from exc

    @staticmethod
    def load_backend(backend_name: str, state_path: Path) -> BaseEmbedder:
        backend_cls = BACKEND_REGISTRY.get(backend_name)
        if backend_cls is None:
            raise EmbeddingError(f"Unknown embedding backend in saved index: {backend_name!r}")
        embedder = backend_cls.__new__(backend_cls)
        embedder.load_state(state_path)
        return embedder
