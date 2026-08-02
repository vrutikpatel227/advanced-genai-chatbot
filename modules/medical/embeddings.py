"""
modules/medical/embeddings.py

Embedding generation for the medical knowledge base.

Primary backend: Sentence Transformers, per the PRD's "Embeddings:
Sentence Transformers" requirement.

Fallback backend: a local TF-IDF vectorizer (scikit-learn), used
automatically if the sentence-transformer model can't be loaded (e.g.
no network access to download it). This mirrors the same
primary-backend + automatic-fallback pattern already established by
Milestone 1's sentiment analyzer, and satisfies the PRD's own
"Embedding failure" error-handling requirement -- the medical
assistant stays functional (with a clearly-logged, lower-quality
retrieval backend) instead of becoming entirely unavailable.

Embeddings are generated once per corpus and cached to disk by
vector_store.py -- this module only knows how to *produce* vectors,
not how to persist the index.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from utils.logger import get_logger

from .config import MedicalConfig, medical_config

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when no embedding backend (primary or fallback) could be used."""


class BaseEmbedder(ABC):
    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Prepare the embedder against the given corpus (loads the
        model for sentence-transformers; fits the vectorizer for TF-IDF)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array of embeddings."""

    @abstractmethod
    def save_state(self, path: Path) -> None:
        """Persist whatever state is needed to embed new queries later
        with the exact same embedding space."""

    @abstractmethod
    def load_state(self, path: Path) -> None:
        """Restore state saved by save_state()."""


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformer"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # optional heavy dependency

        self._model = SentenceTransformer(self._model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def fit(self, corpus: list[str]) -> None:
        self._ensure_loaded()

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype="float32")

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps({"backend": self.name, "model_name": self._model_name}))

    def load_state(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self._model_name = data["model_name"]
        self._model = None
        self.dimension = 0


class TfidfFallbackEmbedder(BaseEmbedder):
    """Dependency-light fallback: a fitted TF-IDF vectorizer, padded/
    normalized to a fixed-width dense vector so it's a drop-in
    replacement for FAISS similarity search. Not as semantically
    strong as real sentence embeddings, but keeps retrieval functional
    with zero network dependency."""

    name = "tfidf-fallback"

    def __init__(self, max_features: int) -> None:
        self._max_features = max_features
        self._vectorizer = None
        self.dimension = max_features

    def fit(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # optional dependency

        self._vectorizer = TfidfVectorizer(max_features=self._max_features, stop_words="english")
        self._vectorizer.fit(corpus)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise EmbeddingError("TfidfFallbackEmbedder.embed() called before fit()/load_state().")

        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype("float32")
        if dense.shape[1] < self._max_features:
            pad = np.zeros((dense.shape[0], self._max_features - dense.shape[1]), dtype="float32")
            dense = np.hstack([dense, pad])

        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def save_state(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"vectorizer": self._vectorizer, "max_features": self._max_features}, fh)

    def load_state(self, path: Path) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._vectorizer = data["vectorizer"]
        self._max_features = data["max_features"]
        self.dimension = self._max_features


BACKEND_REGISTRY: dict[str, type[BaseEmbedder]] = {
    SentenceTransformerEmbedder.name: SentenceTransformerEmbedder,
    TfidfFallbackEmbedder.name: TfidfFallbackEmbedder,
}


class EmbeddingGenerator:
    """Builds embeddings for a corpus, trying the primary backend first
    and transparently falling back on failure. Every result records
    which backend actually produced it (logged + persisted in the
    vector store manifest), so degradation is visible, not silent."""

    def __init__(self, config: MedicalConfig | None = None) -> None:
        self._config = config or medical_config

    def build(self, texts: list[str]) -> tuple[np.ndarray, BaseEmbedder]:
        """Try sentence-transformers; fall back to TF-IDF on any failure.
        Raises EmbeddingError only if *both* backends fail."""
        try:
            embedder: BaseEmbedder = SentenceTransformerEmbedder(self._config.embedding_model_name)
            embedder.fit(texts)
            vectors = embedder.embed(texts)
            logger.info(
                "Generated %d medical embeddings using sentence-transformer backend (%s).",
                len(texts), self._config.embedding_model_name,
            )
            return vectors, embedder
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            logger.warning(
                "Sentence-transformer embedding backend unavailable (%s); "
                "falling back to TF-IDF embedder.", exc,
            )

        try:
            embedder = TfidfFallbackEmbedder(self._config.tfidf_max_features)
            embedder.fit(texts)
            vectors = embedder.embed(texts)
            logger.info("Generated %d medical embeddings using TF-IDF fallback backend.", len(texts))
            return vectors, embedder
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"All embedding backends failed: {exc}") from exc

    @staticmethod
    def load_backend(backend_name: str, state_path: Path) -> BaseEmbedder:
        """Reconstruct the embedder used to build an existing index, so
        new queries are embedded into the exact same vector space."""
        backend_cls = BACKEND_REGISTRY.get(backend_name)
        if backend_cls is None:
            raise EmbeddingError(f"Unknown embedding backend in saved index: {backend_name!r}")
        embedder = backend_cls.__new__(backend_cls)  # bypass __init__, state comes from load_state
        embedder.load_state(state_path)
        return embedder
