"""
modules/medical/vector_store.py

FAISS-backed vector store for the medical knowledge base:
  - Builds an index from a list of MedicalChunk (embedding + indexing).
  - Saves the index, chunk metadata, embedder state, and a manifest
    (fingerprint of the source chunks) locally.
  - Loads an existing index automatically on next run, and skips
    regenerating embeddings entirely if the underlying chunk set
    hasn't changed -- satisfying the PRD's "avoid regenerating
    embeddings if they already exist" requirement.
  - Fast cosine-similarity search via a FAISS flat inner-product index
    (embeddings are L2-normalized by the embedder, so inner product
    equals cosine similarity).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from utils.logger import get_logger

from .config import MedicalConfig, medical_config
from .embeddings import EmbeddingGenerator
from .preprocess import MedicalChunk

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when the vector store can't be built, saved, or loaded."""


@dataclass(frozen=True)
class VectorStoreManifest:
    backend: str
    dimension: int
    chunk_count: int
    fingerprint: str


class MedicalVectorStore:
    def __init__(self, config: MedicalConfig | None = None) -> None:
        self._config = config or medical_config
        self.index = None
        self.chunks: list[MedicalChunk] = []
        self.embedder = None
        self.manifest: VectorStoreManifest | None = None

    # --- file locations ------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._config.vector_store_dir / "medical_index.faiss"

    @property
    def _metadata_path(self) -> Path:
        return self._config.vector_store_dir / "medical_metadata.json"

    @property
    def _manifest_path(self) -> Path:
        return self._config.vector_store_dir / "medical_manifest.json"

    @property
    def _embedder_state_path(self) -> Path:
        return self._config.vector_store_dir / "medical_embedder_state"

    def exists(self) -> bool:
        return (
            self._index_path.exists()
            and self._metadata_path.exists()
            and self._manifest_path.exists()
        )

    # --- build / load ----------------------------------------------------------

    def build(self, chunks: list[MedicalChunk], force_rebuild: bool = False) -> None:
        """Build (or reuse) the vector index for the given chunks.
        Automatically skips regeneration if an up-to-date index already
        exists on disk for this exact chunk set."""
        fingerprint = self._compute_fingerprint(chunks)

        if not force_rebuild and self.exists():
            try:
                self.load()
                if self.manifest and self.manifest.fingerprint == fingerprint:
                    logger.info(
                        "Existing medical vector index is up to date (%d chunks); skipping regeneration.",
                        self.manifest.chunk_count,
                    )
                    return
                logger.info("Medical dataset changed; rebuilding vector index.")
            except Exception as exc:  # noqa: BLE001 - any load failure -> rebuild from scratch
                logger.warning("Could not load existing medical vector index (%s); rebuilding.", exc)

        texts = [c.text for c in chunks]
        generator = EmbeddingGenerator(self._config)
        vectors, embedder = generator.build(texts)

        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu"
            ) from exc

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.manifest = VectorStoreManifest(
            backend=embedder.name,
            dimension=int(vectors.shape[1]),
            chunk_count=len(chunks),
            fingerprint=fingerprint,
        )

        self._save()
        logger.info(
            "Medical vector index built: %d chunks, backend=%s, dimension=%d.",
            len(chunks), embedder.name, vectors.shape[1],
        )

    def load(self) -> None:
        """Load an existing index + metadata + embedder state from disk.
        Raises VectorStoreError if any required file is missing/corrupted."""
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu"
            ) from exc

        if not self.exists():
            raise VectorStoreError("Medical vector index not found on disk.")

        try:
            manifest_data = json.loads(self._manifest_path.read_text())
            self.manifest = VectorStoreManifest(**manifest_data)

            self.index = faiss.read_index(str(self._index_path))

            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                raw_chunks = json.load(fh)
            self.chunks = [MedicalChunk(**c) for c in raw_chunks]

            self.embedder = EmbeddingGenerator.load_backend(self.manifest.backend, self._embedder_state_path)
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            raise VectorStoreError(f"Medical vector index on disk is corrupted: {exc}") from exc

    # --- search -----------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[MedicalChunk, float]]:
        if self.index is None:
            raise VectorStoreError("Vector store not loaded/built -- call build() or load() first.")

        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)

        results: list[tuple[MedicalChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # --- persistence helpers ------------------------------------------------------

    def _save(self) -> None:
        import faiss

        self._config.vector_store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))

        with open(self._metadata_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(c) for c in self.chunks], fh)

        self._manifest_path.write_text(json.dumps(asdict(self.manifest)))
        self.embedder.save_state(self._embedder_state_path)

    @staticmethod
    def _compute_fingerprint(chunks: list[MedicalChunk]) -> str:
        hasher = hashlib.sha256()
        for chunk in chunks:
            hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(str(len(chunks)).encode("utf-8"))
        return hasher.hexdigest()
