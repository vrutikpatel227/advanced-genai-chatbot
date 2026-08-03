"""
modules/knowledge_base/vector_store.py

FAISS-backed vector store supporting true incremental indexing: new
document chunks are embedded and added to the existing index without
rebuilding it from scratch, per the PRD's core "Dynamic" requirement.
This is genuinely different from Milestone 2's vector store (which
does a full rebuild-or-skip based on a corpus fingerprint) -- that
approach doesn't fit "add one new document without touching the
rest", so this file implements the necessary incremental-add logic
rather than reusing Milestone 2's rebuild-oriented one.

Uses a FAISS IndexIDMap wrapping a flat inner-product index, so each
chunk has a stable integer ID independent of insertion order --
needed for reliable incremental adds and future chunk-level lookups.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from utils.logger import get_logger

from .chunker import KnowledgeChunk
from .config import KnowledgeBaseConfig, kb_config
from .embeddings import KnowledgeEmbeddingGenerator

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when the vector store can't be built, saved, or loaded."""


@dataclass(frozen=True)
class KBManifest:
    backend: str
    dimension: int
    chunk_count: int
    next_id: int


class KnowledgeVectorStore:
    def __init__(self, config: KnowledgeBaseConfig | None = None) -> None:
        self._config = config or kb_config
        self.index = None
        self.chunks: dict[int, KnowledgeChunk] = {}
        self.embedder = None
        self.manifest: KBManifest | None = None

    # --- file locations ------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._config.vector_store_dir / "kb_index.faiss"

    @property
    def _metadata_path(self) -> Path:
        return self._config.vector_store_dir / "kb_metadata.json"

    @property
    def _manifest_path(self) -> Path:
        return self._config.vector_store_dir / "kb_manifest.json"

    @property
    def _embedder_state_path(self) -> Path:
        return self._config.vector_store_dir / "kb_embedder_state"

    def exists(self) -> bool:
        return (
            self._index_path.exists()
            and self._metadata_path.exists()
            and self._manifest_path.exists()
        )

    # --- load / init ---------------------------------------------------------

    def load(self) -> None:
        """Load an existing index + metadata + embedder state from disk."""
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu") from exc

        if not self.exists():
            raise VectorStoreError("Knowledge base vector index not found on disk.")

        try:
            manifest_data = json.loads(self._manifest_path.read_text())
            self.manifest = KBManifest(**manifest_data)

            self.index = faiss.read_index(str(self._index_path))

            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                raw_chunks = json.load(fh)
            self.chunks = {int(k): KnowledgeChunk(**v) for k, v in raw_chunks.items()}

            self.embedder = KnowledgeEmbeddingGenerator.load_backend(
                self.manifest.backend, self._embedder_state_path
            )
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            raise VectorStoreError(f"Knowledge base vector index on disk is corrupted: {exc}") from exc

    def ensure_ready(self, sample_texts_hint: list[str] | None = None) -> None:
        """Load the store if it exists; otherwise leave it empty and
        ready to accept the first incremental add. Never raises for
        "doesn't exist yet" -- that's the normal state on first use."""
        if self.exists():
            try:
                self.load()
                return
            except VectorStoreError as exc:
                logger.warning("Could not load existing knowledge base index (%s); starting fresh.", exc)

        if self.embedder is None:
            generator = KnowledgeEmbeddingGenerator(self._config)
            self.embedder = generator.select_backend(sample_texts_hint or ["initialization"])

    # --- incremental add ----------------------------------------------------------

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> int:
        """Embed and add new chunks to the index without touching any
        existing vectors. Returns the number of chunks added."""
        if not chunks:
            return 0

        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu") from exc

        if self.embedder is None:
            self.ensure_ready(sample_texts_hint=[c.text for c in chunks[:5]])

        try:
            vectors = self.embedder.embed([c.text for c in chunks])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to generate embeddings for new document(s): {exc}") from exc

        if self.index is None:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))

        next_id = (max(self.chunks.keys()) + 1) if self.chunks else 0
        ids = np.arange(next_id, next_id + len(chunks), dtype="int64")

        self.index.add_with_ids(vectors, ids)
        for id_, chunk in zip(ids, chunks):
            self.chunks[int(id_)] = chunk

        self.manifest = KBManifest(
            backend=self.embedder.name,
            dimension=int(vectors.shape[1]),
            chunk_count=len(self.chunks),
            next_id=int(next_id + len(chunks)),
        )
        self._save()
        logger.info(
            "Knowledge base index updated incrementally: +%d chunks (total now %d).",
            len(chunks), len(self.chunks),
        )
        return len(chunks)

    # --- full rebuild ------------------------------------------------------------

    def rebuild(self, all_chunks: list[KnowledgeChunk]) -> None:
        """Full rebuild from scratch -- the explicit "Rebuild Index"
        user action, distinct from the normal incremental add path.
        Re-selects the embedding backend (useful if e.g. Sentence
        Transformers has become available since the fallback was used)."""
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu") from exc

        if not all_chunks:
            self.index = None
            self.chunks = {}
            self.manifest = KBManifest(backend="none", dimension=0, chunk_count=0, next_id=0)
            self._save()
            return

        generator = KnowledgeEmbeddingGenerator(self._config)
        embedder = generator.select_backend([c.text for c in all_chunks[:5]])
        vectors = embedder.embed([c.text for c in all_chunks])

        index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
        ids = np.arange(0, len(all_chunks), dtype="int64")
        index.add_with_ids(vectors, ids)

        self.index = index
        self.embedder = embedder
        self.chunks = {int(id_): chunk for id_, chunk in zip(ids, all_chunks)}
        self.manifest = KBManifest(
            backend=embedder.name,
            dimension=int(vectors.shape[1]),
            chunk_count=len(all_chunks),
            next_id=len(all_chunks),
        )
        self._save()
        logger.info("Knowledge base index fully rebuilt: %d chunks, backend=%s.", len(all_chunks), embedder.name)

    # --- search -----------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[tuple[KnowledgeChunk, float]]:
        if self.index is None or self.embedder is None:
            return []

        query_vector = np.asarray(self.embedder.embed([query])[0], dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)

        results: list[tuple[KnowledgeChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks.get(int(idx))
            if chunk is not None:
                results.append((chunk, float(score)))
        return results

    # --- persistence --------------------------------------------------------------

    def _save(self) -> None:
        import faiss

        self._config.vector_store_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(self._index_path))

        with open(self._metadata_path, "w", encoding="utf-8") as fh:
            json.dump({str(k): asdict(v) for k, v in self.chunks.items()}, fh)

        if self.manifest is not None:
            self._manifest_path.write_text(json.dumps(asdict(self.manifest)))

        if self.embedder is not None:
            self.embedder.save_state(self._embedder_state_path)
