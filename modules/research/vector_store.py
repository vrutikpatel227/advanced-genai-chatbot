"""
modules/research/vector_store.py

Reuses KnowledgeVectorStore (Milestone 3) directly via subclassing --
incremental add, save, load, and search are inherited unchanged. The
only genuinely new capability this milestone needs that Milestone 3
doesn't have is per-paper deletion ("Deleting a paper should remove
its vectors without affecting other indexed papers"), so that's the
only method added here.

FAISS's IndexIDMap (which KnowledgeVectorStore already builds) natively
supports remove_ids(), so deletion doesn't require any different index
type -- just a new method using a capability the existing index
structure already had.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from modules.knowledge_base.vector_store import (  # reused, not duplicated
    KBManifest,
    KnowledgeVectorStore,
    VectorStoreError,
)
from utils.logger import get_logger

from .chunker import ResearchChunk

logger = get_logger(__name__)

__all__ = ["ResearchVectorStore", "VectorStoreError", "KBManifest"]


class ResearchVectorStore(KnowledgeVectorStore):
    """Adds per-paper deletion on top of KnowledgeVectorStore's
    inherited incremental add/save/load/search/rebuild."""

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to doc_id from the index and
        metadata, without touching any other document's vectors.
        Returns the number of chunks removed. Safe to call even if
        doc_id has no indexed chunks (returns 0)."""
        if self.index is None or not self.chunks:
            return 0

        ids_to_remove = [cid for cid, chunk in self.chunks.items() if chunk.doc_id == doc_id]
        if not ids_to_remove:
            return 0

        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu") from exc

        try:
            id_array = np.array(ids_to_remove, dtype="int64")
            selector = faiss.IDSelectorArray(id_array)
            self.index.remove_ids(selector)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to remove vectors for document {doc_id}: {exc}") from exc

        for cid in ids_to_remove:
            del self.chunks[cid]

        if self.manifest is not None:
            self.manifest = dataclasses.replace(self.manifest, chunk_count=len(self.chunks))

        self._save()
        logger.info(
            "Removed %d chunks for document %s from the research index (%d chunks remain).",
            len(ids_to_remove), doc_id, len(self.chunks),
        )
        return len(ids_to_remove)

    def get_chunks_for_document(self, doc_id: str) -> list[ResearchChunk]:
        """All chunks belonging to one paper, in original order --
        used by the summarizer, which needs the whole paper's content
        rather than just the top-K most similar chunks to a query."""
        matching = [chunk for chunk in self.chunks.values() if chunk.doc_id == doc_id]
        return sorted(matching, key=lambda c: c.chunk_index)
