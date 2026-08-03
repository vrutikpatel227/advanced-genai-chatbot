"""
modules/knowledge_base/manager.py

Top-level orchestration for the Dynamic Knowledge Base: owns the
upload -> validate -> extract -> chunk -> embed -> index -> record
workflow, and exposes search, stats, and index control operations to
the UI. This is the single entry point the Streamlit page (and any
future caller) should use -- it never lets an internal failure crash
the app; every error path returns a friendly UploadResult instead.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

from .chunker import KnowledgeChunk, chunk_document
from .config import KnowledgeBaseConfig, ensure_kb_directories, kb_config
from .metadata import (
    DocumentRecord,
    KnowledgeBaseStats,
    document_exists,
    get_stats,
    list_documents,
    load_document_text,
    record_document,
    save_document_text,
)
from .parser import InvalidFileError, ParsingError, extract_text, validate_file
from .updater import update_pending_documents
from .vector_store import KnowledgeVectorStore, VectorStoreError

logger = get_logger(__name__)


@dataclass(frozen=True)
class UploadResult:
    status: str          # "success" | "duplicate" | "error"
    message: str
    filename: str
    chunk_count: int = 0
    doc_id: Optional[str] = None


@dataclass(frozen=True)
class RetrievedKnowledgeChunk:
    chunk: KnowledgeChunk
    score: float


class KnowledgeBaseManager:
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None) -> None:
        self._config = config or kb_config
        self._store = KnowledgeVectorStore(self._config)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            ensure_kb_directories()
            self._store.ensure_ready()
            self._loaded = True

    @property
    def is_vector_store_ready(self) -> bool:
        return self._store.index is not None

    @property
    def embedding_backend(self) -> str:
        return self._store.embedder.name if self._store.embedder else "not initialized"

    # --- upload ------------------------------------------------------------------

    def process_upload(self, filename: str, file_bytes: bytes) -> UploadResult:
        """Full pipeline for one uploaded file. Never raises -- every
        failure mode (invalid format, duplicate, parsing failure,
        empty document, embedding/index failure, metadata-save
        failure) maps to a friendly UploadResult instead."""
        self._ensure_loaded()

        try:
            extension = validate_file(filename, file_bytes, self._config)
        except InvalidFileError as exc:
            return UploadResult(status="error", message=str(exc), filename=filename)

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = document_exists(content_hash)
        if existing is not None:
            return UploadResult(
                status="duplicate",
                message=f"'{filename}' has already been indexed (as '{existing.filename}').",
                filename=filename,
            )

        try:
            raw_text = extract_text(filename, file_bytes, extension)
        except (ParsingError, InvalidFileError) as exc:
            return UploadResult(status="error", message=str(exc), filename=filename)

        doc_id = str(uuid.uuid4())
        try:
            chunks = chunk_document(doc_id, filename, raw_text, self._config)
        except Exception as exc:  # noqa: BLE001 - never let chunking failures crash the app
            logger.error("Failed to chunk '%s': %s", filename, exc)
            return UploadResult(
                status="error",
                message=f"Failed to process the text content of '{filename}'. Please try again.",
                filename=filename,
            )

        if not chunks:
            return UploadResult(
                status="error",
                message=f"'{filename}' appears to be empty after text extraction.",
                filename=filename,
            )

        try:
            added = self._store.add_chunks(chunks)
        except VectorStoreError as exc:
            logger.error("Failed to index '%s': %s", filename, exc)
            return UploadResult(
                status="error",
                message=(
                    f"Failed to generate embeddings or update the search index "
                    f"for '{filename}'. Please try again."
                ),
                filename=filename,
            )

        try:
            save_document_text(doc_id, raw_text, self._config)
            record_document(doc_id, filename, extension, content_hash, added, status="indexed")
        except Exception as exc:  # noqa: BLE001 - PRD error handling: database failure
            logger.error("Failed to save metadata for '%s': %s", filename, exc)
            return UploadResult(
                status="error",
                message=(
                    f"'{filename}' was indexed, but saving its metadata failed. "
                    "It may not appear correctly in the document list."
                ),
                filename=filename, chunk_count=added, doc_id=doc_id,
            )

        return UploadResult(
            status="success",
            message=f"'{filename}' indexed successfully ({added} chunk{'s' if added != 1 else ''}).",
            filename=filename, chunk_count=added, doc_id=doc_id,
        )

    # --- search ------------------------------------------------------------------

    def search(self, query: str, top_k: Optional[int] = None) -> list[RetrievedKnowledgeChunk]:
        self._ensure_loaded()
        if not query or not query.strip():
            return []
        k = top_k or self._config.top_k
        results = self._store.search(query, k)
        return [
            RetrievedKnowledgeChunk(chunk=chunk, score=score)
            for chunk, score in results
            if score >= self._config.min_similarity
        ]

    # --- stats / listing -----------------------------------------------------------

    def get_stats(self) -> KnowledgeBaseStats:
        self._ensure_loaded()
        return get_stats(vector_store_ready=self.is_vector_store_ready)

    def list_documents(self) -> list[DocumentRecord]:
        return list_documents()

    # --- index controls -----------------------------------------------------------

    def update_index(self) -> int:
        """Process any pending/failed documents incrementally. Returns
        the number of documents updated."""
        self._ensure_loaded()
        return update_pending_documents(self._store, self._config)

    def rebuild_index(self) -> int:
        """Full rebuild from all stored document texts (re-chunks and
        re-embeds everything). Returns the number of documents rebuilt."""
        self._ensure_loaded()
        documents = list_documents()
        all_chunks: list[KnowledgeChunk] = []
        rebuilt_docs: list[DocumentRecord] = []

        for doc in documents:
            text = load_document_text(doc.doc_id, self._config)
            if text is None:
                logger.warning(
                    "No stored text found for document %s (%s); skipped in rebuild.",
                    doc.doc_id, doc.filename,
                )
                continue
            chunks = chunk_document(doc.doc_id, doc.filename, text, self._config)
            all_chunks.extend(chunks)
            rebuilt_docs.append(doc)

        self._store.rebuild(all_chunks)

        # Chunk counts may have changed (e.g. a different chunk_size
        # since the original upload) -- keep metadata in sync.
        counts: dict[str, int] = {}
        for chunk in all_chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        for doc in rebuilt_docs:
            record_document(
                doc.doc_id, doc.filename, doc.file_type, doc.content_hash,
                counts.get(doc.doc_id, 0), status="indexed",
            )

        return len(rebuilt_docs)
