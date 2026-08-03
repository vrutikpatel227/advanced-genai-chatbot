"""
modules/knowledge_base/metadata.py

Document-level metadata tracking: duplicate detection (via content
hash), document listing/stats, and persisting each document's
extracted+cleaned text to disk so "Rebuild Index" can re-chunk/
re-embed everything without needing the original uploaded files again.

Wraps utils/storage.py's knowledge_documents table -- reuses the
existing SQLite connection/schema-migration infrastructure rather than
rolling a separate database layer, per the PRD's "Reuse SQLite where
appropriate" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger
from utils.storage import (
    get_all_kb_documents,
    get_kb_chunk_total,
    get_kb_document_by_hash,
    get_kb_document_count,
    get_kb_last_update,
    save_kb_document,
)

from .config import KnowledgeBaseConfig, kb_config

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    file_type: str
    content_hash: str
    chunk_count: int
    status: str
    created_at: str


@dataclass(frozen=True)
class KnowledgeBaseStats:
    total_documents: int
    total_chunks: int
    last_update: Optional[str]
    vector_store_ready: bool


def document_exists(content_hash: str) -> Optional[DocumentRecord]:
    """Look up a document by content hash -- the duplicate-detection
    check performed before any parsing/embedding happens."""
    row = get_kb_document_by_hash(content_hash)
    return _row_to_record(row) if row else None


def record_document(
    doc_id: str, filename: str, file_type: str, content_hash: str,
    chunk_count: int, status: str = "indexed",
) -> None:
    save_kb_document(doc_id, filename, file_type, content_hash, chunk_count, status)


def list_documents() -> list[DocumentRecord]:
    return [_row_to_record(row) for row in get_all_kb_documents()]


def get_stats(vector_store_ready: bool) -> KnowledgeBaseStats:
    return KnowledgeBaseStats(
        total_documents=get_kb_document_count(),
        total_chunks=get_kb_chunk_total(),
        last_update=get_kb_last_update(),
        vector_store_ready=vector_store_ready,
    )


def _row_to_record(row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        filename=row["filename"],
        file_type=row["file_type"],
        content_hash=row["content_hash"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        created_at=row["created_at"],
    )


# --- document text persistence (enables "Rebuild Index") ---------------------------


def save_document_text(doc_id: str, text: str, config: Optional[KnowledgeBaseConfig] = None) -> None:
    """Persist a document's cleaned extracted text so it can be
    re-chunked/re-embedded later without re-uploading the original file."""
    cfg = config or kb_config
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (cfg.documents_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")


def load_document_text(doc_id: str, config: Optional[KnowledgeBaseConfig] = None) -> Optional[str]:
    cfg = config or kb_config
    path = cfg.documents_dir / f"{doc_id}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to read stored document text for %s: %s", doc_id, exc)
        return None
