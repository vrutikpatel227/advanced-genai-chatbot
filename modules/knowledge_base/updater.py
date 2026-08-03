"""
modules/knowledge_base/updater.py

"Update Index" operation: reprocesses only documents whose status
isn't "indexed" (e.g. left in a failed/partial state from an earlier
error) and incrementally adds their chunks. Distinct from
vector_store.rebuild(), which rebuilds the entire index from scratch --
this only touches documents that actually need it.
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger

from .chunker import chunk_document
from .config import KnowledgeBaseConfig, kb_config
from .metadata import list_documents, load_document_text, record_document
from .vector_store import KnowledgeVectorStore, VectorStoreError

logger = get_logger(__name__)


def update_pending_documents(
    store: KnowledgeVectorStore, config: Optional[KnowledgeBaseConfig] = None
) -> int:
    """Find documents not in "indexed" status and incrementally add
    their chunks to the vector store. Returns the number of documents
    successfully updated."""
    cfg = config or kb_config
    pending = [doc for doc in list_documents() if doc.status != "indexed"]
    if not pending:
        logger.info("Knowledge base update check: no pending documents found.")
        return 0

    updated = 0
    for doc in pending:
        text = load_document_text(doc.doc_id, cfg)
        if text is None:
            logger.warning(
                "Cannot update document %s (%s): stored text is missing.",
                doc.doc_id, doc.filename,
            )
            continue

        chunks = chunk_document(doc.doc_id, doc.filename, text, cfg)
        if not chunks:
            continue

        try:
            added = store.add_chunks(chunks)
        except VectorStoreError as exc:
            logger.error("Failed to update document %s (%s): %s", doc.doc_id, doc.filename, exc)
            continue

        record_document(doc.doc_id, doc.filename, doc.file_type, doc.content_hash, added, status="indexed")
        updated += 1

    logger.info("Knowledge base update: %d/%d pending documents updated.", updated, len(pending))
    return updated
