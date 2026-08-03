"""
modules/knowledge_base/chunker.py

Splits cleaned document text into overlapping chunks (LangChain's
RecursiveCharacterTextSplitter, matching Milestone 2's approach) and
attaches metadata (doc_id, filename, chunk index) to each chunk.

Reuses clean_text() from modules.medical.preprocess directly rather
than reimplementing it -- it's generic whitespace/artifact cleanup
with no medical-specific logic, so duplicating it here would violate
the PRD's "do not duplicate existing RAG functionality" instruction.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.medical.preprocess import clean_text  # reused, not duplicated

from .config import KnowledgeBaseConfig, kb_config

__all__ = ["KnowledgeChunk", "chunk_document", "clean_text"]


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str          # f"{doc_id}::{index}"
    text: str
    doc_id: str
    filename: str
    chunk_index: int


def chunk_document(
    doc_id: str,
    filename: str,
    text: str,
    config: KnowledgeBaseConfig | None = None,
) -> list[KnowledgeChunk]:
    """Clean and split a single document's text into chunks."""
    cfg = config or kb_config
    cleaned = clean_text(text)
    if not cleaned:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(cleaned) if len(cleaned) > cfg.chunk_size else [cleaned]

    return [
        KnowledgeChunk(
            chunk_id=f"{doc_id}::{i}",
            text=piece,
            doc_id=doc_id,
            filename=filename,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]
