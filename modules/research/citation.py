"""
modules/research/citation.py

Formats retrieved passages into citation objects for display: paper
filename, the retrieved chunk text, and its similarity score. Per the
PRD, citations must never be fabricated -- every Citation here is
built directly from an actual RetrievedPassage, never invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retriever import RetrievedPassage


@dataclass(frozen=True)
class Citation:
    filename: str
    doc_id: str
    chunk_text: str
    similarity_score: float


def build_citations(retrieved: list[RetrievedPassage]) -> list[Citation]:
    """Convert retrieved passages into citation objects, one per
    passage. Deduplication is by chunk_id -- multiple distinct
    passages from the same paper are legitimate, separate citations."""
    seen_chunk_ids: set[str] = set()
    citations: list[Citation] = []
    for item in retrieved:
        if item.chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk.chunk_id)
        citations.append(
            Citation(
                filename=item.chunk.filename,
                doc_id=item.chunk.doc_id,
                chunk_text=item.chunk.text,
                similarity_score=item.score,
            )
        )
    return citations
