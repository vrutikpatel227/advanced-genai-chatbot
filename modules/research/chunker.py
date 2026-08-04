"""
modules/research/chunker.py

Chunking for research papers reuses chunk_document() and
KnowledgeChunk directly from modules.knowledge_base.chunker -- the
generic (chunk_id, text, doc_id, filename, chunk_index) shape already
fits a paper chunk exactly, and that function already reuses
clean_text() from modules.medical.preprocess. No new chunking logic
is needed for this milestone; only a research-appropriate config
(larger chunk size, suited to dense academic prose) is supplied by
modules.research.config.
"""

from __future__ import annotations

from modules.knowledge_base.chunker import (  # reused, not duplicated
    KnowledgeChunk,
    chunk_document,
    clean_text,
)

from .config import ResearchConfig, research_config

__all__ = ["ResearchChunk", "chunk_paper", "clean_text"]

# Alias for readability at call sites in this module -- same type,
# reused rather than redefined.
ResearchChunk = KnowledgeChunk


def chunk_paper(
    paper_id: str,
    filename: str,
    text: str,
    config: ResearchConfig | None = None,
) -> list[ResearchChunk]:
    """Clean and split one paper's extracted text into chunks."""
    cfg = config or research_config
    return chunk_document(paper_id, filename, text, cfg.kb)
