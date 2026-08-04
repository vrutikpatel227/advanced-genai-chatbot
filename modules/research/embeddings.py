"""
modules/research/embeddings.py

Embedding generation for research papers reuses
KnowledgeEmbeddingGenerator directly from modules.knowledge_base --
that class already implements exactly what's needed here (try
Sentence Transformers, fall back to a stateless hashing embedder for
incremental-safe indexing) and the Research Assistant has the same
incremental-indexing requirement as the Knowledge Base. No new
embedding logic is written for this milestone.
"""

from __future__ import annotations

from modules.knowledge_base.embeddings import (  # reused, not duplicated
    BACKEND_REGISTRY,
    BaseEmbedder,
    EmbeddingError,
    HashingEmbedder,
    KnowledgeEmbeddingGenerator,
    SentenceTransformerEmbedder,
)

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "HashingEmbedder",
    "EmbeddingError",
    "BACKEND_REGISTRY",
    "ResearchEmbeddingGenerator",
]

# Alias for readability at call sites in this module -- same class,
# reused rather than redefined.
ResearchEmbeddingGenerator = KnowledgeEmbeddingGenerator
