"""
modules/research/config.py

Configuration for the Research Assistant, read from environment
variables. Self-contained inside modules/research/ (root config.py is
outside this PRD's folder scope, same pattern already established by
modules/medical/config.py and modules/knowledge_base/config.py).

Per the PRD's strong "reuse existing infrastructure, avoid duplicate
implementations" instruction: rather than redefining a config dataclass
with the same fields as KnowledgeBaseConfig (documents_dir,
vector_store_dir, chunk_size, ...), this module reuses that exact type
via composition -- ResearchConfig.kb *is* a KnowledgeBaseConfig
instance, just built with research-specific paths/defaults/env vars.
This is what lets ResearchVectorStore subclass KnowledgeVectorStore
directly (see vector_store.py) without any type mismatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from modules.knowledge_base.config import KnowledgeBaseConfig  # reused type, not duplicated

load_dotenv()

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent

SUPPORTED_EXTENSIONS = (".pdf",)  # Research Assistant: PDF only, per PRD


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _build_kb_config() -> KnowledgeBaseConfig:
    """Build a KnowledgeBaseConfig instance with research-specific
    paths and defaults -- the reused type, populated independently
    from the Dynamic Knowledge Base's own instance."""
    return KnowledgeBaseConfig(
        documents_dir=Path(
            os.getenv("RESEARCH_DOCUMENTS_DIR", str(_PROJECT_ROOT / "data" / "research" / "documents"))
        ),
        vector_store_dir=Path(
            os.getenv("RESEARCH_VECTOR_STORE_DIR", str(_PROJECT_ROOT / "vector_store" / "research"))
        ),
        max_file_size_mb=_get_int("RESEARCH_MAX_FILE_SIZE_MB", 30),
        chunk_size=_get_int("RESEARCH_CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("RESEARCH_CHUNK_OVERLAP", 150),
        embedding_model_name=os.getenv(
            "RESEARCH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hashing_dimension=_get_int("RESEARCH_HASHING_DIMENSION", 512),
        top_k=_get_int("RESEARCH_TOP_K", 5),
        min_similarity=_get_float("RESEARCH_MIN_SIMILARITY", 0.05),
    )


@dataclass(frozen=True)
class ResearchConfig:
    """Wraps a (reused) KnowledgeBaseConfig with the extra settings
    specific to research papers (summarization limits) that the
    Knowledge Base config type has no need for."""

    kb: KnowledgeBaseConfig = field(default_factory=_build_kb_config)
    max_summary_chunks: int = field(default_factory=lambda: _get_int("RESEARCH_MAX_SUMMARY_CHUNKS", 40))
    max_summary_context_chars: int = field(
        default_factory=lambda: _get_int("RESEARCH_MAX_SUMMARY_CONTEXT_CHARS", 16000)
    )


research_config = ResearchConfig()


def ensure_research_directories() -> None:
    research_config.kb.documents_dir.mkdir(parents=True, exist_ok=True)
    research_config.kb.vector_store_dir.mkdir(parents=True, exist_ok=True)
