"""
modules/knowledge_base/config.py

Configuration for the Dynamic Knowledge Base, read from environment
variables. Kept self-contained inside modules/knowledge_base/ (rather
than added to the root config.py) since this milestone's PRD scopes
changes to modules/knowledge_base/, components/, database/, utils/,
README.md, requirements.txt, and app.py -- root config.py is not in
that list. Mirrors the same pattern already established by
modules/medical/config.py in Milestone 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent


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


SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Settings for the Dynamic Knowledge Base. Nothing here is
    hardcoded elsewhere in the module -- every path/limit/model name
    is overridable via .env."""

    documents_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KB_DOCUMENTS_DIR", str(_PROJECT_ROOT / "data" / "knowledge_base" / "documents"))
        )
    )
    vector_store_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KB_VECTOR_STORE_DIR", str(_PROJECT_ROOT / "vector_store" / "knowledge_base"))
        )
    )

    max_file_size_mb: int = field(default_factory=lambda: _get_int("KB_MAX_FILE_SIZE_MB", 20))

    # Chunking (same defaults as Milestone 2's medical module, kept
    # independently configurable since KB documents can differ widely
    # in structure from medical Q&A pairs).
    chunk_size: int = field(default_factory=lambda: _get_int("KB_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("KB_CHUNK_OVERLAP", 120))

    # Embeddings -- reuses modules.medical's SentenceTransformerEmbedder
    # (same model by default) for the primary backend.
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "KB_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Fixed-width fallback embedding dimension when Sentence
    # Transformers isn't available. Unlike Milestone 2's TF-IDF
    # fallback (fit once on a static corpus), the KB needs a backend
    # that can embed brand-new, never-seen text incrementally without
    # refitting -- so the fallback here is a stateless hashing
    # vectorizer instead (see embeddings.py).
    hashing_dimension: int = field(default_factory=lambda: _get_int("KB_HASHING_DIMENSION", 512))

    # Retrieval
    top_k: int = field(default_factory=lambda: _get_int("KB_TOP_K", 4))
    min_similarity: float = field(default_factory=lambda: _get_float("KB_MIN_SIMILARITY", 0.05))


kb_config = KnowledgeBaseConfig()


def ensure_kb_directories() -> None:
    kb_config.documents_dir.mkdir(parents=True, exist_ok=True)
    kb_config.vector_store_dir.mkdir(parents=True, exist_ok=True)
