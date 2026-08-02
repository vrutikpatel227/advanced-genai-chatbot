"""
modules/medical/config.py

Configuration for the Medical Knowledge Assistant (RAG), read from
environment variables -- consistent with the rest of the project's
"no hardcoded values" rule, but kept self-contained inside
modules/medical/ rather than added to the root config.py, since this
milestone's PRD scopes changes to modules/medical/, utils/, database/,
components/, README.md, requirements.txt, and app.py (root config.py
is not in that list).

load_dotenv() is called again here (idempotent) so this module works
correctly even if imported before the root config.py in some future
refactor.
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


@dataclass(frozen=True)
class MedicalConfig:
    """Settings for the Medical Knowledge Assistant. Nothing here is
    hardcoded -- every path/limit/model name is overridable via .env."""

    # Where the MedQuAD dataset lives / gets downloaded to. Never
    # hardcoded elsewhere in the module -- always read from here.
    dataset_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MEDICAL_DATASET_DIR", str(_PROJECT_ROOT / "data" / "medical" / "medquad_raw"))
        )
    )
    dataset_cache_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MEDICAL_DATASET_CACHE_PATH",
                str(_PROJECT_ROOT / "data" / "medical" / "medquad_processed.json"),
            )
        )
    )
    dataset_download_url: str = field(
        default_factory=lambda: os.getenv(
            "MEDICAL_DATASET_DOWNLOAD_URL",
            "https://codeload.github.com/abachaa/MedQuAD/zip/refs/heads/master",
        )
    )

    # Caps to keep first-run indexing fast; 0 = no limit (process the
    # full ~47k-QA-pair dataset). Fully configurable, not a hardcoded
    # restriction on the pipeline's capability.
    max_source_files: int = field(
        default_factory=lambda: _get_int("MEDICAL_MAX_SOURCE_FILES", 400)
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: _get_int("MEDICAL_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("MEDICAL_CHUNK_OVERLAP", 120))

    # Embeddings
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "MEDICAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    tfidf_max_features: int = field(
        default_factory=lambda: _get_int("MEDICAL_TFIDF_MAX_FEATURES", 512)
    )

    # Retrieval
    top_k: int = field(default_factory=lambda: _get_int("MEDICAL_TOP_K", 4))
    min_similarity: float = field(
        default_factory=lambda: float(os.getenv("MEDICAL_MIN_SIMILARITY", "0.05"))
    )

    # Vector store location (reuses the project's existing vector_store/ dir)
    vector_store_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MEDICAL_VECTOR_STORE_DIR", str(_PROJECT_ROOT / "vector_store" / "medical"))
        )
    )


medical_config = MedicalConfig()


def ensure_medical_directories() -> None:
    medical_config.dataset_dir.mkdir(parents=True, exist_ok=True)
    medical_config.dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
    medical_config.vector_store_dir.mkdir(parents=True, exist_ok=True)
