"""
modules/multimodal/config.py

Configuration for the Multimodal AI Assistant, read from environment
variables. Self-contained inside modules/multimodal/ (root config.py
is outside this PRD's folder scope, same pattern already established
by modules/medical, modules/knowledge_base, and modules/research).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class MultimodalConfig:
    """Settings for the Multimodal AI Assistant. Nothing here is
    hardcoded elsewhere in the module -- every path/limit is
    overridable via .env."""

    uploads_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MULTIMODAL_UPLOADS_DIR", str(_PROJECT_ROOT / "data" / "multimodal" / "uploads"))
        )
    )
    max_file_size_mb: int = field(
        default_factory=lambda: _get_int("MULTIMODAL_MAX_FILE_SIZE_MB", 10)
    )


multimodal_config = MultimodalConfig()


def ensure_multimodal_directories() -> None:
    multimodal_config.uploads_dir.mkdir(parents=True, exist_ok=True)


def mime_type_for_extension(extension: str) -> str:
    return _MIME_TYPES.get(extension.lower(), "application/octet-stream")
