"""
modules/research/parser.py

PDF validation + text extraction for research papers. Per the PRD,
only PDF is supported for this milestone (unlike the Dynamic Knowledge
Base's PDF/TXT/MD). Text extraction itself is reused directly from
modules.knowledge_base.parser.extract_text() -- there is no
research-specific PDF-parsing logic to reimplement, so importing it
is the correct "avoid duplicate implementations" choice. Only the
validation rule (PDF-only, research-specific size limit) is new here.
"""

from __future__ import annotations

from modules.knowledge_base.parser import ParsingError, extract_text  # reused, not duplicated

from .config import SUPPORTED_EXTENSIONS, ResearchConfig, research_config

__all__ = ["InvalidPaperError", "ParsingError", "validate_paper", "extract_text"]


class InvalidPaperError(Exception):
    """Raised for a non-PDF upload, oversized file, or empty upload."""


def validate_paper(filename: str, file_bytes: bytes, config: ResearchConfig | None = None) -> str:
    """Validate an uploaded research paper and return its normalized
    extension (always '.pdf' if valid). Raises InvalidPaperError with a
    friendly message on any problem."""
    cfg = config or research_config

    if not filename or "." not in filename:
        raise InvalidPaperError("The uploaded file has no recognizable extension.")

    extension = "." + filename.rsplit(".", 1)[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidPaperError(
            f"Unsupported file type '{extension}'. The Research Assistant only "
            f"accepts PDF files."
        )

    if not file_bytes:
        raise InvalidPaperError(f"'{filename}' is empty (0 bytes).")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > cfg.kb.max_file_size_mb:
        raise InvalidPaperError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{cfg.kb.max_file_size_mb} MB limit."
        )

    return extension
