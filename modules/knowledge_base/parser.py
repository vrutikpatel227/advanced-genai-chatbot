"""
modules/knowledge_base/parser.py

File validation + text extraction. One function per supported format,
dispatched through a small registry -- adding a new file type later
means adding one function and one registry entry, per the PRD's
"design the architecture so additional formats can be added later
without major code changes" requirement.
"""

from __future__ import annotations

import io

from utils.logger import get_logger

from .config import SUPPORTED_EXTENSIONS, KnowledgeBaseConfig, kb_config

logger = get_logger(__name__)


class InvalidFileError(Exception):
    """Raised for an unsupported file type, oversized file, or empty upload."""


class ParsingError(Exception):
    """Raised when text extraction fails (e.g. a corrupted PDF)."""


def validate_file(filename: str, file_bytes: bytes, config: KnowledgeBaseConfig | None = None) -> str:
    """Validate the uploaded file and return its normalized extension.
    Raises InvalidFileError with a friendly message on any problem."""
    cfg = config or kb_config

    if not filename or "." not in filename:
        raise InvalidFileError("The uploaded file has no recognizable extension.")

    extension = "." + filename.rsplit(".", 1)[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidFileError(
            f"Unsupported file type '{extension}'. Supported formats: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )

    if not file_bytes:
        raise InvalidFileError(f"'{filename}' is empty (0 bytes).")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > cfg.max_file_size_mb:
        raise InvalidFileError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{cfg.max_file_size_mb} MB limit."
        )

    return extension


def extract_text(filename: str, file_bytes: bytes, extension: str) -> str:
    """Extract raw text from the file. Raises ParsingError on failure
    (e.g. a corrupted/password-protected PDF) -- never crashes the app."""
    extractor = _EXTRACTORS.get(extension)
    if extractor is None:
        raise InvalidFileError(f"No text extractor registered for '{extension}'.")

    try:
        text = extractor(file_bytes)
    except ParsingError:
        raise
    except Exception as exc:  # noqa: BLE001 - any library failure -> friendly ParsingError
        logger.error("Failed to extract text from '%s': %s", filename, exc)
        raise ParsingError(f"'{filename}' appears to be corrupted or unreadable.") from exc

    return text


def _extract_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


def _extract_markdown(file_bytes: bytes) -> str:
    # Markdown is plain text for extraction purposes -- chunker.clean_text()
    # normalizes whitespace; we deliberately don't strip markdown syntax
    # (headers, links, etc.) since it carries useful semantic structure.
    return _extract_txt(file_bytes)


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParsingError(
            "The 'pypdf' package is not installed. Run: pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 - pypdf raises various error types for bad PDFs
        raise ParsingError("This PDF could not be opened -- it may be corrupted or encrypted.") from exc

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - skip unreadable individual pages, don't abort the whole file
            logger.warning("Skipping an unreadable PDF page: %s", exc)

    return "\n\n".join(pages_text)


_EXTRACTORS = {
    ".txt": _extract_txt,
    ".md": _extract_markdown,
    ".pdf": _extract_pdf,
}
