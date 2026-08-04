"""
modules/research/manager.py

Top-level orchestration for the Research Assistant: upload -> validate
-> extract -> chunk -> embed -> index -> record workflow, plus
retrieval, citation building, summarization, and paper management
(list/delete/re-index). This is the single entry point the Streamlit
page should use -- every failure mode maps to a friendly result
instead of crashing the app.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger
from utils.storage import (
    delete_research_paper,
    get_all_research_papers,
    get_research_chunk_total,
    get_research_paper_by_doc_id,
    get_research_paper_by_hash,
    get_research_paper_count,
    save_research_paper,
)

from .chunker import ResearchChunk, chunk_paper
from .citation import Citation, build_citations
from .config import ResearchConfig, ensure_research_directories, research_config
from .parser import InvalidPaperError, ParsingError, extract_text, validate_paper
from .retriever import ResearchRetriever, RetrievedPassage
from .summarizer import PaperSummary, summarize_paper
from .vector_store import ResearchVectorStore, VectorStoreError

logger = get_logger(__name__)


@dataclass(frozen=True)
class UploadResult:
    status: str          # "success" | "duplicate" | "error"
    message: str
    filename: str
    chunk_count: int = 0
    doc_id: Optional[str] = None


@dataclass(frozen=True)
class PaperRecord:
    doc_id: str
    filename: str
    content_hash: str
    chunk_count: int
    status: str
    created_at: str


@dataclass(frozen=True)
class ResearchStats:
    total_papers: int
    total_chunks: int
    vector_store_ready: bool


def _row_to_record(row) -> PaperRecord:
    return PaperRecord(
        doc_id=row["doc_id"], filename=row["filename"], content_hash=row["content_hash"],
        chunk_count=row["chunk_count"], status=row["status"], created_at=row["created_at"],
    )


class ResearchManager:
    def __init__(self, config: Optional[ResearchConfig] = None) -> None:
        self._config = config or research_config
        self._store = ResearchVectorStore(self._config.kb)
        self._retriever: Optional[ResearchRetriever] = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            ensure_research_directories()
            self._store.ensure_ready()
            self._retriever = ResearchRetriever(self._store, self._config)
            self._loaded = True

    @property
    def is_vector_store_ready(self) -> bool:
        return self._store.index is not None

    @property
    def embedding_backend(self) -> str:
        return self._store.embedder.name if self._store.embedder else "not initialized"

    # --- upload ------------------------------------------------------------------

    def process_upload(self, filename: str, file_bytes: bytes) -> UploadResult:
        """Full pipeline for one uploaded paper. Never raises -- every
        failure mode maps to a friendly UploadResult instead."""
        self._ensure_loaded()

        try:
            extension = validate_paper(filename, file_bytes, self._config)
        except InvalidPaperError as exc:
            return UploadResult(status="error", message=str(exc), filename=filename)

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = get_research_paper_by_hash(content_hash)
        if existing is not None:
            return UploadResult(
                status="duplicate",
                message=f"'{filename}' has already been indexed (as '{existing['filename']}').",
                filename=filename,
            )

        try:
            raw_text = extract_text(filename, file_bytes, extension)
        except (ParsingError, InvalidPaperError) as exc:
            return UploadResult(status="error", message=str(exc), filename=filename)

        doc_id = str(uuid.uuid4())
        try:
            chunks = chunk_paper(doc_id, filename, raw_text, self._config)
        except Exception as exc:  # noqa: BLE001 - never let chunking failures crash the app
            logger.error("Failed to chunk '%s': %s", filename, exc)
            return UploadResult(
                status="error",
                message=f"Failed to process the text content of '{filename}'. Please try again.",
                filename=filename,
            )

        if not chunks:
            return UploadResult(
                status="error",
                message=f"'{filename}' appears to be empty after text extraction.",
                filename=filename,
            )

        try:
            added = self._store.add_chunks(chunks)
        except VectorStoreError as exc:
            logger.error("Failed to index '%s': %s", filename, exc)
            return UploadResult(
                status="error",
                message=(
                    f"Failed to generate embeddings or update the search index "
                    f"for '{filename}'. Please try again."
                ),
                filename=filename,
            )

        try:
            self._save_paper_text(doc_id, raw_text)
            save_research_paper(doc_id, filename, content_hash, added, status="indexed")
        except Exception as exc:  # noqa: BLE001 - PRD error handling: database failure
            logger.error("Failed to save metadata for '%s': %s", filename, exc)
            return UploadResult(
                status="error",
                message=(
                    f"'{filename}' was indexed, but saving its metadata failed. "
                    "It may not appear correctly in the paper list."
                ),
                filename=filename, chunk_count=added, doc_id=doc_id,
            )

        return UploadResult(
            status="success",
            message=f"'{filename}' indexed successfully ({added} chunk{'s' if added != 1 else ''}).",
            filename=filename, chunk_count=added, doc_id=doc_id,
        )

    # --- retrieval / citations -----------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedPassage]:
        self._ensure_loaded()
        return self._retriever.retrieve(query, top_k)

    def build_citations(self, retrieved: list[RetrievedPassage]) -> list[Citation]:
        return build_citations(retrieved)

    # --- summarization -------------------------------------------------------------

    def summarize(self, doc_id: str) -> PaperSummary:
        """Generate the structured summary for one paper. Raises
        ValueError if the paper doesn't exist or has no content."""
        self._ensure_loaded()
        paper = get_research_paper_by_doc_id(doc_id)
        if paper is None:
            raise ValueError("Paper not found.")
        chunks = self._store.get_chunks_for_document(doc_id)
        return summarize_paper(paper["filename"], chunks, self._config)

    # --- management ------------------------------------------------------------------

    def list_papers(self) -> list[PaperRecord]:
        return [_row_to_record(row) for row in get_all_research_papers()]

    def get_stats(self) -> ResearchStats:
        self._ensure_loaded()
        return ResearchStats(
            total_papers=get_research_paper_count(),
            total_chunks=get_research_chunk_total(),
            vector_store_ready=self.is_vector_store_ready,
        )

    def delete_paper(self, doc_id: str) -> bool:
        """Remove a paper's vectors, stored text, and metadata --
        without affecting any other indexed paper. Returns True if the
        paper existed and was deleted."""
        self._ensure_loaded()
        paper = get_research_paper_by_doc_id(doc_id)
        if paper is None:
            return False

        self._store.delete_document(doc_id)  # VectorStoreError propagates -- caller shows friendly message
        self._delete_paper_text(doc_id)
        delete_research_paper(doc_id)
        logger.info("Deleted research paper %s (%s).", doc_id, paper["filename"])
        return True

    def reindex_paper(self, doc_id: str) -> int:
        """Re-chunk and re-embed one paper from its stored text.
        Returns the new chunk count. Raises ValueError if the paper
        doesn't exist or has no stored text available."""
        self._ensure_loaded()
        paper = get_research_paper_by_doc_id(doc_id)
        if paper is None:
            raise ValueError("Paper not found.")

        text = self._load_paper_text(doc_id)
        if text is None:
            raise ValueError(f"No stored text found for '{paper['filename']}' -- cannot re-index.")

        self._store.delete_document(doc_id)
        chunks = chunk_paper(doc_id, paper["filename"], text, self._config)
        added = self._store.add_chunks(chunks)
        save_research_paper(doc_id, paper["filename"], paper["content_hash"], added, status="indexed")
        return added

    # --- stored text persistence (enables re-index) -----------------------------------

    def _save_paper_text(self, doc_id: str, text: str) -> None:
        self._config.kb.documents_dir.mkdir(parents=True, exist_ok=True)
        (self._config.kb.documents_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    def _load_paper_text(self, doc_id: str) -> Optional[str]:
        path = self._config.kb.documents_dir / f"{doc_id}.txt"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read stored text for %s: %s", doc_id, exc)
            return None

    def _delete_paper_text(self, doc_id: str) -> None:
        path = self._config.kb.documents_dir / f"{doc_id}.txt"
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning("Failed to delete stored text file for %s: %s", doc_id, exc)
