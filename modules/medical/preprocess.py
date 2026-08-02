"""
modules/medical/preprocess.py

Text processing pipeline: cleans raw MedQuAD answer text, splits long
answers into overlapping chunks (via LangChain's
RecursiveCharacterTextSplitter, per the PRD's "Framework: LangChain"
requirement), and preserves metadata (question, focus/topic, source,
URL) on every chunk so retrieval results can always be attributed.

Reusable: any future milestone that needs to turn a list of
(question, answer, source) records into embeddable chunks can reuse
clean_text() / chunk_qa_pairs() directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import MedicalConfig, medical_config
from .loader import MedicalQAPair

_WHITESPACE_RE = re.compile(r"\s+")
# MedQuAD XML sometimes contains literal " - " bullet artifacts from
# list flattening (visible in the raw dataset); collapse repeated
# separators without altering actual medical content.
_REPEATED_DASH_RE = re.compile(r"(\s-\s){2,}")


@dataclass(frozen=True)
class MedicalChunk:
    chunk_id: str
    text: str           # the actual text that gets embedded/searched
    question: str
    focus: str
    source: str
    url: str
    doc_id: str
    pair_id: str


def clean_text(text: str) -> str:
    """Normalize whitespace and strip minor XML-flattening artifacts.
    Does not alter medical content/meaning."""
    if not text:
        return ""
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    cleaned = _REPEATED_DASH_RE.sub(" ", cleaned)
    return cleaned


def chunk_qa_pairs(
    pairs: list[MedicalQAPair],
    config: MedicalConfig | None = None,
) -> list[MedicalChunk]:
    """Clean and split each QA pair's answer into one or more chunks,
    each carrying the full metadata needed to display a source and
    answer questions about where an answer came from."""
    cfg = config or medical_config
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[MedicalChunk] = []
    for pair in pairs:
        question = clean_text(pair.question)
        answer = clean_text(pair.answer)
        if not question or not answer:
            continue

        pieces = splitter.split_text(answer) if len(answer) > cfg.chunk_size else [answer]
        for i, piece in enumerate(pieces):
            # Prefixing with the question gives the embedder useful
            # context even for a chunk taken from the middle of a long
            # answer, and improves retrieval relevance.
            chunk_text = f"Question: {question}\nAnswer: {piece}"
            chunks.append(
                MedicalChunk(
                    chunk_id=f"{pair.pair_id}::{i}",
                    text=chunk_text,
                    question=question,
                    focus=pair.focus,
                    source=pair.source,
                    url=pair.url,
                    doc_id=pair.doc_id,
                    pair_id=pair.pair_id,
                )
            )
    return chunks
