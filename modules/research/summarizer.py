"""
modules/research/summarizer.py

Generates a structured summary (Executive Summary, Research Objective,
Methodology, Key Findings, Conclusion, Future Work) for one paper,
using the paper's own indexed chunks as context and the existing
configurable LLM provider (utils/llm_client.py) -- never a
hardcoded Groq/OpenAI/Gemini call.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.llm_client import ChatMessage, get_chat_completion
from utils.logger import get_logger

from .chunker import ResearchChunk
from .config import ResearchConfig, research_config

logger = get_logger(__name__)

SUMMARY_SYSTEM_PROMPT = (
    "You are a research paper summarization assistant. Using ONLY the "
    "provided paper excerpts, produce a structured summary with exactly "
    "these section headers, in this order: Executive Summary, Research "
    "Objective, Methodology, Key Findings, Conclusion, Future Work. If "
    "the excerpts don't contain enough information for a given section, "
    "say so honestly in that section rather than inventing content. "
    "Never state anything not supported by the excerpts."
)


@dataclass(frozen=True)
class PaperSummary:
    filename: str
    text: str            # full structured summary (LLM output, markdown-formatted)
    chunk_count_used: int


def _build_context(chunks: list[ResearchChunk], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        piece = chunk.text
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def summarize_paper(
    filename: str,
    chunks_in_order: list[ResearchChunk],
    config: ResearchConfig | None = None,
) -> PaperSummary:
    """Generate the six-section structured summary for one paper.

    Raises ValueError if there's no content to summarize. Lets
    LLMConfigurationError/LLMRequestError propagate -- callers (the UI)
    already know how to handle these from Milestones 1-2.
    """
    cfg = config or research_config
    if not chunks_in_order:
        raise ValueError("No content is available to summarize for this paper.")

    limited_chunks = chunks_in_order[: cfg.max_summary_chunks]
    context = _build_context(limited_chunks, cfg.max_summary_context_chars)

    messages = [
        ChatMessage(role="system", content=SUMMARY_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                f"Paper: {filename}\n\nExcerpts:\n\n{context}\n\n"
                "Produce the structured summary now."
            ),
        ),
    ]

    summary_text = get_chat_completion(messages)
    logger.info(
        "Generated summary for '%s' using %d/%d chunks.",
        filename, len(limited_chunks), len(chunks_in_order),
    )
    return PaperSummary(filename=filename, text=summary_text, chunk_count_used=len(limited_chunks))
