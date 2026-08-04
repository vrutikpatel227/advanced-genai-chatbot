"""
modules/research/research_pipeline.py

Question-answering orchestration for the Research Assistant:

    question -> retrieve relevant passages -> build grounded prompt
    -> existing configurable LLM provider -> answer + citations

Reuses utils/llm_client.py's get_chat_completion() directly -- this
module never imports Groq/OpenAI/Gemini SDKs, per the PRD's "Do NOT
hardcode Groq, OpenAI, or Gemini" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.llm_client import ChatMessage, get_chat_completion
from utils.logger import get_logger

from .citation import Citation, build_citations
from .manager import ResearchManager
from .retriever import RetrievedPassage

logger = get_logger(__name__)

RESEARCH_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about uploaded "
    "scientific papers. Answer ONLY using the retrieved excerpts provided "
    "below -- never rely on your own general knowledge to add facts beyond "
    "what's in the excerpts. If the excerpts don't contain enough "
    "information to answer confidently, say so honestly and refuse to "
    "guess, rather than hallucinating an answer. Mention which paper(s) "
    "your answer draws from."
)

NO_CONTEXT_NOTICE = (
    "(No relevant passages were found in the indexed research papers for "
    "this question.)"
)


@dataclass(frozen=True)
class ResearchAnswer:
    answer: str
    citations: list[Citation]
    had_context: bool


def build_research_prompt(question: str, retrieved: list[RetrievedPassage]) -> list[ChatMessage]:
    if not retrieved:
        context_block = NO_CONTEXT_NOTICE
    else:
        parts = [
            f"[Excerpt {i} | Paper: {item.chunk.filename}]\n{item.chunk.text}"
            for i, item in enumerate(retrieved, start=1)
        ]
        context_block = "\n\n".join(parts)

    user_content = (
        f"Retrieved excerpts:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, and mention which paper(s) "
        "support your answer. If they don't contain the answer, say you "
        "don't have enough evidence rather than guessing."
    )

    return [
        ChatMessage(role="system", content=RESEARCH_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


class ResearchRAGPipeline:
    """Thin orchestration layer over ResearchManager: builds the
    grounded prompt from retrieved passages and calls the existing
    configurable LLM client."""

    def __init__(self, manager: ResearchManager) -> None:
        self._manager = manager

    def answer(self, question: str) -> ResearchAnswer:
        """Raises ValueError for empty input. Lets
        LLMConfigurationError/LLMRequestError propagate -- callers (the
        UI) already know how to handle these from Milestones 1-2."""
        if not question or not question.strip():
            raise ValueError("Please enter a research question.")

        retrieved = self._manager.retrieve(question)
        messages = build_research_prompt(question, retrieved)

        answer_text = get_chat_completion(messages)
        citations = build_citations(retrieved)

        logger.info(
            "Research QA answered (had_context=%s, %d citation(s)).",
            bool(retrieved), len(citations),
        )
        return ResearchAnswer(answer=answer_text, citations=citations, had_context=bool(retrieved))
