"""
modules/medical/prompts.py

Prompt construction for the medical RAG pipeline. The system prompt is
deliberately strict: the LLM must answer only from the retrieved
passages and say so honestly when it doesn't have enough grounded
information, per the PRD's "must never rely solely on the LLM's
internal knowledge" requirement.
"""

from __future__ import annotations

from utils.llm_client import ChatMessage

from .retriever import RetrievedChunk

MEDICAL_SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer the user's question "
    "using ONLY the retrieved reference passages provided below -- never "
    "add facts from your own general/internal knowledge beyond what's in "
    "the passages. If the passages don't contain enough information to "
    "answer confidently, say so honestly instead of guessing. Be clear, "
    "concise, and factual. Do not diagnose, prescribe, or give "
    "personalized medical advice -- only relay the informational content "
    "in the retrieved passages. Always keep in mind this is for "
    "educational purposes only, not a substitute for professional medical care."
)

NO_CONTEXT_NOTICE = (
    "(No relevant reference passages were found in the medical knowledge "
    "base for this question.)"
)


def build_medical_prompt(question: str, retrieved: list[RetrievedChunk]) -> list[ChatMessage]:
    """Build the system + user messages for the medical RAG LLM call."""
    if not retrieved:
        context_block = NO_CONTEXT_NOTICE
    else:
        parts = [
            f"[Passage {i} | Source: {item.chunk.source} | Topic: {item.chunk.focus or 'N/A'}]\n{item.chunk.text}"
            for i, item in enumerate(retrieved, start=1)
        ]
        context_block = "\n\n".join(parts)

    user_content = (
        f"Retrieved reference passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above. If they don't contain the "
        "answer, say you don't have enough verified information to answer "
        "confidently, rather than guessing."
    )

    return [
        ChatMessage(role="system", content=MEDICAL_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
