"""
modules/research

Milestone 4: Research Assistant.

Public API:
    ResearchManager       - upload/retrieve/summarize/manage orchestration
    ResearchRAGPipeline    - grounded question-answering
    ResearchAnswer          - result of pipeline.answer()
    PaperSummary             - result of manager.summarize()
    Citation                  - a single citation in an answer/summary
    render_research_assistant_page - the Streamlit page
    InvalidPaperError          - non-PDF/oversized/empty upload
    ParsingError                 - text extraction failure
    VectorStoreError               - index build/load/search/delete failure
"""

from .citation import Citation
from .manager import PaperRecord, ResearchManager, ResearchStats, UploadResult
from .parser import InvalidPaperError, ParsingError
from .research_chat import render_research_assistant_page
from .research_pipeline import ResearchAnswer, ResearchRAGPipeline
from .summarizer import PaperSummary
from .vector_store import VectorStoreError

__all__ = [
    "ResearchManager",
    "ResearchRAGPipeline",
    "ResearchAnswer",
    "PaperSummary",
    "Citation",
    "UploadResult",
    "PaperRecord",
    "ResearchStats",
    "render_research_assistant_page",
    "InvalidPaperError",
    "ParsingError",
    "VectorStoreError",
]
