"""
modules/knowledge_base

Milestone 3: Dynamic Knowledge Base.

Public API:
    KnowledgeBaseManager       - upload/search/stats/index-control orchestration
    UploadResult                - result of manager.process_upload()
    RetrievedKnowledgeChunk      - a single search result
    render_knowledge_base_page    - the Streamlit page
    InvalidFileError               - unsupported/oversized/empty file
    ParsingError                    - text extraction failure
    VectorStoreError                 - index build/load/search failure
"""

from .knowledge_chat import render_knowledge_base_page
from .manager import KnowledgeBaseManager, RetrievedKnowledgeChunk, UploadResult
from .parser import InvalidFileError, ParsingError
from .vector_store import VectorStoreError

__all__ = [
    "KnowledgeBaseManager",
    "UploadResult",
    "RetrievedKnowledgeChunk",
    "render_knowledge_base_page",
    "InvalidFileError",
    "ParsingError",
    "VectorStoreError",
]
