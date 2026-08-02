"""
modules/medical

Milestone 2: Medical Knowledge Assistant (RAG).

Public API:
    MedicalRAGPipeline    - orchestrates retrieval + LLM generation
    MedicalAnswer          - result of pipeline.answer()
    MedicalSource            - a single retrieved source in that result
    render_medical_chat_page - the Streamlit page
    DatasetLoadError          - raised on missing/corrupted dataset
    VectorStoreError           - raised on vector index build/load failure
"""

from .loader import DatasetLoadError, MedQuADLoader, MedicalQAPair
from .medical_chat import render_medical_chat_page
from .rag_pipeline import MedicalAnswer, MedicalRAGPipeline, MedicalSource
from .vector_store import MedicalVectorStore, VectorStoreError

__all__ = [
    "MedicalRAGPipeline",
    "MedicalAnswer",
    "MedicalSource",
    "render_medical_chat_page",
    "DatasetLoadError",
    "VectorStoreError",
    "MedQuADLoader",
    "MedicalQAPair",
    "MedicalVectorStore",
]
