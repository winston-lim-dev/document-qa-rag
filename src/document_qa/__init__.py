"""Core package for the document QA application."""

from .generation import OllamaGenerator
from .ingestion import IngestionError, ingest_pdf
from .models import Chunk, Document, QAResult, RetrievalResult
from .retrieval import DocumentRetriever
from .service import DocumentQAService

__all__ = [
    "Chunk",
    "Document",
    "DocumentQAService",
    "DocumentRetriever",
    "IngestionError",
    "OllamaGenerator",
    "QAResult",
    "RetrievalResult",
    "ingest_pdf",
]
