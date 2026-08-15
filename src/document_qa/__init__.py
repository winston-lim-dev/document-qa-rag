"""Core package for the document QA application."""

from .ingestion import IngestionError, ingest_pdf
from .models import Chunk, Document, RetrievalResult
from .retrieval import DocumentRetriever

__all__ = [
    "Chunk",
    "Document",
    "DocumentRetriever",
    "IngestionError",
    "RetrievalResult",
    "ingest_pdf",
]
