"""Core package for the document QA application."""

from .ingestion import IngestionError, ingest_pdf
from .models import Chunk, Document

__all__ = ["Chunk", "Document", "IngestionError", "ingest_pdf"]
