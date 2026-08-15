"""Data models shared by document ingestion code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Stable identity and display name for an ingested document."""

    document_id: str
    filename: str


@dataclass(frozen=True)
class Chunk:
    """A page-aware piece of an ingested document."""

    document_id: str
    filename: str
    page: int
    chunk_id: str
    text: str
