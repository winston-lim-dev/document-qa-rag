"""PDF text extraction and deterministic chunking."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from typing import BinaryIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .models import Chunk

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

PdfContent = bytes | bytearray | memoryview | BinaryIO


class IngestionError(ValueError):
    """Raised when a PDF cannot produce any usable text chunks."""


def ingest_pdf(
    content: PdfContent,
    filename: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Extract and chunk a PDF while retaining stable page provenance."""
    pdf_bytes = _read_content(content)
    document_id = sha256(pdf_bytes).hexdigest()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[Chunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue

        for chunk_index, text in enumerate(splitter.split_text(page_text)):
            if not text.strip():
                continue
            chunk_id = _make_chunk_id(
                document_id=document_id,
                page=page_number,
                chunk_index=chunk_index,
                text=text,
            )
            chunks.append(
                Chunk(
                    document_id=document_id,
                    filename=filename,
                    page=page_number,
                    chunk_id=chunk_id,
                    text=text,
                )
            )

    if not chunks:
        raise IngestionError("The PDF contains no extractable text.")

    return chunks


def _read_content(content: PdfContent) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, (bytearray, memoryview)):
        return bytes(content)

    position = content.tell() if content.seekable() else None
    data = content.read()
    if position is not None:
        content.seek(position)
    if not isinstance(data, bytes):
        raise TypeError("PDF content must be binary.")
    return data


def _make_chunk_id(
    *, document_id: str, page: int, chunk_index: int, text: str
) -> str:
    identity = f"{document_id}\0{page}\0{chunk_index}\0{text}".encode("utf-8")
    return sha256(identity).hexdigest()
