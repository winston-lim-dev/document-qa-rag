"""Embedding, indexing, and provenance-aware Chroma retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .models import Chunk, RetrievalResult

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "document_chunks"

EmbeddingFunction = Callable[[list[str]], Sequence[Sequence[float]]]


class DocumentRetriever:
    """Index and retrieve structured document chunks through Chroma."""

    def __init__(
        self,
        collection: Collection,
        embed: EmbeddingFunction,
    ) -> None:
        self._collection = collection
        self._embed = embed

    @classmethod
    def persistent(
        cls,
        *,
        path: str | PathLike[str] = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> DocumentRetriever:
        """Create a persistent retriever using the application's embedding model."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_or_create_collection(name=collection_name)

        def embed(texts: list[str]) -> Sequence[Sequence[float]]:
            return model.encode(texts).tolist()

        return cls(collection, embed)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Upsert chunks without deleting other documents from the collection."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=list(self._embed(texts)),
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """Return Chroma results in vector-search order with exact provenance."""
        if not question.strip():
            raise ValueError("Question must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = list(self._embed([question]))
        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_result_list(result, "ids")
        documents = _first_result_list(result, "documents")
        metadatas = _first_result_list(result, "metadatas")
        distances = _first_result_list(result, "distances")

        retrieved: list[RetrievalResult] = []
        for vector_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if not isinstance(metadata, dict):
                raise ValueError("Retrieved chunk metadata is missing.")
            chunk_id = _metadata_value(metadata, "chunk_id", str)
            if chunk_id != vector_id:
                raise ValueError("Retrieved chunk ID does not match its vector-store ID.")
            retrieved.append(
                RetrievalResult(
                    chunk=Chunk(
                        document_id=_metadata_value(metadata, "document_id", str),
                        filename=_metadata_value(metadata, "filename", str),
                        page=_metadata_value(metadata, "page", int),
                        chunk_id=chunk_id,
                        text=str(text),
                    ),
                    distance=float(distance),
                )
            )
        return retrieved


def _first_result_list(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key)
    if not values or values[0] is None:
        return []
    return list(values[0])


def _metadata_value(
    metadata: dict[str, Any], key: str, expected_type: type
) -> Any:
    value = metadata.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(f"Retrieved chunk metadata has invalid {key!r}.")
    return value
