from __future__ import annotations

from collections.abc import Sequence

import chromadb
import pytest
from chromadb.errors import NotFoundError

from document_qa.models import Chunk
from document_qa.retrieval import DocumentRetriever


class FakeEmbedder:
    """Small deterministic embedding function; no model loading is required."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


@pytest.fixture
def collection():
    client = chromadb.Client()
    name = "document_qa_retrieval_tests"
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass
    created = client.create_collection(name)
    yield created
    client.delete_collection(name)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "document-one",
    filename: str = "one.pdf",
    page: int = 1,
) -> Chunk:
    return Chunk(document_id, filename, page, chunk_id, text)


def test_index_uses_chunk_ids_and_stores_provenance(collection) -> None:
    chunk = _chunk("chunk-1", "near", page=4)
    retriever = DocumentRetriever(collection, FakeEmbedder({"near": [0.0, 0.0]}))

    retriever.index([chunk])
    stored = collection.get(ids=[chunk.chunk_id], include=["documents", "metadatas"])

    assert stored["ids"] == ["chunk-1"]
    assert stored["documents"] == ["near"]
    assert stored["metadatas"] == [{
        "document_id": "document-one",
        "filename": "one.pdf",
        "page": 4,
        "chunk_id": "chunk-1",
    }]


def test_reindex_is_idempotent_and_second_document_is_preserved(collection) -> None:
    first = _chunk("chunk-1", "near")
    second = _chunk(
        "chunk-2",
        "far",
        document_id="document-two",
        filename="two.pdf",
    )
    retriever = DocumentRetriever(
        collection,
        FakeEmbedder({"near": [0.0, 0.0], "far": [3.0, 0.0]}),
    )

    retriever.index([first])
    retriever.index([first])
    retriever.index([second])

    assert collection.count() == 2
    assert set(collection.get()["ids"]) == {"chunk-1", "chunk-2"}


def test_retrieve_preserves_order_provenance_and_chroma_distance(collection) -> None:
    chunks = [
        _chunk("far-id", "far", page=3),
        _chunk("near-id", "near", page=1),
        _chunk("middle-id", "middle", page=2),
    ]
    embedder = FakeEmbedder({
        "near": [0.0, 0.0],
        "middle": [1.0, 0.0],
        "far": [3.0, 0.0],
        "question": [0.0, 0.0],
    })
    retriever = DocumentRetriever(collection, embedder)
    retriever.index(chunks)

    results = retriever.retrieve("question", top_k=3)
    raw = collection.query(
        query_embeddings=[[0.0, 0.0]],
        n_results=3,
        include=["distances"],
    )

    assert [result.chunk.chunk_id for result in results] == [
        "near-id", "middle-id", "far-id"
    ]
    assert [result.chunk.page for result in results] == [1, 2, 3]
    assert [result.distance for result in results] == raw["distances"][0]
    assert embedder.calls[-1] == ["question"]


def test_top_k_controls_result_count(collection) -> None:
    embedder = FakeEmbedder({
        "near": [0.0, 0.0],
        "middle": [1.0, 0.0],
        "far": [3.0, 0.0],
        "question": [0.0, 0.0],
    })
    retriever = DocumentRetriever(collection, embedder)
    retriever.index([
        _chunk("one", "near"),
        _chunk("two", "middle"),
        _chunk("three", "far"),
    ])

    assert len(retriever.retrieve("question", top_k=2)) == 2


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected(collection, top_k) -> None:
    retriever = DocumentRetriever(collection, FakeEmbedder({}))

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("question", top_k=top_k)


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(collection, question: str) -> None:
    retriever = DocumentRetriever(collection, FakeEmbedder({}))

    with pytest.raises(ValueError, match="Question"):
        retriever.retrieve(question)
