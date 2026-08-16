from __future__ import annotations

import pytest

from document_qa.models import Chunk, RetrievalResult
from document_qa.service import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    DocumentQAService,
)


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, top_k: int = 3) -> list[RetrievalResult]:
        self.calls.append((question, top_k))
        return self.results


class FakeGenerator:
    def __init__(self, answer: str = "Grounded answer [S1]") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def _evidence(
    chunk_id: str,
    text: str,
    distance: float,
    *,
    filename: str,
    page: int,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk("document-id", filename, page, chunk_id, text),
        distance=distance,
    )


def test_answer_passes_ordered_evidence_to_generation_and_result() -> None:
    evidence = [
        _evidence("c2", "First retrieved text", 0.2, filename="alpha.pdf", page=4),
        _evidence("c1", "Second retrieved text", 0.4, filename="beta.pdf", page=7),
    ]
    retriever = FakeRetriever(evidence)
    generator = FakeGenerator()
    service = DocumentQAService(retriever, generator)

    result = service.answer("What is supported?", top_k=2)

    assert retriever.calls == [("What is supported?", 2)]
    assert result.answer == "Grounded answer [S1]"
    assert result.evidence == tuple(evidence)
    assert result.has_sufficient_evidence is True
    prompt = generator.prompts[0]
    assert "Question:\nWhat is supported?" in prompt
    assert "[S1]\nDocument: alpha.pdf\nPage: 4" in prompt
    assert "Content:\nFirst retrieved text" in prompt
    assert "[S2]\nDocument: beta.pdf\nPage: 7" in prompt
    assert "Content:\nSecond retrieved text" in prompt
    assert prompt.index("[S1]") < prompt.index("[S2]")


def test_no_distance_threshold_preserves_all_evidence() -> None:
    evidence = [
        _evidence("near", "Near", 0.1, filename="one.pdf", page=1),
        _evidence("far", "Far", 99.0, filename="one.pdf", page=2),
    ]
    service = DocumentQAService(FakeRetriever(evidence), FakeGenerator())

    result = service.answer("Question", max_distance=None)

    assert result.evidence == tuple(evidence)


def test_max_distance_filters_evidence_without_reordering() -> None:
    evidence = [
        _evidence("one", "One", 0.2, filename="one.pdf", page=1),
        _evidence("two", "Two", 0.8, filename="one.pdf", page=2),
        _evidence("three", "Three", 0.4, filename="one.pdf", page=3),
    ]
    generator = FakeGenerator()
    service = DocumentQAService(FakeRetriever(evidence), generator)

    result = service.answer("Question", max_distance=0.5)

    assert [item.chunk.chunk_id for item in result.evidence] == ["one", "three"]
    assert "One" in generator.prompts[0]
    assert "Three" in generator.prompts[0]
    assert "Two" not in generator.prompts[0]


def test_no_usable_evidence_returns_deterministic_answer_without_generation() -> None:
    evidence = [
        _evidence("far", "Too far", 0.9, filename="one.pdf", page=1),
    ]
    generator = FakeGenerator()
    service = DocumentQAService(FakeRetriever(evidence), generator)

    result = service.answer("Question", max_distance=0.5)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.evidence == ()
    assert result.has_sufficient_evidence is False
    assert generator.prompts == []


def test_empty_retrieval_is_insufficient_without_generation() -> None:
    generator = FakeGenerator()
    service = DocumentQAService(FakeRetriever([]), generator)

    result = service.answer("Question")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.has_sufficient_evidence is False
    assert generator.prompts == []


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected_before_retrieval(question: str) -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator()
    service = DocumentQAService(retriever, generator)

    with pytest.raises(ValueError, match="Question"):
        service.answer(question)

    assert retriever.calls == []
    assert generator.prompts == []
