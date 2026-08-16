"""Grounded question-answering orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Protocol

from .models import QAResult, RetrievalResult

INSUFFICIENT_EVIDENCE_ANSWER = (
    "The indexed documents do not contain enough relevant information "
    "to answer this question."
)


class _Retriever(Protocol):
    def retrieve(
        self, question: str, *, top_k: int = 3
    ) -> list[RetrievalResult]: ...


class DocumentQAService:
    """Retrieve evidence and generate answers grounded only in that evidence."""

    def __init__(
        self,
        retriever: _Retriever,
        generator: Callable[[str], str],
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(
        self,
        question: str,
        *,
        top_k: int = 3,
        max_distance: float | None = None,
    ) -> QAResult:
        """Answer from retrieved evidence, optionally filtering by distance."""
        if not question.strip():
            raise ValueError("Question must not be empty.")
        if max_distance is not None and (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, (int, float))
            or not isfinite(max_distance)
            or max_distance < 0
        ):
            raise ValueError("max_distance must be a finite non-negative number.")

        retrieved = self._retriever.retrieve(question, top_k=top_k)
        evidence = tuple(
            result
            for result in retrieved
            if max_distance is None or result.distance <= max_distance
        )
        if not evidence:
            return QAResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=(),
                has_sufficient_evidence=False,
            )

        prompt = build_grounded_prompt(question, evidence)
        return QAResult(
            answer=self._generator(prompt),
            evidence=evidence,
            has_sufficient_evidence=True,
        )


def build_grounded_prompt(
    question: str,
    evidence: Sequence[RetrievalResult],
) -> str:
    """Build a deterministic prompt with traceable, ordered evidence blocks."""
    blocks = []
    for index, result in enumerate(evidence, start=1):
        chunk = result.chunk
        blocks.append(
            f"[S{index}]\n"
            f"Document: {chunk.filename}\n"
            f"Page: {chunk.page}\n"
            f"Content:\n{chunk.text}"
        )

    evidence_text = "\n\n".join(blocks)
    return f"""You are a grounded document assistant.

Use only the supplied evidence to answer the question.
Do not invent or infer unsupported information.
If the evidence does not support an answer, state that clearly.
Reference evidence identifiers such as [S1] where appropriate.

Evidence:
{evidence_text}

Question:
{question}

Answer:
"""
