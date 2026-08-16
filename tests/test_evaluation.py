from __future__ import annotations

import pytest

from evaluation.evaluate import (
    EvaluationItem,
    QuestionResult,
    _parse_dataset,
    expected_page_hit,
    hit_rate,
)


def _result(hit: bool) -> QuestionResult:
    item = EvaluationItem("Q", "Question?", (1,))
    return QuestionResult(item, (1,) if hit else (2,), hit)


def test_expected_page_matching_uses_any_expected_page() -> None:
    assert expected_page_hit((2, 4), (7, 4, 1)) is True
    assert expected_page_hit((2, 4), (7, 3, 1)) is False


def test_hit_rate_uses_questions_with_a_hit_over_total() -> None:
    assert hit_rate([_result(True), _result(False), _result(True)]) == pytest.approx(2 / 3)


def test_hit_rate_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="empty"):
        hit_rate([])


@pytest.mark.parametrize("dataset", [[], [{"id": "Q1", "question": "", "expected_pages": []}]])
def test_dataset_must_be_non_empty_and_valid(dataset) -> None:
    with pytest.raises(ValueError, match="dataset|Invalid"):
        _parse_dataset(dataset)
