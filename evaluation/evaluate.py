"""Run the small Constitution retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb

from document_qa.ingestion import ingest_pdf
from document_qa.retrieval import DEFAULT_EMBEDDING_MODEL, DocumentRetriever

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "us_constitution.pdf"
DATASET_PATH = Path(__file__).parent / "dataset.json"
TOP_K = 3


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    question: str
    expected_pages: tuple[int, ...]


@dataclass(frozen=True)
class Configuration:
    name: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class QuestionResult:
    item: EvaluationItem
    retrieved_pages: tuple[int, ...]
    hit: bool


CONFIGURATIONS = (
    Configuration("A", 300, 60),
    Configuration("B", 500, 100),
    Configuration("C", 800, 160),
)


def expected_page_hit(
    expected_pages: tuple[int, ...], retrieved_pages: tuple[int, ...]
) -> bool:
    """Return whether any retrieved page belongs to the curated relevant set."""
    return bool(set(expected_pages).intersection(retrieved_pages))


def hit_rate(results: list[QuestionResult]) -> float:
    """Calculate questions with an expected page retrieved divided by total."""
    if not results:
        raise ValueError("Cannot calculate Hit Rate for an empty result set.")
    return sum(result.hit for result in results) / len(results)


def load_dataset(path: Path = DATASET_PATH) -> list[EvaluationItem]:
    """Load and validate the manually curated evaluation dataset."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    return _parse_dataset(raw)


def _parse_dataset(raw: Any) -> list[EvaluationItem]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    items: list[EvaluationItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each evaluation item must be an object.")
        item_id = entry.get("id")
        question = entry.get("question")
        pages = entry.get("expected_pages")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(pages, list)
            or not pages
            or any(isinstance(page, bool) or not isinstance(page, int) or page <= 0 for page in pages)
        ):
            raise ValueError(f"Invalid evaluation item: {entry!r}")
        items.append(EvaluationItem(item_id, question, tuple(pages)))
    return items


def evaluate_configuration(
    configuration: Configuration,
    items: list[EvaluationItem],
    model: Any,
) -> list[QuestionResult]:
    """Evaluate one chunking configuration in an isolated Chroma collection."""
    chunks = ingest_pdf(
        FIXTURE_PATH.read_bytes(),
        FIXTURE_PATH.name,
        chunk_size=configuration.chunk_size,
        chunk_overlap=configuration.chunk_overlap,
    )
    client = chromadb.Client()
    collection_name = f"constitution_evaluation_{configuration.name.lower()}"
    try:
        client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.create_collection(collection_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    retriever = DocumentRetriever(collection, embed)
    retriever.index(chunks)
    results = []
    for item in items:
        retrieved = retriever.retrieve(item.question, top_k=TOP_K)
        pages = tuple(result.chunk.page for result in retrieved)
        results.append(
            QuestionResult(
                item=item,
                retrieved_pages=pages,
                hit=expected_page_hit(item.expected_pages, pages),
            )
        )
    client.delete_collection(collection_name)
    return results


def print_results(configuration: Configuration, results: list[QuestionResult]) -> None:
    """Print concise per-question outcomes and an aggregate summary."""
    print(
        f"Configuration {configuration.name}: "
        f"chunk_size={configuration.chunk_size} "
        f"chunk_overlap={configuration.chunk_overlap} top_k={TOP_K}"
    )
    for result in results:
        outcome = "PASS" if result.hit else "FAIL"
        print(
            f"{result.item.item_id} {outcome} "
            f"expected={list(result.item.expected_pages)} "
            f"retrieved={list(result.retrieved_pages)}"
        )
    hits = sum(result.hit for result in results)
    print(f"Hit Rate@{TOP_K}: {hit_rate(results):.2f} ({hits}/{len(results)})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        choices=[configuration.name for configuration in CONFIGURATIONS],
        help="Run only one named configuration (default: all).",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    items = load_dataset()
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    configurations = [
        configuration
        for configuration in CONFIGURATIONS
        if args.configuration is None or configuration.name == args.configuration
    ]
    for configuration in configurations:
        results = evaluate_configuration(configuration, items, model)
        print_results(configuration, results)


if __name__ == "__main__":
    main()
