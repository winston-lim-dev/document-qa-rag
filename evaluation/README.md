# Lightweight Retrieval Evaluation

This evaluation demonstrates that retrieval settings are compared with a reproducible metric instead of selected only by intuition. It evaluates retrieval separately from answer generation so that evidence selection can be measured without Ollama or the additional variability of generated answers.

## Corpus and attribution

The checked-in eight-page PDF contains selected, unmodified constitutional passages arranged by topic. It was transcribed from:

- National Archives
- *The Constitution of the United States*
- https://www.archives.gov/founding-docs/constitution
- Transcript: https://www.archives.gov/founding-docs/constitution-transcript

The National Archives is credited as the original source. The fixture contains text rather than the large high-resolution document images. `generate_fixture.py` recreates the PDF deterministically using Python's standard library; evaluation runs themselves are completely local.

## Dataset and metric

`dataset.json` contains 12 manually curated direct, paraphrased, and concept-oriented questions. Expected pages were assigned from the fixture's known topic layout before retrieval results were used.

Hit Rate@k is:

```text
questions where at least one top-k chunk comes from an expected page
-------------------------------------------------------------------
                         total questions
```

The primary comparison uses `top_k=3`:

| Configuration | Chunk size | Chunk overlap |
|---|---:|---:|
| A | 300 | 60 |
| B | 500 | 100 |
| C | 800 | 160 |

## Run

From the repository root:

```text
python evaluation/evaluate.py
```

Use `--configuration A`, `B`, or `C` to run one configuration. The command loads `all-MiniLM-L6-v2`, uses an in-memory Chroma collection, prints every question's expected and retrieved pages, and does not access application data or Ollama.

## Observed results

The real evaluation with `all-MiniLM-L6-v2` produced:

| Configuration | Hit Rate@3 | Hits |
|---|---:|---:|
| A (`300/60`) | 0.92 | 11/12 |
| B (`500/100`) | 1.00 | 12/12 |
| C (`800/160`) | 0.92 | 11/12 |

Configuration B is preferred because it retrieved an expected page for all 12 questions and retains the application's existing `chunk_size=500` and `chunk_overlap=100` defaults. Configurations A and C each missed Q02, the question about the two chambers of Congress. Hit Rate@3 does not provide evidence for selecting a `max_distance` threshold, so no threshold is recommended here.

## Limitations

This intentionally small dataset and selected-text fixture are a portfolio demonstration, not a comprehensive RAG benchmark. Page-level Hit Rate@3 measures whether relevant evidence was retrieved, not answer correctness, faithfulness, ranking quality within the top three, or calibrated distance thresholds. A larger and more difficult dataset would be needed to make a stronger tuning claim.
