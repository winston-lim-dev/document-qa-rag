# Document QA RAG — Portfolio Refactor

## Purpose

Document QA RAG began as an exploratory first AI project and currently demonstrates a working local Retrieval-Augmented Generation pipeline.

The purpose of this refactor is **not** to turn the project into a large production platform.

The objective is to make it a compact, credible portfolio project that demonstrates:

* clean Python application structure;
* document ingestion and provenance;
* semantic retrieval;
* grounded LLM question answering;
* evidence-aware source reporting;
* basic RAG evaluation;
* automated testing;
* clear engineering decisions.

Bellcrank remains the higher-priority project and should receive substantially more long-term development effort.

This refactor must therefore remain deliberately bounded.

---

## Status

**Completed — August 2026**

The bounded portfolio refactor is complete.

Final verification:

- 29 automated tests passing
- deterministic document and chunk provenance
- idempotent Chroma indexing
- grounded QA with explicit evidence
- deliberate insufficient-evidence handling
- retrieval evaluation using a fixed public-domain corpus
- documented retrieval configuration comparison
- thin Streamlit presentation layer
- portfolio-focused README

Retrieval evaluation results:

| Configuration | Chunk Size | Overlap | Hit Rate@3 |
|---|---:|---:|---:|
| A | 300 | 60 | 0.92 (11/12) |
| B | 500 | 100 | **1.00 (12/12)** |
| C | 800 | 160 | 0.92 (11/12) |

The retained default configuration is:

- `chunk_size=500`
- `chunk_overlap=100`
- `top_k=3`

The evaluation corpus is intentionally small and should not be interpreted as a comprehensive RAG benchmark.

No further major development is currently planned. The project is considered **portfolio-ready and in maintenance mode**.

---

## Current Baseline

The current application supports:

* PDF upload;
* PDF text extraction;
* recursive text chunking;
* local Sentence Transformer embeddings;
* ChromaDB vector retrieval;
* Ollama-based answer generation;
* page metadata;
* display of retrieved chunks and source pages;
* Streamlit UI.

The current implementation is primarily contained in a single `app.py`.

Application concerns including ingestion, embeddings, persistence, retrieval, generation, prompting, and UI are currently coupled together.

There is no current automated test suite in the repository.

---

## Portfolio Objective

The finished project should demonstrate that a small RAG system can be engineered deliberately rather than merely assembled from AI libraries.

The project should be easy to:

1. understand;
2. run locally;
3. test;
4. evaluate;
5. demonstrate;
6. explain during an interview or technical discussion.

The project does not need enterprise-scale architecture.

---

## Refactor Principles

### 1. Keep the architecture small

Separate major responsibilities without introducing unnecessary abstractions.

A suitable target is approximately:

```text
src/
    document_qa/
        __init__.py
        models.py
        ingestion.py
        retrieval.py
        generation.py
        service.py

app.py

tests/
    test_ingestion.py
    test_retrieval.py
    test_service.py

evaluation/
    dataset.json
    evaluate.py
```

Additional files may be introduced when clearly justified, but the project should remain compact.

### 2. Keep Streamlit thin

`app.py` should primarily handle:

* user input;
* presentation;
* application/service invocation;
* user-visible errors.

It should not directly own:

* PDF parsing;
* chunking rules;
* embeddings;
* vector-store operations;
* retrieval logic;
* prompt construction;
* LLM interaction.

### 3. Preserve evidence provenance

Each indexed chunk should retain enough metadata to identify where it came from.

At minimum:

```text
document_id
filename
page
chunk_id
text
```

Retrieval results should retain their associated provenance.

### 4. Ground answers in retrieved evidence

The QA service should distinguish between:

```text
sufficient evidence
```

and:

```text
insufficient evidence
```

The system should avoid encouraging the model to invent a plausible answer when the indexed documents do not contain sufficient support.

### 5. Treat citations as evidence references

Displayed sources should correspond to evidence actually retrieved for the answer.

Source metadata should be explicit rather than inferred only from a set of retrieved page numbers.

### 6. Make indexing safe and deterministic

Uploading or querying a document should not unnecessarily re-embed the same document on every Streamlit rerun.

Document identity and chunk identity should be stable enough to support predictable indexing behavior.

### 7. Test normal logic without requiring a live LLM

Unit tests should not require Ollama to be running.

External AI and vector-store boundaries should be replaceable or controllable sufficiently for deterministic tests.

Do not introduce a large dependency-injection framework solely for testing.

---

## Required Portfolio Scope

The following are required before this refactor is considered complete.

### Architecture

* Move core logic from `app.py` into a small Python package.
* Separate ingestion, retrieval, generation, and QA orchestration responsibilities.
* Keep Streamlit as a thin application boundary.

### Document Ingestion

* Extract page-aware PDF text.
* Produce deterministic chunks.
* Preserve document, page, and chunk provenance.
* Handle empty or unusable PDFs cleanly.

### Retrieval

* Preserve retrieval scores or equivalent relevance information where available.
* Make retrieval `top_k` configurable within the application layer rather than hard-coded throughout the code.
* Support deterministic testing of retrieval behavior.

### Grounded QA

* Build prompts from explicit retrieved evidence.
* Provide an insufficient-evidence response when appropriate.
* Return answer information together with source/evidence metadata.

### Testing

Add automated tests covering at least:

* PDF/page extraction behavior;
* chunk metadata;
* deterministic identifiers where applicable;
* retrieval result handling;
* insufficient-evidence behavior;
* evidence/source mapping;
* QA orchestration without requiring a real LLM.

### Evaluation

Create a small curated evaluation dataset containing:

* document or corpus identifier;
* question;
* expected relevant page or chunk evidence.

Implement a lightweight retrieval evaluator.

At least one retrieval metric should be reported, preferably:

* Hit Rate@k; and/or
* Recall@k.

Perform at least one documented comparison of retrieval configuration, such as:

* chunk size;
* chunk overlap;
* `top_k`.

The objective is to demonstrate measurement, not build an evaluation platform.

### Documentation

Rewrite the README to clearly explain:

* project purpose;
* architecture;
* local-first design;
* RAG pipeline;
* evaluation approach;
* test strategy;
* setup;
* screenshots/demo;
* known limitations;
* important engineering decisions.

---

## Explicitly Out of Scope

Do not implement these during the portfolio refactor unless a required feature proves impossible without them:

* cloud deployment;
* authentication;
* production multi-user support;
* agent frameworks;
* LangGraph;
* autonomous tool use;
* conversation memory;
* sophisticated chat-history management;
* knowledge graphs;
* multiple vector-database implementations;
* multiple embedding-provider implementations;
* complex reranking;
* hybrid lexical/vector search;
* large evaluation dashboards;
* extensive observability infrastructure;
* microservices;
* Kubernetes;
* commercial features.

---

## Optional Only If Cheap

These may be added near the end only if they require little additional work and clearly improve the portfolio:

* Docker;
* GitHub Actions CI;
* basic structured logging;
* simple configuration file/environment handling;
* multi-PDF indexing.

None should delay completion.

---

## Definition of Done

The portfolio refactor is complete when:

1. the application still performs useful local document QA;
2. core logic is independent of Streamlit;
3. document and chunk provenance are explicit;
4. answers are grounded in retrieved evidence;
5. insufficient evidence is handled deliberately;
6. meaningful automated tests pass;
7. a small reproducible retrieval evaluation exists;
8. at least one retrieval configuration comparison is documented;
9. the README clearly communicates architecture, evaluation, limitations, and engineering decisions;
10. the project can be demonstrated and explained confidently without additional major development.

Once these conditions are met, stop expanding the project.

Future improvements should be driven by a demonstrated portfolio or practical need rather than by available RAG features.

---

## Implementation Sequence

### Step 1 — Engineering baseline

* create package structure;
* define core data models;
* move PDF ingestion out of Streamlit;
* add initial tests.

### Step 2 — Retrieval boundary

* isolate embeddings and Chroma interaction;
* introduce explicit retrieval results with provenance;
* test retrieval behavior.

### Step 3 — Grounded QA

* isolate Ollama generation;
* create QA orchestration;
* implement insufficient-evidence behavior;
* connect sources to returned evidence.

### Step 4 — Evaluation

* create small evaluation dataset;
* implement retrieval metric;
* run one or two configuration comparisons;
* select reasonable defaults based on evidence.

### Step 5 — Portfolio finish

* simplify Streamlit around the new service layer;
* improve error handling;
* rewrite README;
* update screenshots;
* perform final test and repository review.

---

## Scope Rule

If a proposed change does not materially improve:

* AI/RAG engineering evidence;
* software-engineering credibility;
* reliability;
* testability;
* evaluation;
* or portfolio presentation,

defer it.

Bellcrank remains the main long-term development priority.
