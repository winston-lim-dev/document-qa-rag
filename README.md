# Local RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, perform semantic search, and receive AI-generated answers with source citations.

## Features

✅ PDF Upload

✅ Document Chunking

✅ Local Embeddings (Sentence Transformers)

✅ ChromaDB Vector Search

✅ Ollama + Llama 3 Integration

✅ Semantic Retrieval

✅ Page Citations

## Architecture

PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
ChromaDB
↓
Semantic Search
↓
Llama 3
↓
Answer + Citations

## Technologies

- Python
- Streamlit
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3

## Example Question

Question:

What are the major business risks?

Answer:

The document identifies cybersecurity,
supply chain disruption and regulatory
compliance as key risks.

Sources:

Page 12
Page 15

## Screenshots

### Upload Document

screenshots/home.png

### Question Answering

screenshots/answer.png

## Installation

```bash
git clone <repo-url>
cd document-qa-rag
pip install -r requirements.txt
streamlit run app.py

## Roadmap

### Completed

- [x] PDF Upload
- [x] Vector Search
- [x] Ollama Integration
- [x] Page Citations

### Planned

- [ ] Multi-PDF Search
- [ ] Chat History
- [ ] Conversation Memory
- [ ] Evaluation Dashboard
