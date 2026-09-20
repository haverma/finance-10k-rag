# Finance 10-K RAG Demo

A small, local project that compares answers from `llama2:7b` with and without retrieval-augmented generation (RAG).

## Goal

Answer questions about SEC 10-K filings, then demonstrate the difference between:

- **Baseline:** Llama 2 receives only the question.
- **RAG:** the application retrieves relevant 10-K excerpts from a local vector database and supplies them to Llama 2 with citations.

This project is for learning and research, not investment advice. Answers are limited to the selected filings and their filing dates.

## Intended architecture

```text
SEC 10-K HTML filings
        |
  extraction + chunking
        |
 embeddings (local Ollama embedding model)
        |
 Chroma vector database
        |
  question --> optional retrieval --> llama2:7b --> answer + citations
```

Each stored chunk has its text, an embedding vector, and metadata such as company, ticker, fiscal year, filing date, 10-K section, source URL, and chunk ID.

## Project stages

1. Scaffold and document the design. *(complete)*
2. Add the Python application, dependencies, and a RAG/no-RAG command-line switch.
3. Add SEC filing ingestion, cleaning, chunking, embeddings, and vector indexing.
4. Index a small starter set of filings and run a comparison evaluation.

## Commands

```bash
python app.py "What risks did Apple disclose about supply chains?" --no-rag
python app.py "What risks did Apple disclose about supply chains?" --rag --company "Apple Inc." --year 2024
```

Both modes use a temperature of `0.0` by default. The `--rag` flag changes application flow: it embeds the question, searches Chroma, adds retrieved excerpts to the model's messages, and prints their sources. `--no-rag` sends the question directly to the same Llama 2 model.

## Initial technology choices

- Generator: `llama2:7b` through Ollama (released 2023).
- Embeddings: an Ollama embedding model, selected during implementation.
- Vector database: Chroma in persistent local mode.
- Source documents: SEC EDGAR 10-K HTML filings, retaining official source links.

## Guardrails

- RAG answers must be supported only by retrieved filing excerpts.
- RAG answers cite their filing, year, and section.
- The baseline uses no retrieved context, so it is a fair comparison.
- The application says when the selected filings do not contain sufficient evidence.
