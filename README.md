# Finance 10-K RAG Demo

A small, local project that compares answers from `llama2:7b` with and without retrieval-augmented generation (RAG).

## Goal

Answer questions about SEC 10-K filings, then demonstrate the difference between:

- **Baseline:** Llama 2 receives only the question.
- **RAG:** the application retrieves relevant 10-K excerpts from a local vector database and supplies them to Llama 2 with citations.

This project is for learning and research, not investment advice. Answers are limited to the selected filings and their filing dates.

## Intended architecture

```text
SEC 10-K HTML/PDF filings
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

## Batch ingestion

Create a plain-text input file. Each non-comment line consists of a company name followed by either a fiscal year or a local PDF/HTML filing path:

```text
APPLE 2023
ORACLE /Users/harsh/Documents/oracle_10k.pdf
"MICROSOFT CORPORATION" 2023
```

Run the caller script:

```bash
.venv/bin/python src/batch_ingest.py companies.txt \
  --user-agent "Your Name finance-10k-rag your.email@example.com"
```

For a `COMPANY YEAR` line, it resolves the company through SEC EDGAR, finds the Form 10-K whose **report date** is in that fiscal year, downloads its official primary HTML filing, and indexes it. A supplied local PDF or HTML path is extracted directly and indexed; its fiscal year is inferred from its text when possible.

To restrict SEC results to filings submitted in a particular month, add (for example):

```bash
--filing-month 3
```

Use this only when the chosen company actually filed that 10-K in March. `APPLE 2023`, for example, is a fiscal-year request and is not inherently a March filing.

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
