"""CLI for comparing local Llama 2 answers with and without 10-K retrieval."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama

PROJECT_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = PROJECT_DIR / "data" / "chroma"
COLLECTION_NAME = "sec_10k_chunks"
DEFAULT_CHAT_MODEL = "llama2:7b"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"


@dataclass(frozen=True)
class RetrievedPassage:
    """A filing excerpt selected by semantic retrieval."""

    text: str
    metadata: dict[str, Any]
    distance: float | None


def get_collection() -> chromadb.Collection:
    """Open the persistent local collection; create it if indexing has not run."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Chunked SEC 10-K filing passages"},
    )


def retrieve(question: str, top_k: int, company: str | None, year: int | None) -> list[RetrievedPassage]:
    """Embed a question locally and return the most relevant filing passages."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "The 10-K index is empty. Run the ingestion command after it is added in the next project step."
        )

    filters: list[dict[str, Any]] = []
    if company:
        filters.append({"company": {"$eq": company}})
    if year:
        filters.append({"fiscal_year": {"$eq": year}})
    where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}

    embedding = ollama.embed(model=DEFAULT_EMBEDDING_MODEL, input=question)["embeddings"][0]
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedPassage(text=document, metadata=metadata, distance=distance)
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def format_context(passages: list[RetrievedPassage]) -> str:
    """Keep excerpts compact enough for Llama 2's limited context window."""
    blocks = []
    for number, passage in enumerate(passages, start=1):
        meta = passage.metadata
        citation = (
            f"[{number}] {meta.get('company', 'Unknown')} {meta.get('fiscal_year', 'Unknown')} 10-K"
            f", {meta.get('section', 'Unknown section')}"
        )
        blocks.append(f"{citation}\n{passage.text[:1_500]}")
    return "\n\n".join(blocks)


def build_messages(question: str, use_rag: bool, passages: list[RetrievedPassage]) -> list[dict[str, str]]:
    """Only RAG receives source excerpts and the evidence-only instruction."""
    if not use_rag:
        return [
            {
                "role": "system",
                "content": "Answer concisely. If you are uncertain, clearly say so. Do not give investment advice.",
            },
            {"role": "user", "content": question},
        ]

    return [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied SEC 10-K excerpts. Do not use outside knowledge or infer "
                "unsupported facts. If the excerpts do not answer the question, say exactly: "
                "'Insufficient evidence in the retrieved filings.' Cite every factual claim with its excerpt "
                "number, such as [1]. Do not give investment advice."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nSEC 10-K excerpts:\n{format_context(passages)}",
        },
    ]


def ask(args: argparse.Namespace) -> None:
    passages = retrieve(args.question, args.top_k, args.company, args.year) if args.rag else []
    response = ollama.chat(
        model=args.model,
        messages=build_messages(args.question, args.rag, passages),
        options={"temperature": args.temperature},
    )
    print(f"Mode: {'RAG' if args.rag else 'baseline (no RAG)'}")
    print(f"Model: {args.model}")
    print("\nAnswer:\n" + response["message"]["content"].strip())
    if passages:
        print("\nRetrieved sources:")
        for index, passage in enumerate(passages, start=1):
            meta = passage.metadata
            print(
                f"[{index}] {meta.get('company')} | FY{meta.get('fiscal_year')} | "
                f"{meta.get('section')} | {meta.get('source_url')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Llama 2 with and without SEC 10-K RAG.")
    parser.add_argument("question", help="The financial filing question to ask.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rag", action="store_true", help="Retrieve 10-K excerpts before answering.")
    mode.add_argument("--no-rag", dest="rag", action="store_false", help="Ask the model without filing retrieval.")
    parser.add_argument("--company", help="Optional exact company metadata filter, e.g. Apple Inc.")
    parser.add_argument("--year", type=int, help="Optional fiscal-year metadata filter, e.g. 2024")
    parser.add_argument("--top-k", type=int, default=4, help="Number of excerpts to retrieve (default: 4).")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"Ollama chat model (default: {DEFAULT_CHAT_MODEL}).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        ask(args)
    except (RuntimeError, ConnectionError, ollama.ResponseError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
