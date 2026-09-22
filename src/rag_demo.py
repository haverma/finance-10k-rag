"""CLI for comparing local Llama 2 answers with and without 10-K retrieval."""

from __future__ import annotations

import argparse
import re
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
    vector_score: float
    lexical_score: float
    hybrid_score: float


def get_collection() -> chromadb.Collection:
    """Open the persistent local collection; create it if indexing has not run."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Chunked SEC 10-K filing passages",
            "hnsw:space": "cosine",
        },
    )


_STOP_WORDS = {"a", "an", "and", "about", "did", "disclose", "for", "in", "is", "of", "the", "to", "was", "what", "were"}


def _lexical_score(question: str, text: str) -> float:
    """Score exact financial terms and phrases without replacing semantic retrieval."""
    terms = [term for term in re.findall(r"[a-z0-9]+", question.lower()) if term not in _STOP_WORDS]
    if not terms:
        return 0.0
    normalised_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    matched_terms = sum(bool(re.search(rf"\b{re.escape(term)}\b", normalised_text)) for term in set(terms))
    term_score = matched_terms / len(set(terms))
    phrases = [" ".join(terms[index : index + size]) for size in (2, 3, 4) for index in range(len(terms) - size + 1)]
    matched_phrase_lengths = [len(phrase.split()) for phrase in phrases if phrase in normalised_text]
    phrase_score = max(matched_phrase_lengths, default=0) / min(4, len(terms))
    # A phrase such as "diluted earnings per share" is stronger evidence than four
    # isolated words scattered across a filing or a table heading.
    score = 0.35 * term_score + 0.65 * phrase_score
    metric_phrases = [phrase for phrase in phrases if len(phrase.split()) >= 3]
    if any(re.search(rf"{re.escape(phrase)}\s*\$?\s*\d", text, flags=re.IGNORECASE) for phrase in metric_phrases):
        score += 0.20
    return min(1.0, score)


def retrieve(question: str, top_k: int, company: str | None, year: int | None) -> list[RetrievedPassage]:
    """Hybrid retrieve: cosine similarity plus exact-term matching within filtered chunks."""
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
    candidate_count = min(max(top_k * 8, 32), collection.count())
    semantic = collection.query(
        query_embeddings=[embedding],
        n_results=candidate_count,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    lexical = collection.get(where=where, include=["documents", "metadatas"])

    candidates: dict[str, dict[str, Any]] = {}
    distances = semantic["distances"][0]
    for chunk_id, document, metadata, distance in zip(
        semantic["ids"][0], semantic["documents"][0], semantic["metadatas"][0], distances
    ):
        candidates[chunk_id] = {"text": document, "metadata": metadata, "distance": distance}
    for chunk_id, document, metadata in zip(lexical["ids"], lexical["documents"], lexical["metadatas"]):
        candidates.setdefault(chunk_id, {"text": document, "metadata": metadata, "distance": None})

    scored: list[RetrievedPassage] = []
    for candidate in candidates.values():
        # Chroma cosine distance is 1 - cosine similarity. A missing vector score means the
        # chunk entered the candidate set through lexical matching alone.
        vector_score = max(0.0, 1.0 - candidate["distance"]) if candidate["distance"] is not None else 0.0
        lexical_score = _lexical_score(question, candidate["text"])
        # Financial tables frequently have weak dense embeddings, so exact term/phrase
        # evidence receives the larger weight while cosine similarity remains a tie-breaker.
        hybrid_score = 0.15 * vector_score + 0.85 * lexical_score
        scored.append(
            RetrievedPassage(
                text=candidate["text"],
                metadata=candidate["metadata"],
                distance=candidate["distance"],
                vector_score=vector_score,
                lexical_score=lexical_score,
                hybrid_score=hybrid_score,
            )
        )
    return sorted(scored, key=lambda passage: passage.hybrid_score, reverse=True)[:top_k]


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
            source = (
                f"[{index}] {meta.get('company')} | FY{meta.get('fiscal_year')} | "
                f"{meta.get('section')} | {meta.get('source_url')}"
            )
            if args.show_scores:
                source += (
                    f" | cosine={passage.vector_score:.3f} "
                    f"lexical={passage.lexical_score:.3f} hybrid={passage.hybrid_score:.3f}"
                )
            print(source)


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
    parser.add_argument("--show-scores", action="store_true", help="Print cosine, lexical, and combined retrieval scores.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        ask(args)
    except (RuntimeError, ConnectionError, ollama.ResponseError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
