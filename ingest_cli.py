#!/usr/bin/env python3
"""CLI entry point for ingesting 10-K PDFs and inspecting the vector DB."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
from ingest import ingest_pdf, ingest_directory, inspect_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest SEC 10-K PDFs into ChromaDB or inspect the vector store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- ingest sub-command ----
    ingest_p = sub.add_parser("ingest", help="Ingest one or more 10-K PDFs.")
    source = ingest_p.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path, help="Path to a single PDF file.")
    source.add_argument("--dir", type=Path, help="Directory of PDF files to ingest.")
    ingest_p.add_argument("--company", required=True, help="Company name, e.g. 'Apple Inc.'")
    ingest_p.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL")
    ingest_p.add_argument("--year", type=int, required=True, help="Fiscal year, e.g. 2024")
    ingest_p.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in chars (default: 1000)")
    ingest_p.add_argument("--overlap", type=int, default=200, help="Overlap in chars (default: 200)")

    # ---- inspect sub-command ----
    inspect_p = sub.add_parser("inspect", help="Show vector DB contents.")
    inspect_p.add_argument("--limit", type=int, default=5, help="Number of sample records (default: 5)")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "ingest":
        if args.pdf:
            ingest_pdf(args.pdf, args.company, args.ticker, args.year, args.chunk_size, args.overlap)
        else:
            ingest_directory(args.dir, args.company, args.ticker, args.year, args.chunk_size, args.overlap)
    elif args.command == "inspect":
        inspect_db(limit=args.limit)


if __name__ == "__main__":
    main()
