"""Ingest SEC 10-K PDF filings into the ChromaDB vector store.

Pipeline: PDF → extract_text → clean_text → chunk_text → embed → store.
Only the first three stages are implemented so far.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str | Path) -> str:
    """Read every page of a PDF and return the concatenated plain text.

    Pages are separated by a single newline so downstream cleaning can
    normalise whitespace in one pass.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path.name}")
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Common 10-K header/footer noise
_PAGE_NUMBER_RE = re.compile(
    r"^[\s]*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FORM_HEADER_RE = re.compile(
    r"^\s*(?:FORM|Form)\s+10-?K\s*$", re.MULTILINE
)
_REPEATED_DASHES_RE = re.compile(r"-{3,}")
_EXCESS_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Strip filing noise and normalise whitespace.

    Removes stand-alone page numbers, repeated "FORM 10-K" headers,
    decorative dash lines, and collapses excessive whitespace while
    preserving meaningful paragraph breaks.
    """
    text = _PAGE_NUMBER_RE.sub("", raw)
    text = _FORM_HEADER_RE.sub("", text)
    text = _REPEATED_DASHES_RE.sub("", text)
    text = _EXCESS_WHITESPACE_RE.sub(" ", text)
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextChunk:
    """A contiguous slice of the cleaned document text."""

    text: str
    index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[TextChunk]:
    """Split *text* into overlapping windows of roughly *chunk_size* characters.

    The overlap keeps surrounding context at chunk boundaries so that
    sentences split across a boundary can still be retrieved.

    Parameters
    ----------
    text:
        Cleaned document text to chunk.
    chunk_size:
        Target number of characters per chunk.  Actual chunks may be
        slightly shorter (the last chunk) but never longer.
    overlap:
        Number of characters shared between consecutive chunks.
        Must be less than *chunk_size*.

    Returns
    -------
    list[TextChunk]
        Ordered list of chunks with positional metadata.

    Raises
    ------
    ValueError
        If *chunk_size* or *overlap* are invalid.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                TextChunk(
                    text=chunk_str,
                    index=index,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
        start += step

    return chunks
