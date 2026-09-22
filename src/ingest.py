"""Internal SEC 10-K ingestion functions used by the single and batch CLIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import chromadb
import pymupdf
import ollama

from rag_demo import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, VECTOR_DB_DIR

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


@dataclass(frozen=True)
class TextChunk:
    """A contiguous slice of cleaned filing text."""

    text: str
    index: int
    start_char: int
    end_char: int
    section: str


@dataclass(frozen=True)
class SecFiling:
    """The official SEC location and metadata for one Form 10-K."""

    company: str
    ticker: str
    fiscal_year: int
    filing_date: str
    source_url: str


def read_local_filing(path: str | Path) -> str:
    """Extract text from a local SEC filing PDF or HTML document."""
    filing_path = Path(path).expanduser().resolve()
    if not filing_path.is_file():
        raise FileNotFoundError(f"Filing not found: {filing_path}")

    suffix = filing_path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(filing_path) as document:
            text = "\n".join(page.get_text("text") for page in document)
    elif suffix in {".htm", ".html", ".xhtml"}:
        soup = BeautifulSoup(filing_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        text = soup.get_text("\n")
    else:
        raise ValueError(f"Expected a PDF or HTML filing, got: {filing_path.suffix}")

    if not text.strip():
        raise ValueError(f"No extractable text found in {filing_path.name}")
    return text


def clean_text(raw: str) -> str:
    """Remove common filing noise while retaining paragraph boundaries."""
    text = re.sub(r"^[\s]*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*FORM\s+10-?K\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_ITEM_HEADING_RE = re.compile(
    r"(?im)^\s*(Item\s+(?:1[0-6]|1A|1B|1C|7A|9A|[1-9])[.]?)\s*(.*)$"
)


def _filing_body_start(text: str) -> int:
    """Skip a table of contents when a filing repeats the Item 1 heading in its body."""
    matches = list(re.finditer(r"(?im)^\s*Item\s+1[.]?\s*(?:\n\s*)?Business\b", text))
    return matches[1].start() if len(matches) >= 2 else 0


def _sections(text: str) -> list[tuple[str, int, str]]:
    """Return actual SEC Item sections, avoiding inline references such as 'Item 8 of this Form 10-K'."""
    body_start = _filing_body_start(text)
    body = text[body_start:]
    headings = list(_ITEM_HEADING_RE.finditer(body))
    if not headings:
        return [("Unclassified", body_start, body)]

    sections: list[tuple[str, int, str]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        item = re.sub(r"\s+", " ", heading.group(1)).strip()
        title = re.sub(r"\s+", " ", heading.group(2)).strip()
        label = f"{item} {title}".strip()
        sections.append((label, body_start + heading.start(), body[heading.start():end].strip()))
    return sections


def _chunk_section(section: str, start_offset: int, text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Chunk one section without allowing unrelated SEC Items into the same retrieval record."""
    units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if unit.strip()]
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_length = 0
    index = 0

    def flush() -> None:
        nonlocal buffer, buffer_length, index
        if not buffer:
            return
        chunk = " ".join(buffer).strip()
        if chunk:
            offset = text.find(chunk)
            chunks.append(TextChunk(chunk, index, start_offset + max(offset, 0), start_offset + max(offset, 0) + len(chunk), section))
            index += 1
        overlap_text = chunk[-overlap:].strip()
        buffer = [overlap_text] if overlap_text else []
        buffer_length = len(overlap_text)

    for unit in units:
        if buffer and buffer_length + len(unit) + 1 > chunk_size:
            flush()
        while len(unit) > chunk_size:
            partial = unit[:chunk_size]
            chunks.append(TextChunk(partial, index, start_offset + text.find(partial), start_offset + text.find(partial) + len(partial), section))
            index += 1
            unit = unit[chunk_size - overlap:]
        buffer.append(unit)
        buffer_length += len(unit) + 1
    flush()
    return chunks


def chunk_text(text: str, chunk_size: int = 1_200, overlap: int = 150) -> list[TextChunk]:
    """Create overlapping, section-aware chunks suitable for retrieval."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be between 0 and chunk_size.")
    chunks: list[TextChunk] = []
    for section, start_offset, section_text in _sections(text):
        chunks.extend(_chunk_section(section, start_offset, section_text, chunk_size, overlap))
    return [
        TextChunk(chunk.text, index, chunk.start_char, chunk.end_char, chunk.section)
        for index, chunk in enumerate(chunks)
    ]


def _sec_get(url: str, user_agent: str) -> bytes:
    if "@" not in user_agent:
        raise ValueError("SEC requests require --user-agent with a contact email.")
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json,text/html"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _normalise_company(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_company(company: str, user_agent: str) -> tuple[int, str, str]:
    """Resolve a company name such as APPLE to its SEC CIK and ticker."""
    payload = json.loads(_sec_get(SEC_TICKERS_URL, user_agent))
    query = _normalise_company(company)
    matches = []
    for item in payload.values():
        title = item["title"]
        normalised_title = _normalise_company(title)
        if normalised_title == query or normalised_title.startswith(query) or query in normalised_title:
            matches.append(item)
    if not matches:
        raise ValueError(f"No SEC company match found for {company!r}.")
    exact = [item for item in matches if _normalise_company(item["title"]) == query]
    selected = (exact or matches)[0]
    return int(selected["cik_str"]), selected["title"], selected["ticker"]


def find_sec_10k(company: str, fiscal_year: int, user_agent: str, filing_month: int | None = None) -> SecFiling:
    """Find the official HTML 10-K whose SEC report date matches *fiscal_year*."""
    cik, legal_name, ticker = resolve_company(company, user_agent)
    submissions = json.loads(_sec_get(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent))
    recent = submissions["filings"]["recent"]
    candidates = []
    for index, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        report_date = recent["reportDate"][index]
        filing_date = recent["filingDate"][index]
        if not report_date.startswith(str(fiscal_year)):
            continue
        if filing_month and int(filing_date[5:7]) != filing_month:
            continue
        candidates.append(index)
    if not candidates:
        month_label = f" filed in month {filing_month}" if filing_month else ""
        raise ValueError(f"No 10-K found for {legal_name} with fiscal year {fiscal_year}{month_label}.")

    index = max(candidates, key=lambda candidate: recent["filingDate"][candidate])
    accession = recent["accessionNumber"][index].replace("-", "")
    document = recent["primaryDocument"][index]
    if not document:
        raise ValueError(f"SEC did not provide a primary HTML document for {legal_name} FY{fiscal_year}.")
    return SecFiling(
        company=legal_name,
        ticker=ticker,
        fiscal_year=fiscal_year,
        filing_date=recent["filingDate"][index],
        source_url=SEC_ARCHIVES_URL.format(cik=cik, accession=accession, document=document),
    )


def download_sec_filing(filing: SecFiling, user_agent: str) -> Path:
    """Download an official SEC HTML document once and return its local cache path."""
    destination = RAW_DATA_DIR / f"{filing.ticker}-{filing.fiscal_year}-10-k.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_sec_get(filing.source_url, user_agent))
    return destination


def infer_fiscal_year(text: str) -> int:
    """Infer a local filing's fiscal year; return 0 when the filing does not state one clearly."""
    match = re.search(r"fiscal year ended.{0,80}?(19|20)\d{2}", text, flags=re.IGNORECASE | re.DOTALL)
    return int(match.group(0)[-4:]) if match else 0


def index_filing(
    company: str,
    ticker: str,
    fiscal_year: int,
    filing_date: str,
    source_url: str,
    local_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    """Extract, chunk, embed, and store one filing in persistent Chroma."""
    text = clean_text(read_local_filing(local_path))
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No chunks were created from {local_path.name}")

    source_id = hashlib.sha256(source_url.encode()).hexdigest()
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={
            "description": "Chunked SEC 10-K filing passages",
            "hnsw:space": "cosine",
        },
    )
    collection.delete(where={"source_id": source_id})

    for start in range(0, len(chunks), 32):
        batch = chunks[start : start + 32]
        documents = [chunk.text for chunk in batch]
        vectors = ollama.embed(model=embedding_model, input=documents)["embeddings"]
        collection.upsert(
            ids=[hashlib.sha256(f"{source_id}:{chunk.index}".encode()).hexdigest() for chunk in batch],
            documents=documents,
            embeddings=vectors,
            metadatas=[
                {
                    "company": company,
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "filing_date": filing_date,
                    "form": "10-K",
                    "section": chunk.section,
                    "source_url": source_url,
                    "source_id": source_id,
                    "chunk_index": chunk.index,
                }
                for chunk in batch
            ],
        )
    return len(chunks)


def ingest_sec_company(company: str, fiscal_year: int, user_agent: str, filing_month: int | None = None) -> tuple[SecFiling, int]:
    """Resolve, download, and index a company/year request from SEC EDGAR."""
    filing = find_sec_10k(company, fiscal_year, user_agent, filing_month)
    chunk_count = index_filing(
        company=filing.company,
        ticker=filing.ticker,
        fiscal_year=filing.fiscal_year,
        filing_date=filing.filing_date,
        source_url=filing.source_url,
        local_path=download_sec_filing(filing, user_agent),
    )
    return filing, chunk_count


def ingest_local_filing(company: str, path: str | Path) -> int:
    """Index a supplied PDF or HTML filing, inferring its fiscal year when possible."""
    resolved_path = Path(path).expanduser().resolve()
    raw_text = read_local_filing(resolved_path)
    return index_filing(
        company=company,
        ticker="UNKNOWN",
        fiscal_year=infer_fiscal_year(raw_text),
        filing_date="unknown",
        source_url=resolved_path.as_uri(),
        local_path=resolved_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one SEC 10-K by company/year or a local PDF/HTML path.")
    parser.add_argument("company")
    parser.add_argument("source", help="Fiscal year, e.g. 2023, or a local PDF/HTML path.")
    parser.add_argument("--user-agent", help="Name/project and email; required for SEC downloads.")
    parser.add_argument("--filing-month", type=int, choices=range(1, 13), help="Optional SEC filing month filter.")
    args = parser.parse_args()
    if args.source.isdigit() and len(args.source) == 4:
        if not args.user_agent:
            raise SystemExit("Error: --user-agent is required for SEC downloads.")
        filing, count = ingest_sec_company(args.company, int(args.source), args.user_agent, args.filing_month)
        print(f"Indexed {count} chunks from {filing.company} FY{filing.fiscal_year}: {filing.source_url}")
    else:
        count = ingest_local_filing(args.company, args.source)
        print(f"Indexed {count} chunks from local filing: {args.source}")


if __name__ == "__main__":
    main()
