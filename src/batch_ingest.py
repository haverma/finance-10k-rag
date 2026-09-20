"""Read a company/year or company/local-filing input file and ingest every line."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

from ingest import ingest_local_filing, ingest_sec_company


@dataclass(frozen=True)
class IngestRequest:
    company: str
    source: str
    line_number: int

    @property
    def is_sec_year(self) -> bool:
        return self.source.isdigit() and len(self.source) == 4


def read_requests(path: str | Path) -> list[IngestRequest]:
    """Parse lines like `APPLE 2023` or `ORACLE /path/to/oracle_10k.pdf`."""
    requests: list[IngestRequest] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if len(parts) < 2:
            raise ValueError(f"Line {line_number}: expected COMPANY followed by YEAR or LOCAL_PATH.")
        requests.append(IngestRequest(company=" ".join(parts[:-1]), source=parts[-1], line_number=line_number))
    if not requests:
        raise ValueError("The input file contains no ingestion requests.")
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest SEC or local 10-K filings.")
    parser.add_argument("input_file", help="Text file containing COMPANY YEAR or COMPANY LOCAL_PATH per line.")
    parser.add_argument("--user-agent", help="Name/project and email; required if any line uses a year.")
    parser.add_argument("--filing-month", type=int, choices=range(1, 13), help="Optional SEC filing month filter.")
    args = parser.parse_args()
    requests = read_requests(args.input_file)
    if any(request.is_sec_year for request in requests) and not args.user_agent:
        raise SystemExit("Error: --user-agent is required when downloading from SEC EDGAR.")

    failures = 0
    for request in requests:
        try:
            if request.is_sec_year:
                filing, count = ingest_sec_company(request.company, int(request.source), args.user_agent, args.filing_month)
                print(f"OK line {request.line_number}: {filing.company} FY{filing.fiscal_year} -> {count} chunks")
            else:
                count = ingest_local_filing(request.company, request.source)
                print(f"OK line {request.line_number}: {request.company} local filing -> {count} chunks")
        except Exception as error:  # Continue so a single bad filing does not stop the batch.
            failures += 1
            print(f"ERROR line {request.line_number} ({request.company}): {error}")
    if failures:
        raise SystemExit(f"Finished with {failures} failed request(s).")


if __name__ == "__main__":
    main()
