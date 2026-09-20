#!/usr/bin/env python3
"""Run the Finance 10-K RAG demo without installing it as a package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_demo import main


if __name__ == "__main__":
    main()
