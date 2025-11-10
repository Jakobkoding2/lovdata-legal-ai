#!/usr/bin/env python3
"""
Chunk Lovdata corpus into retrieval-ready windows.
"""

from __future__ import annotations

import argparse

from lovdata_rag.chunking import build_chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk Lovdata corpus")
    parser.add_argument("--force", action="store_true", help="Rebuild chunks even if cached")
    parser.add_argument("--chunk-size", type=int, default=350, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=60, help="Token overlap between chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_chunks(chunk_size=args.chunk_size, overlap=args.overlap, force=args.force)


if __name__ == "__main__":
    main()
