#!/usr/bin/env python3
"""
Generate embeddings and FAISS index for Lovdata chunks.
"""

from __future__ import annotations

import argparse

from lovdata_rag.embeddings import build_embeddings, build_faiss_index


def parse_args():
    parser = argparse.ArgumentParser(description="Build embeddings and FAISS index")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_embeddings(force=args.force)
    build_faiss_index(force=args.force)


if __name__ == "__main__":
    main()
