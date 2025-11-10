#!/usr/bin/env python3
"""
Build the Lovdata corpus parquet from public tarballs.
"""

from __future__ import annotations

import argparse

from lovdata_rag.data_pipeline import build_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lovdata corpus parquet")
    parser.add_argument("--force", action="store_true", help="Rebuild even if files already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_corpus(force=args.force)


if __name__ == "__main__":
    main()
