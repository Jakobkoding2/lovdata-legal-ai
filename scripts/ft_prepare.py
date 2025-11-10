#!/usr/bin/env python3

from __future__ import annotations

import argparse

from lovdata_rag.ft import prepare_ft_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Norwegian law Q&A dataset for fine-tuning")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_ft_dataset(limit=args.limit)


if __name__ == "__main__":
    main()
