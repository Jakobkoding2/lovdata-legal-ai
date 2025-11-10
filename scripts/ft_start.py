#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from lovdata_rag.ft import FT_DATASET_PATH, start_fine_tune


def parse_args():
    parser = argparse.ArgumentParser(description="Kick off fine-tuning job")
    parser.add_argument("--model", default="gpt-5-mini", help="Base model to fine-tune")
    parser.add_argument("--dataset", default=str(FT_DATASET_PATH), help="Training JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found at {dataset_path}. Run scripts/ft_prepare.py first.")
    job_id = start_fine_tune(dataset_path, args.model)
    print(job_id)


if __name__ == "__main__":
    main()
