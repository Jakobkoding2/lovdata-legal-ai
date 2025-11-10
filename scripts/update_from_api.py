#!/usr/bin/env python3
"""
Check Lovdata for updates and rebuild assets if necessary.
"""

from __future__ import annotations

import argparse

from lovdata_rag.update import update_from_api


def parse_args():
    parser = argparse.ArgumentParser(description="Update Lovdata assets if data changed")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if no change detected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_from_api(force=args.force)


if __name__ == "__main__":
    main()
