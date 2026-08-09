#!/usr/bin/env python3
"""Validate the structured state record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from state_record import validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base")
    args = parser.parse_args()

    errors = validate(args.root, base=args.base)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: structured state record is relationally consistent and immutable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
