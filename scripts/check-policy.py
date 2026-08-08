#!/usr/bin/env python3
"""Validate the authoritative repository policy and waiver registries."""

from __future__ import annotations

import argparse
from pathlib import Path

from policy_contract import validate_policy


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="validate registries and named paths during the ordered bootstrap",
    )
    args = parser.parse_args()
    errors = validate_policy(ROOT, schema_only=args.schema_only)
    if errors:
        print("policy contract FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    mode = "schema" if args.schema_only else "full"
    print(f"OK: policy contract ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
