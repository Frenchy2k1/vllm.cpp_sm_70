#!/usr/bin/env python3
"""Resolve container tags from release/container-matrix.json.

The publish workflow needs three different views of the same matrix, and none
of them belongs inline in YAML where it cannot be unit-tested:

  --immutable     one `<package>:<version>-<lane>` per lane
  --moving        `<source> <target>` pairs, immutable tag -> moving tag
  --lanes         lane ids, one per line
  --build-matrix  GitHub Actions `include:` JSON of lane x architecture

Usage: scripts/container_tags.py --version 0.1.0 [--immutable|--moving|--lanes]
       scripts/container_tags.py --build-matrix [--release]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "release/container-matrix.json"


def immutable_tags(matrix: dict, version: str) -> list[str]:
    package = matrix["package"]
    return [
        f"{package}:{lane['version_tag'].format(version=version)}"
        for lane in matrix["lanes"]
    ]


def moving_pairs(matrix: dict, version: str) -> list[tuple[str, str]]:
    package = matrix["package"]
    pairs: list[tuple[str, str]] = []
    for lane in matrix["lanes"]:
        source = f"{package}:{lane['version_tag'].format(version=version)}"
        for moving in lane["moving_tags"]:
            pairs.append((source, f"{package}:{moving}"))
    return pairs


def main_pairs(matrix: dict) -> list[tuple[str, str]]:
    """`<immutable source> <main tag>` pairs for a publish from main.

    Main images are a convenience, not a release: they move, they carry no
    support claim, and they must never touch `latest*` or a version tag.
    """
    package = matrix["package"]
    return [
        (f"{package}:{lane['id']}", f"{package}:{lane['main_tag']}")
        for lane in matrix["lanes"]
    ]


def main_tags(matrix: dict) -> list[str]:
    package = matrix["package"]
    return [f"{package}:{lane['main_tag']}" for lane in matrix["lanes"]]


def lane_ids(matrix: dict) -> list[str]:
    return [lane["id"] for lane in matrix["lanes"]]


def build_matrix(matrix: dict, release: bool) -> list[dict]:
    """lane x architecture entries for a GitHub Actions `include:` list.

    A release run builds every lane on every architecture. A pull request builds
    a REDUCED set, and the reason is cost, not confidence: a ten-SM fat CUDA
    image does not fit a hosted runner's budget on every push, and building one
    per PR would make the gate something people route around. The full set still
    runs before anything is published -- the publish job rebuilds and revalidates
    each lane immediately before pushing it -- so the worst case a reduced PR
    matrix can produce is a failed release run, never a bad published image.

    Lanes opt in to PR verification with `verify_on_pull_request`.
    """
    entries: list[dict] = []
    for lane in matrix["lanes"]:
        if not release and not lane.get("verify_on_pull_request"):
            continue
        for arch in lane["architectures"]:
            if not release and not arch.get("verify_on_pull_request"):
                continue
            entries.append(
                {
                    "lane": lane["id"],
                    "platform": arch["platform"],
                    "runner": arch["runner"],
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--version")
    parser.add_argument("--release", action="store_true")
    view = parser.add_mutually_exclusive_group(required=True)
    view.add_argument("--immutable", action="store_true")
    view.add_argument("--moving", action="store_true")
    view.add_argument("--lanes", action="store_true")
    view.add_argument("--build-matrix", action="store_true")
    view.add_argument("--main-tags", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))

    if args.lanes:
        print("\n".join(lane_ids(matrix)))
        return 0

    if args.main_tags:
        print("\n".join(main_tags(matrix)))
        return 0

    if args.build_matrix:
        entries = build_matrix(matrix, args.release)
        if not entries:
            print("build matrix is empty", file=sys.stderr)
            return 1
        print(json.dumps(entries, separators=(",", ":"), sort_keys=True))
        return 0

    if not args.version:
        parser.error("--version is required for --immutable and --moving")

    if args.immutable:
        print("\n".join(immutable_tags(matrix, args.version)))
    else:
        print("\n".join(f"{source} {target}" for source, target in moving_pairs(matrix, args.version)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
