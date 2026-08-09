#!/usr/bin/env python3
"""Keep .agents/NOW.md a short, current, one-Read resume surface.

NOW.md is the single small file a cold session reads first to become productive.
It is a SNAPSHOT, never a log: it is rewritten in place, and the history it used
to summarise now lives where history belongs, in git.

This checker owns exactly one obligation -- structure and budget, so NOW.md
cannot decay into another status log. The other half, "NOW.md must be refreshed
when the live position moves", is owned by check-doc-checkpoint.py, which
already requires NOW.md on a lifecycle change. Splitting one obligation across
two checkers is how it ends up enforced twice and satisfiable by neither.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NOW = ROOT / ".agents/NOW.md"
NOW_PATH = ".agents/NOW.md"

# Budgets. NOW.md exists to be read in full, every session, by every agent. The
# moment it stops fitting in one screenful of attention it has become the thing
# it was meant to replace.
MAX_LINES = 100
MAX_CHARS = 6000
MAX_ENTRY_CHARS = 400

REQUIRED_HEADINGS = (
    "live claims",
    "current gate",
    "next actions",
)

STAMP = re.compile(r"^<!--\s*now-updated:\s*(\d{4}-\d{2}-\d{2})\s*-->$", re.MULTILINE)


def structure_errors(text: str) -> list[str]:
    """Return budget/shape problems with the NOW digest."""
    errors: list[str] = []

    if not STAMP.search(text):
        errors.append(
            "missing the freshness stamp <!-- now-updated: YYYY-MM-DD -->; it "
            "records when this snapshot was last known true"
        )

    lowered = text.lower()
    for heading in REQUIRED_HEADINGS:
        if f"## {heading}" not in lowered:
            errors.append(
                f"missing the '## {heading}' section; a cold session needs all "
                f"of {', '.join(REQUIRED_HEADINGS)} to resume without reading "
                "the full record"
            )

    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        errors.append(
            f"is {len(lines)} lines, over the {MAX_LINES}-line budget; move "
            "detail to the row's spec and keep only the live position here"
        )
    if len(text) > MAX_CHARS:
        errors.append(
            f"is {len(text)} characters, over the {MAX_CHARS}-character budget; "
            "this is a digest, not a status log"
        )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "|")) and len(stripped) > MAX_ENTRY_CHARS:
            errors.append(
                f"an entry is {len(stripped)} characters, over the "
                f"{MAX_ENTRY_CHARS}-character budget: {stripped[:60]!r}...; "
                "link the row's spec instead of inlining the narrative"
            )

    return errors


def main(argv: list[str]) -> int:
    # --base/--head/--commit/--staged are accepted and ignored: CI passes a
    # range, and this check is range-independent now that freshness coupling
    # belongs to check-doc-checkpoint.py. Silently accepting them keeps the CI
    # invocation stable.
    del argv

    if not NOW.exists():
        print(f"ERROR: {NOW_PATH} does not exist", file=sys.stderr)
        return 1

    failures = [
        f"{NOW_PATH} {error}"
        for error in structure_errors(NOW.read_text(encoding="utf-8"))
    ]

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "NOW.md is the one-Read resume surface: the live claims, the gate "
            "being chased, and the next actions, rewritten in place. Detail "
            "belongs in the row's spec and the area matrices; history belongs "
            "in git.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {NOW_PATH} is a current, in-budget resume digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
