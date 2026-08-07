#!/usr/bin/env python3
"""Enforce that README.md stays a human-readable, user-facing document.

Per AGENTS.md, README.md is the LocalAI house-style user-facing document, not a
status-tracking log. This checker fails if the README loses one of the required
user-facing sections (Features / Build / usage-CLI / OpenAI server / Consuming),
grows a table cell into a "wall of prose", or contains an em-dash (house style).

The validation logic is a pure function `readme_errors(text) -> list[str]` so it
is unit-testable and mutation-testable (see
tests/scripts/test_check_readme_structure.py), mirroring check-doc-checkpoint.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
STATUS = ROOT / "docs/STATUS.md"

# Each required user-facing section is (label, matchers): the README must have an
# H2 heading whose lowercased text contains ANY of the matcher substrings.
REQUIRED_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Features", ("features",)),
    ("Supported models", ("supported models", "models")),
    ("Performance", ("performance",)),
    ("Build", ("build",)),
    ("Usage / CLI", ("cli", "running inference", "usage")),
    ("OpenAI server", ("server",)),
    ("Consuming (library / C API)", ("library", "consuming", "c api", "c-api")),
)

# A table cell longer than this is the "wall of prose" smell: forensic detail
# belongs in docs/STATUS.md and docs/BENCHMARKS.md, not in a README table cell.
MAX_CELL_CHARS = 220

# The README is a landing page, not the status ledger. These budgets are what
# stop it drifting back into a log one checkpoint at a time: per AGENTS.md the
# per-capability lifecycle obligation lands in docs/STATUS.md, and anything that
# would push the README past these limits is exactly that kind of content.
# Measured in characters, not lines, so the budget does not move with how the
# prose happens to be wrapped. The landing page was 61,909 chars when it was
# still the status ledger and is ~23,000 as a landing page; 30,000 leaves real
# headroom while making a slide back to a log fail CI.
MAX_README_CHARS = 30000
MAX_PARAGRAPH_CHARS = 900

# The README must point at the status ledger, and the ledger must actually carry
# the capability table (otherwise "move it to STATUS.md" silently loses it).
STATUS_LINK = "docs/STATUS.md"
CONTRIBUTOR_LINK = "CONTRIBUTING.md"
STATUS_REQUIRED_HEADINGS = ("capability status",)


def _h2_headers(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= set("-: ") for cell in cells)


def _prose_paragraphs(text: str) -> list[tuple[int, str]]:
    """Yield (start_line, paragraph) for prose only.

    Fenced code blocks, tables, headings, and list items are excluded: the rule
    targets the wall-of-prose narrative paragraph, not legitimate long tables or
    code samples.
    """
    paragraphs: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    in_fence = False

    def flush() -> None:
        if current:
            paragraphs.append((start, " ".join(current)))
            current.clear()

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        is_prose = bool(stripped) and not (
            stripped.startswith("|")
            or stripped.startswith("#")
            or stripped.startswith("-")
            or stripped.startswith("*")
            or stripped.startswith(">")
        )
        if is_prose:
            if not current:
                start = lineno
            current.append(stripped)
        else:
            flush()
    flush()
    return paragraphs


def status_errors(text: str) -> list[str]:
    """Return problems with docs/STATUS.md, the per-capability status ledger."""
    errors: list[str] = []
    headers_lower = [h.lower() for h in _h2_headers(text)]
    for needle in STATUS_REQUIRED_HEADINGS:
        if not any(needle in h for h in headers_lower):
            errors.append(
                f"docs/STATUS.md is missing the '{needle}' section (it is the "
                "surface AGENTS.md points the per-capability obligation at)"
            )
    return errors


def readme_errors(text: str) -> list[str]:
    """Return a list of human-readable problems with the README text."""
    errors: list[str] = []

    headers_lower = [h.lower() for h in _h2_headers(text)]
    for label, matchers in REQUIRED_SECTIONS:
        if not any(any(m in h for m in matchers) for h in headers_lower):
            errors.append(f"missing required user-facing section: {label}")

    if "—" in text:  # em-dash
        count = text.count("—")
        errors.append(
            f"README contains {count} em-dash(es); house style forbids them "
            "(use commas, periods, parentheses, or hyphens)"
        )

    if len(text) > MAX_README_CHARS:
        errors.append(
            f"README is {len(text)} chars, over the {MAX_README_CHARS}-char "
            "landing-page budget; per-capability status belongs in "
            "docs/STATUS.md, not here"
        )

    if STATUS_LINK not in text:
        errors.append(
            f"README does not link to {STATUS_LINK}; the landing page must "
            "point at the per-capability status ledger"
        )

    if CONTRIBUTOR_LINK not in text:
        errors.append(
            f"README does not link to {CONTRIBUTOR_LINK}; contributors need a "
            "public entry point to the agent protocol"
        )

    for lineno, para in _prose_paragraphs(text):
        if len(para) > MAX_PARAGRAPH_CHARS:
            errors.append(
                f"line {lineno}: prose paragraph of {len(para)} chars exceeds "
                f"{MAX_PARAGRAPH_CHARS} (wall-of-prose smell; move the detail "
                "to docs/STATUS.md and link to it)"
            )

    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if _is_separator_row(cells):
                continue
            for cell in cells:
                if len(cell) > MAX_CELL_CHARS:
                    errors.append(
                        f"line {lineno}: table cell of {len(cell)} chars exceeds "
                        f"{MAX_CELL_CHARS} (wall-of-prose smell; move forensic "
                        "detail to docs/STATUS.md / docs/BENCHMARKS.md)"
                    )
    return errors


def main() -> int:
    if not README.exists():
        print("ERROR: README.md is missing", file=sys.stderr)
        return 1
    if not STATUS.exists():
        print("ERROR: docs/STATUS.md is missing (it is the per-capability "
              "status ledger AGENTS.md requires)", file=sys.stderr)
        return 1
    errors = readme_errors(README.read_text(encoding="utf-8"))
    errors += status_errors(STATUS.read_text(encoding="utf-8"))
    if errors:
        print("ERROR: the user-facing docs are not valid:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("OK: README.md is a valid landing page and docs/STATUS.md carries "
          "the capability ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
