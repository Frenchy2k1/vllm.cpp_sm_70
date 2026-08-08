#!/usr/bin/env python3
"""Fail if a production env var is neither documented nor classified.

Every `VT_*` / `VLLM_*` environment name read from `src/` and `include/` must be
either documented in `docs/ENVIRONMENT.md` (the user-facing/behavior-changing
knobs) or listed on `scripts/env-doc-allowlist.txt` (the kernel-internal tuning
tail). This stops a new production env var from landing silently, the way the
153-variable blind spot did before docs/ENVIRONMENT.md existed.

The validation logic is a pure function `undocumented_env_vars(...) -> list[str]`
so it is unit-testable and mutation-testable (see
tests/scripts/test_check_env_doc.py), mirroring check-readme-structure.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "include")
SCAN_SUFFIXES = (".cpp", ".cc", ".cu", ".cuh", ".c", ".h", ".hpp", ".hh", ".hip")
ENV_DOC = ROOT / "docs/ENVIRONMENT.md"
ALLOWLIST = ROOT / "scripts/env-doc-allowlist.txt"

# A production env var is read as a quoted string literal, e.g. getenv("VT_FOO").
# Match the quoted name so a bare mention in a comment is not counted as a read.
_QUOTED = re.compile(r'"((?:VT_|VLLM_)[A-Z0-9_]+)"')
# A name token, used to harvest the documented set from the markdown / allowlist.
_TOKEN = re.compile(r'\b((?:VT_|VLLM_)[A-Z0-9_]+)\b')


def scan_env_names(root: Path) -> set[str]:
    """Return every VT_/VLLM_ env name read as a string literal in src/+include/."""
    names: set[str] = set()
    for rel in SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in SCAN_SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            names.update(_QUOTED.findall(text))
    return names


def documented_names(text: str) -> set[str]:
    """Names mentioned in docs/ENVIRONMENT.md."""
    return set(_TOKEN.findall(text))


def allowlisted_names(text: str) -> set[str]:
    """Names on the kernel-internal allowlist (one per line, # comments ignored)."""
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def undocumented_env_vars(
    scanned: set[str], documented: set[str], allowlisted: set[str]
) -> list[str]:
    """The scanned names covered by neither surface. Empty == the check passes."""
    return sorted(scanned - documented - allowlisted)


def main() -> int:
    if not ENV_DOC.exists():
        print("ERROR: docs/ENVIRONMENT.md is missing", file=sys.stderr)
        return 1
    scanned = scan_env_names(ROOT)
    documented = documented_names(ENV_DOC.read_text(encoding="utf-8"))
    allowlisted = (
        allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        if ALLOWLIST.exists()
        else set()
    )
    missing = undocumented_env_vars(scanned, documented, allowlisted)
    if missing:
        print(
            "ERROR: production env var(s) read from src/+include/ are neither "
            "documented in docs/ENVIRONMENT.md nor on "
            "scripts/env-doc-allowlist.txt:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "Document it in docs/ENVIRONMENT.md (if it is a user-facing / "
            "behavior-changing knob) or add it to scripts/env-doc-allowlist.txt "
            "(if it is a kernel-internal tuning switch).",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: all {len(scanned)} production env vars are documented "
        "or classified kernel-internal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
