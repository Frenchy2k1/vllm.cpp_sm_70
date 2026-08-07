#!/usr/bin/env python3
"""Fail closed when the accepted binary-release spike contract drifts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BEGIN = "<!-- release-binary-contract:begin -->"
END = "<!-- release-binary-contract:end -->"
SPEC_PATH = ".agents/specs/release-binary-matrix.md"

IDENTITY = "ENG-RELEASE-BINARIES"
PRIMARY_CUDA_SMS = (
    "80",
    "86",
    "87",
    "89",
    "90a",
    "100a",
    "103a",
    "110",
    "120a",
    "121a",
)
WORK_DEPS = {
    "W1": (),
    "W2": ("W1",),
    "W3": (),
    "W4": (),
    "W5": (),
    "W6": (),
    "W7": ("W1", "W2", "W3", "W4", "W5", "W6"),
    "W8": ("W5", "W7"),
    "W9": ("W3", "W4", "W5", "W6", "W7"),
    "W10": ("W1", "W2", "W5", "W6", "W7"),
    "W11": ("W5", "W6", "W7"),
    "W12": ("W1", "W2", "W5", "W6", "W7"),
    "W13": ("W5", "W7", "W8", "W9", "W10", "W11"),
}

ANCHORS = {
    ".agents/engine-matrix.md": "| `ENG-RELEASE-BINARIES` |",
    ".agents/roadmap_v1.md": "| REL | `ROAD-V1-RELEASE` |",
    ".agents/NOW.md": "| Release | SPIKE | #129 |",
    ".agents/coordination.md": (
        "| `CLAIM-ENG-RELEASE-BINARIES-SPIKE` | "
        "`ENG-RELEASE-BINARIES` |"
    ),
    ".agents/state.md": (
        "## 2026-08-07 — Release matrix revised: fat CUDA and adaptive CPU "
        "are the primary downloads"
    ),
    "docs/STATUS.md": (
        "[Release spike](../.agents/specs/release-binary-matrix.md)"
    ),
    "docs/BENCHMARKS.md": (
        "| **Binary release matrix (spiked)** | `ENG-RELEASE-BINARIES`:"
    ),
}

EXPECTED_FIELDS = {
    "identity": IDENTITY,
    "lifecycle": "SPIKE",
    "primary_cuda_artifact": "one-fat-binary-per-os-host-abi",
    "primary_cuda_sms": ",".join(PRIMARY_CUDA_SMS),
    "per_sm_cuda": "optional-non-primary",
    "primary_cpu_artifact": "one-adaptive-binary-per-os-host-abi",
    "x86_64_baseline": "portable-sse2-without-avx2",
    "work_W12_policy": "optional-non-blocking",
    "archive_claims": "pending",
    "runtime_claims": "pending",
    "required_anchor_paths": ",".join(ANCHORS),
    **{f"work_{work}": ",".join(deps) for work, deps in WORK_DEPS.items()},
}

WORK_ROW = re.compile(r"^\|\s*(W(?:[1-9]|1[0-3]))\s*\|\s*([^|]*)\|", re.M)


def parse_contract(text: str) -> tuple[dict[str, str], list[str]]:
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        return {}, [
            f"{SPEC_PATH} must contain exactly one machine-readable release "
            f"contract block ({BEGIN} ... {END})"
        ]
    start = text.find(BEGIN) + len(BEGIN)
    end = text.find(END, start)
    if end < start:
        return {}, [f"{SPEC_PATH} has a malformed release contract block"]

    fields: dict[str, str] = {}
    errors: list[str] = []
    for line in text[start:end].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            errors.append(f"release contract line is not key=value: {stripped!r}")
            continue
        key, value = stripped.split("=", 1)
        if key in fields:
            errors.append(f"release contract repeats field {key!r}")
        fields[key] = value
    return fields, errors


def _field_error(key: str, actual: str | None, expected: str) -> str:
    names = {
        "identity": "release spec identity",
        "primary_cuda_artifact": "primary CUDA artifact",
        "primary_cuda_sms": "primary CUDA SM set",
        "per_sm_cuda": "per-SM CUDA policy",
        "primary_cpu_artifact": "primary CPU artifact",
        "x86_64_baseline": "x86_64 baseline",
        "work_W12_policy": "W12 policy",
    }
    for work in WORK_DEPS:
        names[f"work_{work}"] = f"{work} dependencies"
    label = names.get(key, f"release contract field {key}")
    return f"{label} is {actual!r}; expected {expected!r}"


def _normalize_deps(cell: str) -> tuple[str, ...]:
    value = cell.replace("`", "").strip()
    if value in {"", "—", "-", "[]"}:
        return ()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return tuple(part.strip() for part in value.split(",") if part.strip())


def contract_errors(root: Path) -> list[str]:
    spec = root / SPEC_PATH
    if not spec.is_file():
        return [f"{SPEC_PATH} is missing"]
    text = spec.read_text(encoding="utf-8")
    fields, errors = parse_contract(text)

    missing = set(EXPECTED_FIELDS) - set(fields)
    extra = set(fields) - set(EXPECTED_FIELDS)
    if missing:
        errors.append(f"release contract is missing fields: {sorted(missing)}")
    if extra:
        errors.append(f"release contract has unknown fields: {sorted(extra)}")
    for key, expected in EXPECTED_FIELDS.items():
        if fields.get(key) != expected:
            errors.append(_field_error(key, fields.get(key), expected))

    if "Status: accepted spike for `ENG-RELEASE-BINARIES`." not in text:
        errors.append(
            "release spec identity/status line must name accepted spike "
            "ENG-RELEASE-BINARIES"
        )

    header = "| Work | Deps | Deliverable | Exit gate |"
    if header not in text:
        errors.append("release work table is missing its explicit Deps column")
    rows = {work: _normalize_deps(cell) for work, cell in WORK_ROW.findall(text)}
    if set(rows) != set(WORK_DEPS):
        errors.append(
            f"release work table rows are {sorted(rows)}; expected W1-W13 exactly"
        )
    for work, expected in WORK_DEPS.items():
        if rows.get(work) != expected:
            errors.append(
                f"{work} dependencies in work table are {rows.get(work)!r}; "
                f"expected {expected!r}"
            )

    for relative, anchor in ANCHORS.items():
        path = root / relative
        if not path.is_file() or anchor not in path.read_text(encoding="utf-8"):
            errors.append(f"{relative} is missing required release anchor {anchor!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (used by the mutation suite)",
    )
    args = parser.parse_args(argv)
    errors = contract_errors(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Release binary contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
