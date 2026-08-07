#!/usr/bin/env python3
"""Fail closed when the accepted binary-release spike contract drifts."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


BEGIN = "<!-- release-binary-contract:begin -->"
END = "<!-- release-binary-contract:end -->"
SPEC_PATH = ".agents/specs/release-binary-matrix.md"
TEST_PATH = "tests/scripts/test_check_release_binary_contract.py"

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

BENCHMARKS_RELEASE_ROW = (
    "| **Binary release matrix (spiked)** | `ENG-RELEASE-BINARIES`: primary "
    "host-ABI fat-CUDA + adaptive-CPU static-core bundles; optional per-SM "
    "diagnostics; experimental literal-static musl CPU | **PENDING:** pins 10-SM "
    "fat CUDA, adaptive no-AVX2 CPU, W1-W13/W10-W12 policy, public pending states, "
    "and 15 tests. No archive, staged smoke, runtime, correctness, or performance "
    "evidence "
    "| n/a |"
)

STATUS_RELEASE_FRAGMENTS = (
    "Supported subset; bundles SPIKED, no artifacts",
    "primary fat CUDA/adaptive CPU, W1-W13/W10-W12 policy, pending claims, and "
    "15-test inventory mutation-gated; per-SM diagnostics optional; no "
    "archive/runtime claim",
)

HUMAN_CONTRACT = {
    "human primary CPU/CUDA contract": (
        "The primary CPU download is one conservative-baseline, runtime-adaptive "
        "binary per OS+host ABI; the primary CUDA download is one fat binary per "
        "OS+host ABI containing every supported SM."
    ),
    "human x86_64 no-AVX2 contract": (
        "For x86_64, the baseline must run without AVX2: portable/SSE2 code "
        "remains callable, and higher instructions live only in per-function or "
        "per-TU tiers."
    ),
}

WORK_CONTENT = {
    "W10": (
        "primary Linux CUDA fat bundles for x86_64 and aarch64 host ABIs",
        "each extracted archive contains all ten SMs and six exact AOT trees; "
        "per-SM evidence remains independent; no host ABI is inferred from the "
        "other",
    ),
    "W12": (
        "optional single-SM CUDA diagnostic/performance variants",
        "generated from the same explicit matrix and evidence; never advertised "
        "as the primary KISS download or used to bypass W10",
    ),
}

REQUIRED_TEST_METHODS = (
    "test_repository_contract_passes",
    "test_spec_identity_is_fail_closed",
    "test_each_primary_cuda_sm_is_required",
    "test_primary_cuda_must_stay_one_fat_binary_per_host_abi",
    "test_per_sm_cuda_must_not_become_primary",
    "test_primary_cpu_must_stay_one_adaptive_binary_per_host_abi",
    "test_x86_64_baseline_must_not_require_avx2",
    "test_work_table_has_explicit_deps_column",
    "test_each_work_dependency_edge_is_pinned",
    "test_optional_w12_does_not_block_w13",
    "test_each_required_record_anchor_is_fail_closed",
    "test_public_release_rows_remain_pending",
    "test_human_w12_is_optional_and_cannot_replace_w10",
    "test_human_primary_artifact_contract_matches_machine_block",
    "test_required_mutation_test_inventory_is_pinned",
)

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

WORK_ROW = re.compile(
    r"^\|\s*(W(?:[1-9]|1[0-3]))\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|",
    re.M,
)


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


def _normalize_prose(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _test_inventory_errors(root: Path) -> list[str]:
    path = root / TEST_PATH
    if not path.is_file():
        return [f"required mutation-test inventory file {TEST_PATH} is missing"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return [f"required mutation-test inventory is not valid Python: {error}"]

    methods = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    expected = set(REQUIRED_TEST_METHODS)
    actual = set(methods)
    errors: list[str] = []
    if len(methods) != len(REQUIRED_TEST_METHODS) or actual != expected:
        errors.append(
            "required mutation-test inventory drifted: "
            f"expected {len(REQUIRED_TEST_METHODS)} named methods "
            f"{sorted(expected)}, found {len(methods)} {sorted(methods)}"
        )

    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_TEST_METHODS"
            for target in node.targets
        )
    ]
    try:
        declared = tuple(ast.literal_eval(declarations[0].value))
    except (IndexError, TypeError, ValueError, SyntaxError):
        declared = ()
    if len(declarations) != 1 or declared != REQUIRED_TEST_METHODS:
        errors.append(
            "required mutation-test inventory declaration does not match the "
            "checker's exact count and names"
        )
    return errors


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
    parsed_rows = WORK_ROW.findall(text)
    rows = {work: _normalize_deps(deps) for work, deps, _, _ in parsed_rows}
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

    work_content = {
        work: (_normalize_prose(deliverable), _normalize_prose(exit_gate))
        for work, _, deliverable, exit_gate in parsed_rows
    }
    for work, (deliverable, exit_gate) in WORK_CONTENT.items():
        actual = work_content.get(work)
        if actual is None or actual[0] != deliverable:
            errors.append(
                f"{work} deliverable is {None if actual is None else actual[0]!r}; "
                f"expected {deliverable!r}"
            )
        if actual is None or actual[1] != exit_gate:
            errors.append(
                f"{work} exit gate is {None if actual is None else actual[1]!r}; "
                f"expected {exit_gate!r}"
            )

    normalized_spec = _normalize_prose(text)
    for label, statement in HUMAN_CONTRACT.items():
        if statement not in normalized_spec:
            errors.append(
                f"{label} must match the accepted machine-readable release contract"
            )

    for relative, anchor in ANCHORS.items():
        path = root / relative
        if not path.is_file() or anchor not in path.read_text(encoding="utf-8"):
            errors.append(f"{relative} is missing required release anchor {anchor!r}")
    benchmarks = root / "docs/BENCHMARKS.md"
    if not benchmarks.is_file() or BENCHMARKS_RELEASE_ROW not in benchmarks.read_text(
        encoding="utf-8"
    ):
        errors.append(
            "docs/BENCHMARKS.md release row must stay PENDING with no archive, "
            "runtime, correctness, or performance evidence"
        )
    status = root / "docs/STATUS.md"
    status_text = status.read_text(encoding="utf-8") if status.is_file() else ""
    status_row = next(
        (line for line in status_text.splitlines() if line.startswith("| OpenAI server |")),
        "",
    )
    if not all(fragment in status_row for fragment in STATUS_RELEASE_FRAGMENTS):
        errors.append(
            "docs/STATUS.md release row must stay SPIKED with no artifacts and no "
            "runtime claim"
        )
    errors.extend(_test_inventory_errors(root))
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
