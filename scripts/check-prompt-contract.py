#!/usr/bin/env python3
"""Validate the closed grammar of versioned runtime prompt contracts."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



PROMPT_BUDGET = 4096
ROLES = ("implementer", "reviewer", "operator")
RUNTIME_PROMPT_FILES = frozenset(f"{role}.md" for role in ROLES)
SECTION_NAMES = ("Task envelope", "Method", "Required output", "Stop conditions")
METADATA_KEYS = ("prompt-contract-version", "role")
ENVELOPE_ROWS = (
    "- Goal: REQUIRED",
    "- Context: REQUIRED",
    "- Constraints: REQUIRED",
    "- Done when: REQUIRED",
    "- Required evidence: REQUIRED",
    "- Authority: REQUIRED",
    "- Missing input: NEEDS_CONTEXT",
)
METHOD_ROWS = {
    "implementer": (
        "- `IMP-TEST-FIRST` | required | Write the failing test first, run it before implementation, and confirm the stated failure.",
        "- `IMP-VERIFY` | required | Run every declared verification and require exit zero or an unchanged proven baseline.",
        "- `IMP-MUTATE` | required | Delete or invert each behavior named by every added test, require its focused suite to fail, then restore it.",
        "- `IMP-SCOPE` | forbidden | Change files or behavior outside Authority.",
        "- `IMP-EVIDENCE` | evidence | Record each command, exit status, negative mutation, observed failure, and restoration check.",
    ),
    "reviewer": (
        "- `REV-STATIC` | required | Perform independent static review of requirements, plan, diff, and relevant surrounding code.",
        "- `REV-MUTATION` | required | In a scratch copy, delete or invert each behavior changed tests claim to pin and rerun its focused test.",
        "- `REV-FULL-GATE` | required | Run the full declared gate exactly once on the unchanged reviewed commit.",
        "- `REV-NO-REPAIR` | forbidden | Repair a finding.",
        "- `REV-WORKTREE` | forbidden | Mutate the reviewed worktree, index, HEAD, or branch.",
        "- `REV-FINDINGS` | evidence | List findings first in severity-descending order with severity, path:line, evidence, violated rule or requirement, and required remediation.",
    ),
    "operator": (
        "- `OP-DELEGATE` | required | Delegate implementation and repairs to fresh implementers.",
        "- `OP-CONTINUE` | required | For every actionable in-scope reviewer FAIL dispatch a fresh implementer, run focused and full gates, and dispatch a fresh scoped reviewer; repeat until PASS because attempt or retry budgets are scheduling controls and never terminal blockers.",
        "- `OP-VERIFY` | required | Run claimed verification on the returned commit without trusting the implementer report.",
        "- `OP-REVIEW` | required | Dispatch a fresh reviewer for independent static review and targeted scratch mutation.",
        "- `OP-DISPOSITION` | required | Merge a verified PR in-session or close an obsolete PR with its recorded reason.",
        "- `OP-REPAIR` | forbidden | Repair an implementer or reviewer finding in the coordinating context.",
        "- `OP-EVIDENCE` | evidence | Record verification, review, PR disposition, blocker, and remote-state evidence.",
    ),
}
OUTPUT_ROWS = {
    "implementer": (
        "- status: COMPLETE | BLOCKED | NEEDS_CONTEXT | NEEDS_DECISION",
        "- summary: REQUIRED",
        "- changed_files: LIST",
        "- commands_and_exit_status: EVIDENCE",
        "- negative_mutation: EVIDENCE",
        "- deviations: EVIDENCE | NONE",
        "- risks: EVIDENCE | NONE",
        "- omitted_gates: EVIDENCE | NONE",
        "- commit_sha: SHA | NONE",
    ),
    "reviewer": (
        "- findings: NONE | LIST[severity,path:line,evidence,violated_rule_or_requirement,required_remediation]; ORDER=severity_descending",
        "- verdict: PASS | FAIL",
        "- static_review: EVIDENCE",
        "- mutation_evidence: EVIDENCE",
        "- full_gate: EVIDENCE",
        "- remaining_concern: EVIDENCE | NONE",
    ),
    "operator": (
        "- status: MERGED | CLOSED | BLOCKED | REMOTE_UNVERIFIED",
        "- verification: EVIDENCE",
        "- review: EVIDENCE",
        "- disposition: EVIDENCE",
        "- remaining_concern: EVIDENCE | NONE",
    ),
}
STOP_ROWS = {
    "implementer": (
        "- `STOP-AUTHORITY` | BLOCKED | An edit outside Authority is required.",
        "- `STOP-VERIFY` | BLOCKED | Declared verification cannot pass or match a proven baseline.",
        "- `STOP-CONTEXT` | NEEDS_CONTEXT | Missing input changes the contract or public behavior.",
        "- `STOP-DECISION` | NEEDS_DECISION | A required choice exceeds Authority.",
    ),
    "reviewer": (
        "- `STOP-SCRATCH` | FAIL | A safe scratch mutation environment cannot be created.",
        "- `STOP-EVIDENCE` | FAIL | Required source, diff, test, or gate evidence is unavailable.",
    ),
    "operator": (
        "- `STOP-AUTHORITY` | BLOCKED | A required external action exceeds Authority and the precise missing authority is named.",
        "- `STOP-RESOURCE` | BLOCKED | A required external resource is unavailable and the precise resource is named.",
        "- `STOP-DEVELOPER` | BLOCKED | The developer explicitly directs the review loop to stop before PASS.",
        "- `STOP-REMOTE` | REMOTE_UNVERIFIED | Required remote state cannot be queried.",
    ),
}


def _parse_metadata(lines: list[str], label: str) -> tuple[dict[str, str], int, list[str]]:
    errors: list[str] = []
    if not lines or lines[0] != "---":
        return {}, 0, [f"{label}: missing exact metadata opening delimiter"]
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, len(lines), [f"{label}: missing exact metadata closing delimiter"]

    pairs: list[tuple[str, str]] = []
    for line in lines[1:end]:
        if ": " not in line:
            errors.append(f"{label}: unparsed metadata line {line!r}")
            continue
        key, value = line.split(": ", 1)
        pairs.append((key, value))
    counts = Counter(key for key, _ in pairs)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"{label}: duplicate metadata keys: {', '.join(duplicates)}")
    keys = tuple(key for key, _ in pairs)
    if keys != METADATA_KEYS:
        errors.append(f"{label}: metadata keys must be exactly {METADATA_KEYS!r}")
    return dict(pairs), end + 1, errors


def _parse_sections(
    lines: list[str], start: int, label: str
) -> tuple[list[tuple[str, tuple[str, ...]]], list[str]]:
    sections: list[tuple[str, list[str]]] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines[start:], start=start + 1):
        if not line:
            continue
        if line.startswith("## "):
            sections.append((line[3:], []))
            continue
        if not sections:
            errors.append(f"{label}:{line_number}: unparsed normative line {line!r}")
            continue
        sections[-1][1].append(line)
    return [(name, tuple(rows)) for name, rows in sections], errors


def _row_errors(
    label: str, section: str, actual: tuple[str, ...], expected: tuple[str, ...]
) -> list[str]:
    errors: list[str] = []
    counts = Counter(actual)
    duplicates = sorted(row for row, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"{label}: {section} has duplicate rows: {duplicates!r}")
    for row in expected:
        if row not in actual:
            errors.append(f"{label}: {section} is missing row {row!r}")
    for row in actual:
        if row not in expected:
            errors.append(f"{label}: {section} has unparsed row {row!r}")
    if not errors and actual != expected:
        errors.append(f"{label}: {section} rows are out of order")
    return errors


def validate_prompt(path: Path, role: str) -> list[str]:
    """Return every closed-grammar defect in one runtime prompt."""

    label = path.as_posix()
    if role not in ROLES:
        return [f"{label}: unknown runtime role {role!r}"]
    if not path.is_file():
        return [f"{label}: missing runtime prompt"]
    if path.stat().st_size > PROMPT_BUDGET:
        return [f"{label}: exceeds the {PROMPT_BUDGET}-byte prompt budget"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"{label}: not valid UTF-8: {exc}"]

    lines = text.splitlines()
    metadata, body_start, errors = _parse_metadata(lines, label)
    expected_metadata = {
        "prompt-contract-version": "1",
        "role": role,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            errors.append(f"{label}: metadata {key!r} must be exactly {expected!r}")
    sections, section_errors = _parse_sections(lines, body_start, label)
    errors.extend(section_errors)
    names = tuple(name for name, _ in sections)
    if names != SECTION_NAMES:
        errors.append(f"{label}: H2 sections must be exactly {SECTION_NAMES!r}")
    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"{label}: duplicate H2 sections: {', '.join(duplicates)}")

    actual_by_name = {name: rows for name, rows in sections if counts[name] == 1}
    expected_by_name = {
        "Task envelope": ENVELOPE_ROWS,
        "Method": METHOD_ROWS[role],
        "Required output": OUTPUT_ROWS[role],
        "Stop conditions": STOP_ROWS[role],
    }
    for name, expected_rows in expected_by_name.items():
        actual_rows = actual_by_name.get(name)
        if actual_rows is not None:
            errors.extend(_row_errors(label, name, actual_rows, expected_rows))
    return errors


def repository_errors(root: Path) -> list[str]:
    """Validate every runtime prompt owned by this contract checker."""

    errors: list[str] = []
    prompt_root = root / ".agents/prompts"
    if prompt_root.is_dir():
        for artifact in sorted(prompt_root.iterdir(), key=lambda path: path.name):
            if artifact.name not in RUNTIME_PROMPT_FILES:
                errors.append(
                    f"{artifact.as_posix()}: unexpected runtime prompt artifact; "
                    f"expected exactly {', '.join(sorted(RUNTIME_PROMPT_FILES))}"
                )
    else:
        errors.append(f"{prompt_root.as_posix()}: missing runtime prompt namespace")
    for role in ROLES:
        errors.extend(
            validate_prompt(
                prompt_root / f"{role}.md", role
            )
        )
    return errors


def main() -> int:
    errors = repository_errors(ROOT)
    if errors:
        print("prompt contract FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"OK: {len(ROLES)} closed runtime prompt contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
