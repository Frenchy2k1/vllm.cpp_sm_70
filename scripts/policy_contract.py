#!/usr/bin/env python3
"""Shared parser for the repository policy and waiver registries."""

from __future__ import annotations

import csv
import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath


POLICY_HEADER = (
    "rule_id",
    "scope",
    "trigger",
    "requirement",
    "enforcement",
    "waiver_class",
    "procedure",
)
WAIVER_HEADER = (
    "waiver_id",
    "rule_id",
    "scope",
    "owner",
    "reason",
    "evidence",
    "expires",
)
WAIVER_CLASSES = frozenset({"never", "expiring", "migration-only"})
POLICY_MAX_BYTES = 16 * 1024
POLICY_MAX_RULES = 60
RULE_ID = re.compile(r"POL-[A-Z0-9]+(?:-[A-Z0-9]+)*\Z")
WAIVER_ID = re.compile(r"WAIVER-[A-Z0-9]+(?:-[A-Z0-9]+)*\Z")
SCOPE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
BROAD_SCOPE_VALUES = frozenset(
    {"all", "any", "everything", "global", "repo", "repository"}
)
AGENTS_MAX_BYTES = 12 * 1024
PROCEDURE_BUDGETS = {
    ".agents/workflow.md": 12 * 1024,
    ".agents/verification.md": 8 * 1024,
    ".agents/porting.md": 8 * 1024,
}
T0_RULE_IDS = (
    "POL-AUTH-REGISTRY",
    "POL-BOOT-ENTRYPOINT",
    "POL-ROLE-DECLARED",
    "POL-SPIKE-FIRST",
    "POL-MIRROR-VLLM",
    "POL-PORT-TESTS",
    "POL-CORRECTNESS-GATE",
    "POL-PREFLIGHT",
    "POL-REVIEW-FRESH",
    "POL-REVIEW-NO-REPAIR",
    "POL-OPERATOR-VERIFY",
    "POL-ONE-SURFACE",
    "POL-EVIDENCE-PRESERVE",
    "POL-PR-DISPOSITION",
)
BOOT_BLOCK = """<!-- policy-boot:begin -->
1. Run `scripts/agent-start.py`; pass explicit intent when known, follow its role action, and rerun it after declaration.
2. Resolve `.env` and `.agents/developer-preferences.md` from the shared checkout, requesting only values required by the current task.
3. Read `.agents/NOW.md` for the live snapshot.
4. Read `.agents/policy.csv`, then the procedure named by each applicable rule.
5. Read only the claimed task's spec, owning row, evidence, and coordination entry.
<!-- policy-boot:end -->"""
LEGACY_ACTIVE_PATHS = (
    ".agents/directives.md",
    ".agents/ai-coding-assistants.md",
    ".agents/specs/operator-helper-protocol.md",
    ".agents/gates.md",
    ".agents/benchmark-protocol.md",
    ".agents/discipline.md",
    ".agents/test-porting.md",
)
POLICY_ARCHIVES = (
    ".agents/completed/policy-directives-legacy.md",
    ".agents/completed/ai-coding-assistants-legacy.md",
    ".agents/completed/operator-helper-protocol-legacy.md",
    ".agents/completed/mvp-gates-legacy.md",
    ".agents/completed/benchmark-protocol-legacy.md",
    ".agents/completed/porting-discipline-legacy.md",
    ".agents/completed/test-porting-legacy.md",
)
ARCHIVE_NAME_PATTERN = re.compile(
    r"(?:directives|ai-coding-assistants|operator-helper|gates|benchmark-protocol|discipline|test-porting)",
    re.IGNORECASE,
)
POLICY_REF = re.compile(r"(?<![A-Z0-9-])POL-[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Z0-9-])")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    scope: str
    trigger: str
    requirement: str
    enforcement: str
    waiver_class: str
    procedure: str


@dataclass(frozen=True)
class Waiver:
    waiver_id: str
    rule_id: str
    scope: str
    owner: str
    reason: str
    evidence: str
    expires: dt.date


def _read_rows(path: Path, header: tuple[str, ...]) -> tuple[list[list[str]], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], [f"missing registry: {path.relative_to(path.parents[1])}"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [], [f"{path.name}: not valid UTF-8: {exc}"]

    physical = text.splitlines()
    if any(not line.strip() for line in physical[1:]):
        errors.append(f"{path.name}: blank physical records are forbidden")

    try:
        reader = csv.reader(text.splitlines(keepends=True), strict=True)
        parsed = list(reader)
    except csv.Error as exc:
        return [], errors + [f"{path.name}: invalid RFC 4180 CSV: {exc}"]

    if not parsed:
        return [], errors + [f"{path.name}: missing header"]
    if tuple(parsed[0]) != header:
        errors.append(
            f"{path.name}: header must be exactly {','.join(header)}"
        )

    rows: list[list[str]] = []
    for number, row in enumerate(parsed[1:], start=2):
        if len(row) != len(header):
            errors.append(
                f"{path.name}:{number}: expected {len(header)} columns; got {len(row)}"
            )
            continue
        if any("\n" in field or "\r" in field for field in row):
            errors.append(
                f"{path.name}:{number}: multiline fields are forbidden; one physical record is required"
            )
        rows.append(row)
    return rows, errors


def _repo_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
    )


def _is_exact_scope(scope: str) -> bool:
    """Return whether *scope* identifies one concrete waiver target."""

    try:
        kind, value = scope.split(":", 1)
    except ValueError:
        return False
    if not value or value.casefold() in BROAD_SCOPE_VALUES:
        return False
    if kind == "pr":
        return bool(re.fullmatch(r"[1-9][0-9]*", value))
    if kind == "commit":
        return bool(re.fullmatch(r"(?:[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})", value))
    if kind == "path":
        path = PurePosixPath(value)
        return (
            _repo_relative(value)
            and value != "."
            and not value.endswith("/")
            and path.as_posix() == value
        )
    if kind in {"task", "gate", "hardware"}:
        return bool(SCOPE_ID.fullmatch(value))
    return False


def _parse_policy(root: Path) -> tuple[dict[str, PolicyRule], list[str]]:
    path = root / ".agents/policy.csv"
    rows, errors = _read_rows(path, POLICY_HEADER)
    if path.is_file() and path.stat().st_size > POLICY_MAX_BYTES:
        errors.append(
            f"policy.csv exceeds the {POLICY_MAX_BYTES}-byte budget ({path.stat().st_size} bytes)"
        )
    if len(rows) > POLICY_MAX_RULES:
        errors.append(f"policy.csv has more than {POLICY_MAX_RULES} rules ({len(rows)})")

    rules: dict[str, PolicyRule] = {}
    for number, values in enumerate(rows, start=2):
        row = dict(zip(POLICY_HEADER, values, strict=True))
        rule_id = row["rule_id"]
        if not RULE_ID.fullmatch(rule_id):
            errors.append(f"policy.csv:{number}: malformed rule_id {rule_id!r}")
        if rule_id in rules:
            errors.append(f"policy.csv:{number}: duplicate rule_id {rule_id!r}")
        for field in ("scope", "trigger", "requirement", "enforcement", "procedure"):
            if not row[field].strip():
                errors.append(f"policy.csv:{number}: {field} is empty")
        if row["waiver_class"] not in WAIVER_CLASSES:
            errors.append(
                f"policy.csv:{number}: waiver_class {row['waiver_class']!r} is not one of "
                + ", ".join(sorted(WAIVER_CLASSES))
            )

        for checker_value in row["enforcement"].split(";"):
            checker = checker_value.strip()
            if (
                not _repo_relative(checker)
                or not checker.startswith("scripts/check-")
                or Path(checker).suffix not in {".py", ".sh"}
            ):
                errors.append(
                    f"policy.csv:{number}: enforcement {checker!r} is not a repo-relative checker entrypoint"
                )
            elif not (root / checker).is_file():
                errors.append(f"policy.csv:{number}: unknown enforcement path {checker!r}")

        procedure = row["procedure"]
        if not _repo_relative(procedure):
            errors.append(
                f"policy.csv:{number}: procedure {procedure!r} is not repo-relative"
            )
        elif Path(procedure).suffix != ".md":
            errors.append(f"policy.csv:{number}: unknown procedure path {procedure!r}")
        else:
            procedure_path = root / procedure
            if procedure_path.is_symlink():
                errors.append(f"policy.csv:{number}: procedure path {procedure!r} is a symlink")
            elif not procedure_path.is_file():
                errors.append(f"policy.csv:{number}: unknown procedure path {procedure!r}")
            elif procedure_path.stat().st_size == 0:
                errors.append(f"policy.csv:{number}: empty procedure path {procedure!r}")

        if rule_id not in rules:
            rules[rule_id] = PolicyRule(**row)

    if not rows:
        errors.append("policy.csv must contain at least one rule")
    return rules, errors


def load_policy(root: Path) -> dict[str, PolicyRule]:
    rules, errors = _parse_policy(root)
    if errors:
        raise ValueError("\n".join(errors))
    return rules


def load_waivers(
    root: Path,
    rules: dict[str, PolicyRule],
    today: dt.date | None = None,
) -> list[Waiver]:
    current = today or dt.date.today()
    rows, errors = _read_rows(root / ".agents/waivers.csv", WAIVER_HEADER)
    waivers: list[Waiver] = []
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    for number, values in enumerate(rows, start=2):
        row = dict(zip(WAIVER_HEADER, values, strict=True))
        waiver_id = row["waiver_id"]
        if not WAIVER_ID.fullmatch(waiver_id):
            errors.append(f"waivers.csv:{number}: malformed waiver_id {waiver_id!r}")
        if waiver_id in seen_ids:
            errors.append(f"waivers.csv:{number}: duplicate waiver_id {waiver_id!r}")
        seen_ids.add(waiver_id)

        rule = rules.get(row["rule_id"])
        if rule is None:
            errors.append(
                f"waivers.csv:{number}: unknown rule {row['rule_id']!r}"
            )
        elif rule.waiver_class == "never":
            errors.append(
                f"waivers.csv:{number}: rule {row['rule_id']!r} cannot be waived"
            )

        scope = row["scope"]
        if any(character in scope for character in "*?[]{}"):
            errors.append(f"waivers.csv:{number}: wildcard scope {scope!r} is forbidden")
        if not _is_exact_scope(scope):
            errors.append(
                f"waivers.csv:{number}: {scope!r} is not an exact scope "
                "(task/PR/commit/path/gate/hardware required)"
            )
        target = (row["rule_id"], scope)
        if target in seen_targets:
            errors.append(
                f"waivers.csv:{number}: duplicate rule/scope {row['rule_id']!r} {scope!r}"
            )
        seen_targets.add(target)

        for field in ("owner", "reason", "evidence"):
            if not row[field].strip():
                errors.append(f"waivers.csv:{number}: {field} is empty")

        expiry: dt.date | None = None
        try:
            expiry = dt.date.fromisoformat(row["expires"])
        except ValueError:
            errors.append(
                f"waivers.csv:{number}: expires {row['expires']!r} is not a valid ISO date"
            )
        if expiry is not None and expiry <= current:
            errors.append(
                f"waivers.csv:{number}: waiver {waiver_id!r} expired on {expiry.isoformat()}"
            )

        if waiver_id and not _waiver_is_referenced(root, waiver_id):
            errors.append(
                f"waivers.csv:{number}: unused waiver {waiver_id!r}; cite it in its evidence or consuming record"
            )

        if expiry is not None:
            waivers.append(Waiver(**{**row, "expires": expiry}))

    if errors:
        raise ValueError("\n".join(errors))
    return waivers


def _waiver_is_referenced(root: Path, waiver_id: str) -> bool:
    """A waiver is live debt only when a non-code record names it explicitly."""

    needle = re.compile(
        rf"(?<![A-Z0-9-]){re.escape(waiver_id)}(?![A-Z0-9-])"
    )
    ignored_roots = {".git", "tests", "build", ".cache"}
    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file() or relative == Path(".agents/waivers.csv"):
            continue
        if relative.parts and relative.parts[0] in ignored_roots:
            continue
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".json", ".yml", ".yaml"}:
            continue
        try:
            if needle.search(path.read_text(encoding="utf-8")):
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def validate_policy(root: Path, *, schema_only: bool = False) -> list[str]:
    """Return every policy/waiver contract defect found under *root*.

    ``schema_only`` validates the registries and named files early in bootstrap.
    Full mode additionally validates the compact bootstrap, procedure coverage,
    archives, and policy-surface links.
    """

    rules, errors = _parse_policy(root)
    if errors:
        return errors
    try:
        load_waivers(root, rules)
    except (ValueError, KeyError) as exc:
        errors.extend(str(exc).splitlines())
    if not schema_only:
        errors.extend(_validate_consolidation(root, rules))
    return errors


def _marker_block(text: str, begin: str, end: str) -> str | None:
    if text.count(begin) != 1 or text.count(end) != 1:
        return None
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    return text[start:finish]


def _generated_t0(rules: dict[str, PolicyRule]) -> str:
    lines = ["<!-- policy-t0:begin -->"]
    for rule_id in T0_RULE_IDS:
        rule = rules.get(rule_id)
        if rule is not None:
            lines.append(f"- `{rule_id}` — {rule.requirement}")
    lines.append("<!-- policy-t0:end -->")
    return "\n".join(lines)


def _controlled_paragraphs(text: str, relative: str) -> tuple[list[str], list[str]]:
    begin = "<!-- policy-procedure:begin -->"
    end = "<!-- policy-procedure:end -->"
    errors: list[str] = []
    if text.count(begin) != text.count(end):
        return [], [f"{relative}: unbalanced policy-procedure markers"]
    paragraphs: list[str] = []
    position = 0
    while True:
        start = text.find(begin, position)
        if start < 0:
            break
        finish = text.find(end, start + len(begin))
        if finish < 0:
            break
        body = text[start + len(begin):finish].strip()
        paragraphs.extend(part.strip() for part in re.split(r"\n\s*\n", body) if part.strip())
        position = finish + len(end)
    return paragraphs, errors


def _validate_consolidation(root: Path, rules: dict[str, PolicyRule]) -> list[str]:
    errors: list[str] = []
    agents = root / "AGENTS.md"
    if agents.is_symlink() or not agents.is_file():
        return ["AGENTS.md must be a non-symlink regular file"]
    agents_text = agents.read_text(encoding="utf-8")
    if agents.stat().st_size >= AGENTS_MAX_BYTES:
        errors.append(
            f"AGENTS.md exceeds the strict 12 KiB budget ({agents.stat().st_size} bytes)"
        )
    if _marker_block(agents_text, "<!-- policy-boot:begin -->", "<!-- policy-boot:end -->") != BOOT_BLOCK:
        errors.append("AGENTS.md boot block is missing, duplicated, or out of order")
    expected_t0 = _generated_t0(rules)
    if _marker_block(agents_text, "<!-- policy-t0:begin -->", "<!-- policy-t0:end -->") != expected_t0:
        errors.append("AGENTS.md generated T0 block does not match policy.csv")

    for relative in LEGACY_ACTIVE_PATHS:
        if os.path.lexists(root / relative):
            errors.append(f"retired active policy path still exists: {relative}")

    expected_archives = {Path(path).name for path in POLICY_ARCHIVES}
    for relative in POLICY_ARCHIVES:
        path = root / relative
        if path.is_symlink():
            errors.append(f"policy archive is a symlink: {relative}")
        elif not path.is_file():
            errors.append(f"missing policy archive: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty policy archive: {relative}")
    completed = root / ".agents/completed"
    if completed.is_dir():
        for path in completed.iterdir():
            if (
                path.name not in expected_archives
                and ARCHIVE_NAME_PATTERN.search(path.name)
            ):
                errors.append(f"ambiguous policy archive: .agents/completed/{path.name}")

    references: dict[str, list[str]] = {rule_id: [] for rule_id in rules}
    rule_paragraphs: dict[str, list[str]] = {rule_id: [] for rule_id in rules}
    procedure_paths = set(rule.procedure for rule in rules.values())
    allowed_paths = set(PROCEDURE_BUDGETS)
    unexpected = sorted(procedure_paths - allowed_paths)
    if unexpected:
        errors.append("registry uses non-canonical procedure path(s): " + ", ".join(unexpected))
    for relative in sorted(procedure_paths):
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            continue
        budget = PROCEDURE_BUDGETS.get(relative)
        if budget is not None and path.stat().st_size > budget:
            errors.append(f"{relative} exceeds its {budget}-byte procedure budget")
        text = path.read_text(encoding="utf-8")
        paragraphs, paragraph_errors = _controlled_paragraphs(text, relative)
        errors.extend(paragraph_errors)
        for number, paragraph in enumerate(paragraphs, start=1):
            ids = POLICY_REF.findall(paragraph)
            if len(ids) != 1:
                errors.append(
                    f"{relative}: controlled paragraph {number} must contain exactly one policy reference"
                )
                continue
            rule_id = ids[0]
            rule = rules.get(rule_id)
            if rule is None:
                errors.append(f"{relative}: unknown policy reference {rule_id}")
            elif rule.procedure != relative:
                errors.append(
                    f"{relative}: {rule_id} belongs to procedure {rule.procedure}"
                )
            else:
                references[rule_id].append(relative)
                rule_paragraphs[rule_id].append(" ".join(paragraph.split()))

    for rule_id, locations in references.items():
        if not locations:
            errors.append(f"{rule_id}: missing procedure back-reference")
        elif len(locations) > 1:
            errors.append(f"{rule_id}: duplicate procedure back-reference")

    required_method = {
        "implementation phase": (
            "Implementation starts from the committed spike",
            "Write or port the smallest test that fails",
            "green focused tests before broader validation",
        ),
        "mutation review": ("static inspection and targeted scratch mutations",),
        "fresh-agent repair": ("fresh implementer",),
        "operator verification": ("operator independently checks",),
        "PR disposition": ("merge the PR in that session",),
    }
    workflow_text = (root / ".agents/workflow.md").read_text(encoding="utf-8") if (root / ".agents/workflow.md").is_file() else ""
    normalized_workflow = " ".join(workflow_text.split())
    for label, fragments in required_method.items():
        if any(fragment not in normalized_workflow for fragment in fragments):
            errors.append(f"workflow missing {label} contract")

    purpose_fragments = {
        "POL-DOC-STATUS": "every feature or iteration checkpoint",
        "POL-DOC-BENCHMARKS": "every feature or iteration checkpoint",
        "POL-DOC-FEATURES": "feature, model, backend, or quantization surface",
        "POL-DOC-USAGE": "commands, C API, configuration, installation, or user workflows",
        "POL-DOC-README": "only for a user-visible landing-page headline",
        "POL-NOW-COUPLING": "same change as every qualifying appended structured event",
    }
    for rule_id, fragment in purpose_fragments.items():
        if not any(fragment in paragraph for paragraph in rule_paragraphs.get(rule_id, [])):
            errors.append(f"workflow public-document purpose contract drift: {rule_id}")

    errors.extend(_validate_policy_links(root, {"AGENTS.md", *procedure_paths}))
    return errors


def _validate_policy_links(root: Path, relative_paths: set[str]) -> list[str]:
    errors: list[str] = []
    retired = set(LEGACY_ACTIVE_PATHS)
    for relative in sorted(relative_paths):
        source = root / relative
        if not source.is_file() or source.is_symlink():
            continue
        text = source.read_text(encoding="utf-8")
        for raw in MARKDOWN_LINK.findall(text):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (source.parent / target_path).resolve(strict=False)
            try:
                normalized = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                errors.append(f"{relative}: Markdown link escapes repository: {target}")
                continue
            if normalized in retired:
                errors.append(f"{relative}: Markdown link uses retired active path: {normalized}")
            elif not resolved.exists():
                errors.append(f"{relative}: broken Markdown link: {target}")
            elif resolved.is_symlink():
                errors.append(f"{relative}: Markdown link targets symlink: {target}")
    return errors
