#!/usr/bin/env python3
"""Validate helper claims from live pull-request state.

The remote is the authority.  Local preflight checks only that no obsolete
snapshot has been committed; ready/integration checks consume live PR JSON.
An unavailable remote is neither an empty claim set nor success.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
COORD = ROOT / ".agents/coordination.md"
ROW_BRANCH = re.compile(r"^row/([A-Za-z0-9][A-Za-z0-9_.-]*)$")
REMOTE_UNVERIFIED_EXIT = 4
OPEN_STATE = "OPEN"
LIVE_FIELDS = (
    "number,state,headRefName,isDraft,title,author,headRepository"
)
FIXTURE_KEYS = frozenset({"expected", "prs"})
EXPECTED_KEYS = frozenset({"repository", "base", "task_id", "head", "number"})
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class RemoteUnverified(RuntimeError):
    """The authoritative PR state could not be obtained or decoded."""


def _load_record(root: Path):
    path = root / "scripts/check-agent-record.py"
    spec = importlib.util.spec_from_file_location("claim_view_agent_record", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load canonical record parser from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _location(row: object, root: Path) -> str:
    path = Path(row.path)
    try:
        rendered = path.relative_to(root).as_posix()
    except ValueError:
        rendered = path.as_posix()
    return f"{rendered}:{row.line_no}"


def canonical_task_rows(root: Path = ROOT) -> list:
    """Parse the task record once, rejecting duplicate identity before indexing."""

    record = _load_record(root)
    errors: list[str] = []
    rows = []
    locations: dict[str, list[str]] = {}
    for path in record.MATRIX_PATHS:
        for row in record.parse_claim_rows(path, errors):
            rows.append(row)
            locations.setdefault(row.item_id, []).append(_location(row, root))
    errors.extend(
        f"duplicate task ID {item_id} at " + ", ".join(found)
        for item_id, found in sorted(locations.items())
        if len(found) > 1
    )
    if errors:
        raise ValueError("canonical task record is invalid: " + "; ".join(errors))
    return rows


def known_task_ids(root: Path = ROOT) -> set[str]:
    """Return task IDs from the existing canonical matrix record.

    The task-aware helper transaction (and its governance-task registry) was
    deliberately deferred from this PR.  This adapter therefore has one source:
    the matrices already parsed by check-agent-record.py.  Live validation and
    readiness both call this function, so they cannot disagree about identity.
    """

    return {row.item_id for row in canonical_task_rows(root)}


def fetch_prs() -> list[dict]:
    """Fetch the open PR set; raise a typed unknown-state result on failure."""

    try:
        result = subprocess.run(
            [
                "gh", "pr", "list", "--state", "open", "--limit", "500",
                "--json", LIVE_FIELDS,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RemoteUnverified(str(exc)) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh exited {result.returncode}"
        raise RemoteUnverified(detail)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteUnverified(f"invalid PR JSON: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RemoteUnverified("PR response must be a JSON array of objects")
    return value


def _closed_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_expected(value: object) -> dict:
    if not isinstance(value, dict):
        raise RemoteUnverified("fixture expected must be an object")
    unknown = sorted(set(value) - EXPECTED_KEYS)
    if unknown:
        raise RemoteUnverified("fixture expected has unknown keys: " + ", ".join(unknown))
    repository = value.get("repository")
    if not isinstance(repository, str) or REPOSITORY.fullmatch(repository) is None:
        raise RemoteUnverified("fixture expected.repository is required and must be owner/name")
    base = value.get("base")
    if base is not None and (not isinstance(base, str) or not base.strip()):
        raise RemoteUnverified("fixture expected.base must be a nonempty string")
    task_id = value.get("task_id")
    if task_id is not None and (
        not isinstance(task_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id) is None
    ):
        raise RemoteUnverified("fixture expected.task_id must be a canonical task ID")
    head = value.get("head")
    if head is not None and (
        not isinstance(head, str)
        or ROW_BRANCH.fullmatch(head) is None
        or task_id is None
    ):
        raise RemoteUnverified("fixture expected.head requires task_id and canonical row head")
    number = value.get("number")
    if number is not None and _pr_number(number) is None:
        raise RemoteUnverified("fixture expected.number must be a positive integer")
    return value


def load_pr_fixture(path: Path) -> tuple[list[dict], dict]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_closed_object
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RemoteUnverified(f"cannot read PR fixture {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != FIXTURE_KEYS:
        raise RemoteUnverified("PR fixture must be a closed {expected, prs} object")
    prs = value["prs"]
    if not isinstance(prs, list) or any(not isinstance(item, dict) for item in prs):
        raise RemoteUnverified("fixture prs must be an array of objects")
    return prs, _validate_expected(value["expected"])


def repository_identity(root: Path = ROOT) -> str:
    """Derive exact owner/name authority from this checkout's origin URL."""

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RemoteUnverified("cannot derive expected repository from git origin")
    remote = result.stdout.strip()
    if "://" in remote:
        path = urlsplit(remote).path
    else:
        match = re.fullmatch(r"[^@:/]+@[^:/]+:(.+)", remote)
        path = match.group(1) if match else remote
    candidate = path.strip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    parts = candidate.split("/")
    identity = "/".join(parts[-2:]) if len(parts) >= 2 else ""
    if REPOSITORY.fullmatch(identity) is None:
        raise RemoteUnverified(f"cannot derive owner/name from origin {remote!r}")
    return identity


def _pr_number(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def validate_live_claims(
    prs: list[dict],
    known_tasks: set[str],
    expected: dict | None = None,
) -> list[str]:
    """Validate a complete live claim set with deterministic diagnostics."""

    errors: list[str] = []
    numbers: dict[int, int] = {}
    task_claims: dict[str, list[int]] = {}
    valid_claims: list[tuple[int, str, str, dict]] = []
    expected_repository = expected.get("repository") if expected is not None else None
    if prs and (
        not isinstance(expected_repository, str)
        or REPOSITORY.fullmatch(expected_repository) is None
    ):
        errors.append("nonempty PR input lacks expected repository identity")

    for index, item in enumerate(prs):
        label = f"PR entry {index + 1}"
        number = _pr_number(item.get("number"))
        if number is None:
            errors.append(f"{label} has malformed PR number")
        else:
            numbers[number] = numbers.get(number, 0) + 1
            label = f"PR #{number}"

        state = item.get("state")
        if state not in {"OPEN", "CLOSED", "MERGED"}:
            errors.append(f"{label} has malformed state {state!r}")

        repository = item.get("headRepository")
        repository_name = (
            repository.get("nameWithOwner") if isinstance(repository, dict) else None
        )
        if not isinstance(repository_name, str) or REPOSITORY.fullmatch(repository_name) is None:
            errors.append(f"{label} has malformed head repository identity")
        elif expected_repository is not None and repository_name != expected_repository:
            errors.append(
                f"{label} repository {repository_name!r} does not match expected "
                f"{expected_repository!r}"
            )

        head = item.get("headRefName")
        if not isinstance(head, str):
            errors.append(f"{label} has malformed head identity")
            continue
        match = ROW_BRANCH.fullmatch(head)
        if match is None:
            # Unrelated PRs are not claims.  A row-like malformed branch is an
            # attempted claim and must not disappear from validation.
            if head.startswith("row/"):
                errors.append(f"{label} has malformed row head {head!r}")
            continue

        task = match.group(1)
        if state != OPEN_STATE:
            errors.append(f"{label} for task {task} is {state}, not a live claim")
        if task not in known_tasks:
            errors.append(f"{label} claims unknown task {task}")
        if number is not None:
            task_claims.setdefault(task, []).append(number)
            valid_claims.append((number, task, head, item))

    for number, count in sorted(numbers.items()):
        if count > 1:
            errors.append(f"PR number #{number} is ambiguous ({count} entries)")
    for task, claims in sorted(task_claims.items()):
        distinct = sorted(set(claims))
        if len(distinct) > 1:
            rendered = ", ".join(f"PR #{number}" for number in distinct)
            errors.append(f"task {task} has duplicate open claims: {rendered}")

    if expected is not None and "task_id" in expected:
        task = expected.get("task_id")
        if not isinstance(task, str) or task not in known_tasks:
            errors.append(f"expected claim has unknown or malformed task_id {task!r}")
        else:
            matches = [claim for claim in valid_claims if claim[1] == task]
            if not matches:
                if len(valid_claims) == 1:
                    number, _, head, _ = valid_claims[0]
                    expected_head = expected.get("head", f"row/{task}")
                    errors.append(
                        f"PR #{number} has wrong head {head!r}; expected {expected_head!r}"
                    )
                errors.append(f"expected live claim for task {task} is missing")
            else:
                expected_head = expected.get("head", f"row/{task}")
                for number, _, head, item in matches:
                    if head != expected_head:
                        errors.append(
                            f"PR #{number} has wrong head {head!r}; expected {expected_head!r}"
                        )
                expected_number = expected.get("number")
                if expected_number is not None and not any(
                    number == expected_number for number, *_ in matches
                ):
                    got = ", ".join(f"#{number}" for number, *_ in matches)
                    errors.append(
                        f"expected PR #{expected_number} for task {task}; found {got}"
                    )
    return sorted(set(errors))


def claimed_tasks(prs: list[dict]) -> set[str]:
    """Return only canonical OPEN row claims after callers validate the set."""

    result = set()
    for item in prs:
        match = ROW_BRANCH.fullmatch(item.get("headRefName", ""))
        if match is not None and item.get("state") == OPEN_STATE:
            result.add(match.group(1))
    return result


def local_errors(text: str) -> list[str]:
    if "<!-- claim-view:begin -->" in text or "<!-- claim-view:end -->" in text:
        return [
            "committed claim-view snapshots are forbidden; live PR state is authoritative"
        ]
    return []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-local", action="store_true")
    mode.add_argument("--check-live", action="store_true")
    mode.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pr-json", type=Path, help="offline live-state fixture")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check_local or args.check:
        if args.pr_json is not None:
            _parser().error("--pr-json requires --check-live")
        failures = local_errors(COORD.read_text(encoding="utf-8"))
    else:
        try:
            if args.pr_json:
                prs, expected = load_pr_fixture(args.pr_json)
            else:
                prs = fetch_prs()
                expected = {"repository": repository_identity(ROOT)}
            failures = validate_live_claims(prs, known_task_ids(ROOT), expected)
        except (RemoteUnverified, ValueError) as exc:
            print(f"REMOTE_UNVERIFIED: {exc}", file=sys.stderr)
            return REMOTE_UNVERIFIED_EXIT

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    mode = "local record" if (args.check_local or args.check) else "live PR claims"
    print(f"OK: {mode} validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
