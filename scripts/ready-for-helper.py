#!/usr/bin/env python3
"""Offer only helper tasks whose readiness is mechanically proven."""

from __future__ import annotations

import argparse
import io
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import PurePosixPath
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PICKABLE_STATES = frozenset({"READY"})
SATISFIED_STATES = frozenset({"DONE", "NOT-APPLICABLE"})
PLACEHOLDERS = frozenset({"", "-", "none", "n/a", "not applicable"})
LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")
CONTRACT = re.compile(
    r"<!--\s*helper-readiness:v1\s*\n(?P<body>.*?)\n\s*-->", re.DOTALL
)
CONTRACT_KEYS = frozenset({"gate", "mutation"})
ALLOWED_PROGRAMS = frozenset(
    {"python3", "python", "pytest", "ctest", "cmake", "make", "bash", "sh"}
)
SHELL_META = re.compile(r"[;&|`$<>\\\r\n*?{}\[\]]")
ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
RUN_TIMEOUT_SECONDS = 30
DIAGNOSTIC_LIMIT = 2000
CPU_GATE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:gate\s+)?hardware\s*:\s*CPU\s*\.??\s*$")
EXACT_HARDWARE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:gate\s+)?hardware\s*:\s*"
    r"(?=[^\n]*(?:\bGB[0-9]+\b|\bsm_[0-9]+\b|\bsm[0-9]{2,3}\b))"
    r"[^\n]+\s*\.??\s*$"
)


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rec = _load("ready_agent_record", "scripts/check-agent-record.py")
claim_view = _load("claim_view", "scripts/claim-view.py")


def _field(row, name: str) -> str:
    direct = row.field(name)
    if direct:
        return direct
    wanted = rec.normalize_header(name)
    for index, header in enumerate(row.header):
        if header == wanted or (name == "dependencies" and "dependenc" in header):
            return row.cells[index] if index < len(row.cells) else ""
    return ""


def _spec_paths(row, root: Path) -> list[str]:
    """Return lexical repository paths without consulting mutable worktree files."""

    spec_root = PurePosixPath(".agents/specs")
    paths: list[str] = []
    for target in LINK.findall(_field(row, "spec")):
        raw = target.strip().strip("<>").split("#", 1)[0]
        if not raw or raw.startswith(("http://", "https://")):
            continue
        try:
            matrix_relative = row.path.relative_to(root).parent.as_posix()
        except ValueError:
            continue
        candidate = PurePosixPath(matrix_relative) / PurePosixPath(raw)
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        normalized = PurePosixPath(*[part for part in candidate.parts if part != "."])
        try:
            normalized.relative_to(spec_root)
        except ValueError:
            continue
        if normalized.suffix == ".md":
            paths.append(normalized.as_posix())
    return paths


def _git(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _resolve_base(root: Path, base: str) -> tuple[str | None, str | None]:
    if not isinstance(base, str) or not base.strip():
        return None, "base commit is not configured"
    result = _git(root, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}")
    commit = result.stdout.strip() if result.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return None, f"base commit {base!r} is not configured and reachable"
    return commit, None


def _base_mode(root: Path, commit: str, relative: str) -> str | None:
    result = _git(root, "ls-tree", "-z", commit, "--", relative, text=False)
    if result.returncode != 0 or not result.stdout:
        return None
    entries = result.stdout.split(b"\0")
    exact = []
    for entry in entries:
        if not entry or b"\t" not in entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        if path.decode("utf-8", "surrogateescape") == relative:
            exact.append(metadata.split(b" ", 1)[0].decode("ascii", "replace"))
    return exact[0] if len(exact) == 1 else None


def _base_blob(root: Path, commit: str, relative: str) -> str | None:
    mode = _base_mode(root, commit, relative)
    if mode not in {"100644", "100755"}:
        return None
    result = _git(root, "show", f"{commit}:{relative}", text=False)
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _reachable_spec(row, root: Path, commit: str) -> tuple[str, str] | None:
    for relative in _spec_paths(row, root):
        text = _base_blob(root, commit, relative)
        if text is not None:
            return relative, text
    return None


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _parse_contract(text: str) -> tuple[dict[str, list[str]] | None, list[str]]:
    matches = list(CONTRACT.finditer(text))
    if len(matches) != 1:
        return None, ["spec needs exactly one structured helper-readiness:v1 block"]
    try:
        value = json.loads(
            matches[0].group("body"), object_pairs_hook=_pairs_no_duplicates
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return None, [f"structured readiness JSON is invalid: {exc}"]
    if not isinstance(value, dict) or set(value) != CONTRACT_KEYS:
        return None, ["structured readiness object must contain exactly gate and mutation"]
    errors = []
    for name in ("gate", "mutation"):
        argv = value.get(name)
        if (
            not isinstance(argv, list)
            or not argv
            or len(argv) > 64
            or any(not isinstance(arg, str) or not arg or len(arg) > 1024 for arg in argv)
        ):
            errors.append(
                f"structured {name} must be a nonempty bounded argv string array"
            )
    return (value if not errors else None), errors


def _validate_argv(
    name: str, argv: list[str], root: Path, commit: str
) -> list[str]:
    errors: list[str] = []
    for arg in argv:
        path = PurePosixPath(arg)
        if (
            path.is_absolute()
            or ".." in path.parts
            or re.search(r"(?:^|[=/])\.\.(?:/|$)", arg)
            or re.search(r"(?:^|=)/", arg)
            or SHELL_META.search(arg)
            or ENV_ASSIGNMENT.match(arg)
            or arg == "-c"
            or arg.startswith("-c")
            or arg == "--eval"
            or arg.startswith("--eval=")
            or "\x00" in arg
        ):
            errors.append(f"unsafe {name} argv element {arg!r}")
    if errors:
        return errors

    program = argv[0]
    if "/" not in program:
        if program not in ALLOWED_PROGRAMS:
            return [f"{name} executable {program!r} is not in the allowlist"]
        executable = shutil.which(program, path=os.defpath)
        if executable is None or not Path(executable).resolve().is_file():
            return [f"{name} executable {program!r} is missing or nonregular"]
        if program in {"python3", "python", "bash", "sh"} and (
            len(argv) < 2 or argv[1].startswith("-")
        ):
            return [f"unsafe {name} argv: {program} requires a relative base script"]
        if program in {"python3", "python", "bash", "sh"} and _base_mode(
            root, commit, PurePosixPath(argv[1]).as_posix()
        ) not in {"100644", "100755"}:
            return [f"{name} argv path {argv[1]!r} is not a regular base file"]
    else:
        mode = _base_mode(root, commit, PurePosixPath(program).as_posix())
        if mode != "100755":
            return [f"{name} executable must be a regular executable base file"]

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        mode = _base_mode(root, commit, PurePosixPath(arg).as_posix())
        if "/" not in arg and mode is None:
            continue
        if mode not in {"100644", "100755"}:
            errors.append(f"{name} argv path {arg!r} is not a regular base file")
    return errors


def _bounded(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value[:DIAGNOSTIC_LIMIT]


def _safe_tmp_root() -> Path | None:
    candidate = Path(os.environ.get("TMPDIR", "/tmp"))
    try:
        info = candidate.lstat()
    except OSError:
        return None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or candidate.resolve() != candidate
        or info.st_uid not in {0, os.getuid()}
        or (info.st_mode & stat.S_IWOTH and not info.st_mode & stat.S_ISVTX)
    ):
        return None
    return candidate


def _execute_contract(
    root: Path, commit: str, contract: dict[str, list[str]]
) -> list[str]:
    errors = []
    for name in ("gate", "mutation"):
        errors.extend(_validate_argv(name, contract[name], root, commit))
    if errors:
        return errors

    safe_tmp = _safe_tmp_root()
    if safe_tmp is None:
        return ["configured safe temporary directory is unavailable"]
    try:
        archive = _git(root, "archive", "--format=tar", commit, text=False)
        if archive.returncode != 0:
            return ["could not materialize exact base archive"]
        with tempfile.TemporaryDirectory(prefix="vllm-helper-ready-", dir=safe_tmp) as directory:
            checkout = Path(directory) / "checkout"
            checkout.mkdir(mode=0o700)
            with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as stream:
                stream.extractall(checkout, filter="data")
            run_tmp = checkout / ".helper-tmp"
            run_tmp.mkdir(mode=0o700)
            environment = {
                "HOME": str(checkout),
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.defpath,
                "PYTHONHASHSEED": "0",
                "TMPDIR": str(run_tmp),
            }
            results = {}
            for name in ("gate", "mutation"):
                try:
                    results[name] = subprocess.run(
                        contract[name],
                        cwd=checkout,
                        env=environment,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=False,
                        timeout=RUN_TIMEOUT_SECONDS,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"{name} could not execute: {_bounded(str(exc))}")
            gate = results.get("gate")
            mutation = results.get("mutation")
            if gate is not None and gate.returncode != 0:
                detail = _bounded(gate.stderr or gate.stdout).strip()
                errors.append(
                    f"gate exited {gate.returncode}" + (f": {detail}" if detail else "")
                )
            if mutation is not None and mutation.returncode == 0:
                errors.append("mutation unexpectedly exited 0")
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"could not materialize exact base archive: {_bounded(str(exc))}")
    return errors


def _hardware_proof(text: str) -> bool:
    return bool(CPU_GATE.search(text) or EXACT_HARDWARE.search(text))


def _parse_dependencies(value: str, known_tasks: set[str]) -> tuple[list[str], bool]:
    normalized = value.strip().strip("`").strip().casefold()
    if normalized in PLACEHOLDERS:
        return [], True
    quoted = re.findall(r"`([A-Za-z0-9][A-Za-z0-9_.-]*)`", value)
    if not quoted or any(item not in known_tasks for item in quoted):
        return [], False
    residue = re.sub(r"`[A-Za-z0-9][A-Za-z0-9_.-]*`", "", value)
    residue = re.sub(r"[\s,;+&()]+", "", residue)
    if residue:
        return [], False
    return quoted, True


def evaluate(
    row,
    live_claims: set[str],
    known_tasks: set[str],
    task_states: dict[str, str],
    *,
    root: Path = ROOT,
    base: str = "origin/main",
) -> list[str]:
    """Return every failed readiness proof for one candidate row."""

    missing: list[str] = []
    if row.item_id not in known_tasks:
        missing.append("unknown task ID")
    if row.state not in PICKABLE_STATES:
        missing.append(f"lifecycle {row.state} is not pickable")
        return missing

    commit, base_error = _resolve_base(root, base)
    if base_error is not None or commit is None:
        missing.append(base_error or "base commit is unavailable")
        text = ""
    else:
        spec = _reachable_spec(row, root, commit)
        if spec is None:
            text = ""
        else:
            _, text = spec

    if text:
        contract, contract_errors = _parse_contract(text)
        missing.extend(contract_errors)
        if contract is not None and commit is not None:
            missing.extend(_execute_contract(root, commit, contract))
    elif base_error is None:
        missing.append("no base-reachable committed spec")
    if not _hardware_proof(text):
        missing.append("CPU or exact gate hardware is not declared")

    dependencies, parsed = _parse_dependencies(
        _field(row, "dependencies"), known_tasks
    )
    if not parsed:
        missing.append("dependencies are not fully parsed")
    else:
        for dependency in dependencies:
            state = task_states.get(dependency)
            if state not in SATISFIED_STATES:
                missing.append(
                    f"dependency {dependency} is {state or 'UNKNOWN'}, not satisfied"
                )

    if row.item_id in live_claims:
        missing.append("already claimed by an open PR")
    return missing


def _rows(root: Path = ROOT) -> list:
    return claim_view.canonical_task_rows(root)


def queue(
    prs: list[dict] | None = None,
    *,
    root: Path = ROOT,
    base: str = "origin/main",
    expected: dict | None = None,
) -> tuple[list, dict[str, int]]:
    if prs is None:
        # Local callers (including onboarding and preflight before Task22's
        # final wiring) get no advertised work, never an assumed-empty remote.
        rows = _rows(root)
        return [], {
            "live claims not verified": sum(
                row.state in PICKABLE_STATES for row in rows
            )
        }
    rows = _rows(root)
    known = {row.item_id for row in rows}
    if expected is None:
        expected = {"repository": claim_view.repository_identity(root), "base": base}
    claim_errors = claim_view.validate_live_claims(prs, known, expected)
    if claim_errors:
        raise ValueError("; ".join(claim_errors))
    states = {row.item_id: row.state for row in rows}
    live = claim_view.claimed_tasks(prs)
    pickable, reasons = [], {}
    for row in rows:
        failures = evaluate(
            row,
            live,
            known,
            states,
            root=root,
            base=base,
        )
        if failures:
            for reason in failures:
                reasons[reason] = reasons.get(reason, 0) + 1
        else:
            pickable.append(row)
    return pickable, reasons


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--check-local", action="store_true")
    mode.add_argument("--check-live", action="store_true")
    parser.add_argument("--pr-json", type=Path, help="offline live-state fixture")
    parser.add_argument("--base", default="origin/main")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    # Compatibility until the final gate-wiring task updates agent-preflight:
    # bare --check is deliberately LOCAL_ONLY.  It validates no remote absence
    # and advertises no queue.  Supplying --pr-json, or printing the queue
    # without --check, is the remote-aware path.
    local_check = args.check or args.check_local
    if local_check and args.pr_json is not None:
        parser.error("local checks reject --pr-json")
    if local_check and args.base != "origin/main":
        parser.error("local checks reject --base")
    if args.pr_json is not None and not args.check_live:
        parser.error("--pr-json requires --check-live")
    if local_check:
        local = claim_view.local_errors(
            (ROOT / ".agents/coordination.md").read_text(encoding="utf-8")
        )
        try:
            rows = _rows(ROOT)
            known = {row.item_id for row in rows}
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if local:
            for failure in local:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 1
        unknown = sorted(row.item_id for row in rows if row.item_id not in known)
        if unknown:
            print("ERROR: local rows contain unknown task IDs: " + ", ".join(unknown), file=sys.stderr)
            return 1
        print(
            "OK: LOCAL_ONLY readiness structure validated; live claims were not "
            "verified and no helper queue was advertised."
        )
        return 0
    if not args.check_live and args.pr_json is None:
        pickable, reasons = queue(None, base=args.base)
        print(f"READY-FOR-HELPER queue: {len(pickable)} task(s) [LOCAL_ONLY]\n")
        print("Live claims were not verified; no helper task is advertised.")
        if reasons:
            for reason, count in sorted(reasons.items()):
                print(f"  {count:4}  {reason}")
        return 0
    try:
        if args.pr_json is not None:
            prs, expected = claim_view.load_pr_fixture(args.pr_json)
        else:
            prs = claim_view.fetch_prs()
            expected = {
                "repository": claim_view.repository_identity(ROOT),
                "base": args.base,
            }
        known = claim_view.known_task_ids(ROOT)
        failures = claim_view.validate_live_claims(prs, known, expected)
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 1
        base = expected.get("base", args.base)
        pickable, reasons = queue(prs, base=base, expected=expected)
    except claim_view.RemoteUnverified as exc:
        print(f"REMOTE_UNVERIFIED: {exc}", file=sys.stderr)
        return claim_view.REMOTE_UNVERIFIED_EXIT
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            f"OK: {len(pickable)} task(s) are READY-FOR-HELPER; every offered "
            "task has reachable spec, failing-mutation gate, hardware, dependencies, "
            "lifecycle and live-claim proof."
        )
        return 0

    print(f"READY-FOR-HELPER queue: {len(pickable)} task(s)\n")
    for row in pickable:
        print(f"  {row.item_id[:64]:64} {row.state:8} {row.path.name}")
    if reasons:
        print("\nWhy the rest are not pickable:")
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count:4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
