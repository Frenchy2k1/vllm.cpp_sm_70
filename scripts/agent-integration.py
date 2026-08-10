#!/usr/bin/env python3
"""Gate integration on readiness, review, trailers, cutover, and fresh base."""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CUTOVER_FILE = ROOT / ".agents/policy-cutover"


def _load_ready():
    spec = importlib.util.spec_from_file_location("agent_ready", ROOT / "scripts/agent-ready.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ready = _load_ready()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def cutover_oid() -> str:
    try:
        text = CUTOVER_FILE.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"missing policy cutover: {exc}") from exc
    if re.fullmatch(r"[0-9a-f]{40}\n", text) is None:
        raise ValueError("policy cutover must contain exactly one lowercase 40-hex commit")
    oid = text.strip()
    exists = _git("cat-file", "-e", f"{oid}^{{commit}}")
    ancestor = _git("merge-base", "--is-ancestor", oid, "HEAD")
    if exists.returncode or ancestor.returncode:
        raise ValueError("policy cutover is not a commit ancestor of HEAD")
    return oid


def base_freshness_errors(base: str, expected_base_oid: str) -> list[str]:
    resolved = _git("rev-parse", f"{base}^{{commit}}")
    if resolved.returncode:
        return [f"base {base!r} does not resolve"]
    oid = resolved.stdout.strip()
    errors: list[str] = []
    if oid != expected_base_oid:
        errors.append("local integration base does not match the current PR base SHA")
    if _git("merge-base", "--is-ancestor", oid, "HEAD").returncode:
        errors.append("integration base is not an ancestor of HEAD")
    merge_base = _git("merge-base", oid, "HEAD")
    if merge_base.returncode or merge_base.stdout.strip() != oid:
        errors.append("HEAD is not fresh against the complete integration base")
    return errors


def review_errors(payload: dict[str, Any], expected: dict[str, str]) -> list[str]:
    candidates = [
        pr for pr in payload.get("prs", [])
        if isinstance(pr, dict) and pr.get("headRefName") == expected["head_branch"]
    ]
    if len(candidates) != 1:
        return ["cannot disposition review without one exact live PR"]
    if candidates[0].get("reviewDecision") != "APPROVED":
        return ["PR review disposition is not APPROVED"]
    return []


def run_ready(fixture: Path | None) -> bool:
    command = [sys.executable, str(ROOT / "scripts/agent-ready.py")]
    if fixture is not None:
        command.extend(("--pr-json", str(fixture)))
    return subprocess.run(command, cwd=ROOT).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pr-json", type=Path)
    args = parser.parse_args()
    if not run_ready(args.pr_json):
        print("INTEGRATION FAILED: ready gate is red", file=sys.stderr)
        return 1
    try:
        expected = ready.local_expected()
        payload = ready.load_payload(args.pr_json) if args.pr_json else ready.query_remote(expected)
        errors = ready.ready_errors(payload, expected)
        errors.extend(review_errors(payload, expected))
        errors.extend(base_freshness_errors(args.base, expected["base_oid"]))
        cutover = cutover_oid()
    except (ready.RemoteUnverified, ValueError) as exc:
        print(f"INTEGRATION FAILED: {exc}", file=sys.stderr)
        return 1
    trailers = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-commit-trailers.py"),
         "--range", f"{args.base}..HEAD", "--cutover", cutover],
        cwd=ROOT,
    )
    if trailers.returncode:
        errors.append("post-cutover trailer validation failed")
    if errors:
        for error in errors:
            print(f"INTEGRATION FAILED: {error}", file=sys.stderr)
        return 1
    print("INTEGRATION: ready, review, trailers, cutover, and base are green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
