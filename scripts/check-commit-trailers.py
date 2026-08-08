#!/usr/bin/env python3
"""Validate commit messages with Git's own trailer parser."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    from scripts.policy_contract import PolicyRule, Waiver, load_policy, load_waivers
except ModuleNotFoundError:  # direct execution from outside the repository root
    from policy_contract import PolicyRule, Waiver, load_policy, load_waivers


ROOT = Path(__file__).resolve().parents[1]
RAW_PROTOCOL_MARKER = "FOLLOWING_AGENTS_PROTOCOL"
PROTOCOL_RULE = "POL-COMMIT-TRAILERS"
ATTRIBUTION_RULE = "POL-AI-ATTRIBUTION"
ASSISTED_BY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_.+-]*"
    r"(?: \[[A-Za-z0-9][A-Za-z0-9_. +:/-]*\])+\Z"
)
# Closed, reviewable vocabulary for obvious agent/vendor/model/tool authorship.
# Boundaries prevent human names such as "Alice" from matching the token "ai".
AI_AUTHORSHIP_TOKENS = (
    "ai",
    "agent",
    "anthropic",
    "bot",
    "chatgpt",
    "claude",
    "claudecode",
    "codex",
    "copilot",
    "gemini",
    "gpt",
    "llm",
    "openai",
)
AI_IDENTITY = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(token) for token in AI_AUTHORSHIP_TOKENS)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Git command failed"
        raise ValueError(detail)
    return result.stdout.strip()


def parsed_trailers(message: str) -> str:
    """Return exactly what ``git interpret-trailers --parse`` returns."""

    result = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=message,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _paragraphs(message: str) -> list[str]:
    normalized = message.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return re.split(r"\n[ \t]*\n+", normalized) if normalized else []


def _trailer_map(message: str) -> dict[str, list[tuple[str, str]]]:
    trailers: dict[str, list[tuple[str, str]]] = {}
    for line in parsed_trailers(message).splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        trailers.setdefault(key.casefold(), []).append((key, value))
    return trailers


def _strict_errors(message: str) -> list[str]:
    errors: list[str] = []
    paragraphs = _paragraphs(message)
    marker_indexes = [
        index for index, paragraph in enumerate(paragraphs)
        if paragraph == RAW_PROTOCOL_MARKER
    ]
    if len(marker_indexes) != 1 or marker_indexes[0] == len(paragraphs) - 1:
        errors.append(
            f"[{PROTOCOL_RULE}] {RAW_PROTOCOL_MARKER} must appear exactly once "
            "as a separate paragraph before the trailer paragraph"
        )

    trailers = _trailer_map(message)
    protocol = trailers.get("following-agents-protocol", [])
    if len(protocol) != 1:
        errors.append(
            f"[{PROTOCOL_RULE}] Following-Agents-Protocol must appear exactly once"
        )
    elif protocol[0][1] != "true":
        errors.append(
            f"[{PROTOCOL_RULE}] Following-Agents-Protocol must be exactly true"
        )

    declarations = trailers.get("ai-assisted", [])
    assisted = trailers.get("assisted-by", [])
    if len(declarations) != 1:
        errors.append(f"[{ATTRIBUTION_RULE}] AI-Assisted must appear exactly once")
    else:
        declaration = declarations[0][1]
        if declaration not in {"true", "false"}:
            errors.append(f"[{ATTRIBUTION_RULE}] AI-Assisted must be true or false")
        elif declaration == "true" and not assisted:
            errors.append(
                f"[{ATTRIBUTION_RULE}] AI-Assisted true requires Assisted-by"
            )
        elif declaration == "false" and assisted:
            errors.append(
                f"[{ATTRIBUTION_RULE}] AI-Assisted false must omit Assisted-by"
            )

    for _, value in assisted:
        if ASSISTED_BY.fullmatch(value) is None:
            errors.append(f"[{ATTRIBUTION_RULE}] malformed Assisted-by value {value!r}")

    assisted_identities: set[str] = set()
    for _, value in assisted:
        if ASSISTED_BY.fullmatch(value) is None:
            continue
        identity, remainder = value.split(":", 1)
        model = remainder.split(" ", 1)[0]
        assisted_identities.update((identity.casefold(), model.casefold()))
        assisted_identities.update(
            tool.casefold() for tool in re.findall(r"\[([^]]+)\]", value)
        )
    for key in ("signed-off-by", "co-authored-by"):
        for original, value in trailers.get(key, []):
            folded_value = value.casefold()
            if AI_IDENTITY.search(value) or any(
                identity in folded_value for identity in assisted_identities
            ):
                errors.append(
                    f"[{ATTRIBUTION_RULE}] AI authorship trailer {original} is forbidden"
                )
    return errors


def validate_waiver_targets(repo: Path, waivers: Iterable[Waiver]) -> None:
    """Reject path waivers whose supposedly exact target is a directory."""

    for waiver in waivers:
        if not waiver.scope.startswith("path:"):
            continue
        target = waiver.scope.removeprefix("path:")
        candidate = repo / target
        if candidate.is_dir():
            raise ValueError(
                f"waiver {waiver.waiver_id} path target {target!r} is a directory"
            )


def validate_commit_message(message: str, *, strict: bool) -> list[str]:
    """Return trailer-contract errors for one complete commit message."""

    if not strict:
        marker_count = sum(
            paragraph == RAW_PROTOCOL_MARKER for paragraph in _paragraphs(message)
        )
        if marker_count == 1:
            return []
    return _strict_errors(message)


def exact_waiver(
    waivers: Iterable[Waiver], rule_id: str, scope: str
) -> Waiver | None:
    """Select one exact, already-validated waiver or fail on ambiguity."""

    matches = [
        waiver
        for waiver in waivers
        if waiver.rule_id == rule_id and waiver.scope == scope
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate applicable waivers for {rule_id} {scope}")
    return matches[0] if matches else None


def _resolve_commit(repo: Path, revision: str) -> str:
    if not revision or "\x00" in revision or "\n" in revision:
        raise ValueError(f"invalid revision {revision!r}")
    if not revision.startswith("refs/"):
        candidates = (
            f"refs/heads/{revision}",
            f"refs/tags/{revision}",
            f"refs/remotes/{revision}",
        )
        matches = 0
        for candidate in candidates:
            result = subprocess.run(
                ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", candidate],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                matches += 1
            elif result.returncode not in {1}:
                raise ValueError(f"could not resolve revision {revision!r}")
        if matches > 1:
            raise ValueError(f"ambiguous revision {revision!r}")
    resolved = _git(
        repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
    ).splitlines()
    if len(resolved) != 1 or re.fullmatch(r"[0-9a-f]{40}", resolved[0]) is None:
        raise ValueError(f"revision {revision!r} did not resolve to one commit")
    return resolved[0]


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in {0, 1}:
        raise ValueError("could not establish commit ancestry")
    return result.returncode == 0


def _error_rule(error: str) -> str:
    match = re.match(r"\[([^]]+)\]", error)
    return match.group(1) if match else PROTOCOL_RULE


def validate_range(
    repo: Path,
    base: str,
    head: str,
    *,
    cutover: str | None,
    rules: dict[str, PolicyRule] | None,
    waivers: Iterable[Waiver],
) -> list[str]:
    """Validate an exact first-parent-independent ``BASE..HEAD`` commit set."""

    base_oid = _resolve_commit(repo, base)
    head_oid = _resolve_commit(repo, head)
    if not _is_ancestor(repo, base_oid, head_oid):
        raise ValueError("range base must be an ancestor of range head")
    cutover_oid = _resolve_commit(repo, cutover) if cutover is not None else None
    if cutover_oid is not None and not _is_ancestor(repo, cutover_oid, head_oid):
        raise ValueError("cutover must be reachable from range head")

    commits_text = _git(repo, "rev-list", "--reverse", f"{base_oid}..{head_oid}")
    failures: list[str] = []
    for commit in (line for line in commits_text.splitlines() if line):
        if cutover_oid is None:
            strict = True
        elif _is_ancestor(repo, cutover_oid, commit):
            strict = True
        elif _is_ancestor(repo, commit, cutover_oid):
            strict = False
        else:
            raise ValueError(f"commit {commit} is incomparable with cutover")

        message = _git(repo, "show", "-s", "--format=%B", commit) + "\n"
        for error in validate_commit_message(message, strict=strict):
            rule_id = _error_rule(error)
            if rules is not None and rule_id not in rules:
                raise ValueError(f"policy registry is missing {rule_id}")
            if exact_waiver(waivers, rule_id, f"commit:{commit}") is not None:
                continue
            failures.append(f"{commit[:12]}: {error}")
    return failures


def _range(value: str) -> tuple[str, str]:
    if value.count("..") != 1 or "..." in value:
        raise argparse.ArgumentTypeError("range must be exactly BASE..HEAD")
    base, head = value.split("..", 1)
    if not base or not head:
        raise argparse.ArgumentTypeError("range must be exactly BASE..HEAD")
    return base, head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="revision_range", type=_range, required=True)
    parser.add_argument("--cutover")
    args = parser.parse_args()
    try:
        rules = load_policy(ROOT)
        waivers = load_waivers(ROOT, rules)
        validate_waiver_targets(ROOT, waivers)
        failures = validate_range(
            ROOT,
            *args.revision_range,
            cutover=args.cutover,
            rules=rules,
            waivers=waivers,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"commit trailer check FAILED: {exc}", file=sys.stderr)
        return 1
    if failures:
        print("commit trailer check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("OK: commit trailer contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
