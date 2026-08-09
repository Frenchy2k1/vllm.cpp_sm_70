#!/usr/bin/env python3
"""Feature code reaches main only through a reviewed row/* PR. (W1)

This is how "the operator drives features only via sub-agents" is enforced. It
deliberately does NOT try to detect who typed the code -- authorship is
self-asserted and unprovable. It enforces the PATH instead: feature code arrives
on `main` through a merged `row/<ROW-ID>` PR, whoever produced it. A sub-agent, a
helper session, or the developer all satisfy it the same way, and a direct push
of feature code does not.

Non-product paths are classified by closed lexical forms rather than exempting
whole mutable trees.  This checker still owns the feature-arrival rule; the PR
and size gates independently review the other classes.

ACTIVATION. ENFORCING since the cutover commit 44e8225cf (user-directed
2026-08-05). Every commit from that one ONWARD must land feature code through a
merged `row/*` PR; everything before it is exempt, because it was created under
the previous direct-push policy and rewriting that judgement retroactively would
redden honest history. The cutover itself is a records-only commit, so it passes.

What this changes in practice: feature paths (src/, include/, tests/, examples/,
cmake/, CMakeLists.txt) can no longer be pushed straight to main. Integration
paths (scripts/, .agents/, docs/, .github/) still can, deliberately, so the
operator can fix a gate or repair the record without a round trip.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath


ROOT = Path(__file__).resolve().parents[1]

# Set to the cutover commit SHA to switch enforcement on. None = report only.
ROLE_DISCIPLINE_SINCE: str | None = "44e8225cf95fff12de6c5d4f3c3b4ecc9f0b1f94"

# Product code. A change here must arrive through a reviewed row/* PR.
FEATURE_PREFIXES = (
    "src/",
    "include/",
    "examples/",
    "tools/",
    "tests/",
    "cmake/",
)
FEATURE_FILES = {"CMakeLists.txt"}
INTEGRATION_FILES = {
    ".agents/state.md",
    ".agents/state.csv",
    ".agents/completed/state-migration-manifest.csv",
    "docs/STATUS.md",
    "docs/BENCHMARKS.md",
    "docs/FEATURES.md",
    "docs/USAGE.md",
    "README.md",
}
CHECKER_PATH = re.compile(r"scripts/check-[A-Za-z0-9_.-]+\.(?:py|sh)\Z")
CHECKER_TEST_PATH = re.compile(r"tests/scripts/test_[A-Za-z0-9_.-]+\.py\Z")
CI_PATH = re.compile(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml\Z")
STRUCTURED_STATE_PATH = re.compile(
    r"\.agents/(?:state-index/\d{4}-\d{2}-\d{3}\.csv|"
    r"state-events/\d{4}-\d{2}/STATE-[A-Za-z0-9-]+\.md)\Z"
)

ROW_BRANCH = re.compile(r"row/[A-Za-z0-9_.-]+")
PR_REFERENCE = re.compile(r"\(#\d+\)|#\d+")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def is_integration_path(path: str) -> bool:
    """Return whether path is an explicit record, checker, or CI surface."""

    return bool(
        path in INTEGRATION_FILES
        or STRUCTURED_STATE_PATH.fullmatch(path)
        or CHECKER_PATH.fullmatch(path)
        or CHECKER_TEST_PATH.fullmatch(path)
        or CI_PATH.fullmatch(path)
    )


def is_feature_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or "\\" in path
        or "//" in path
        or candidate.as_posix() != path
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        return True
    if is_integration_path(path):
        return False
    return path in FEATURE_FILES or path.startswith(FEATURE_PREFIXES)


def arrives_via_row_pr(
    parents: list[str], subject: str, body: str, merged_messages: tuple[str, ...] = ()
) -> bool:
    """Whether this commit reached main through a reviewed row/* PR."""
    message = f"{subject}\n{body}"
    if len(parents) >= 2:
        # A merge commit is a PR merge when it names the branch or the PR.
        if ROW_BRANCH.search(message) or PR_REFERENCE.search(message):
            return True
        # ... or when the branch it MERGES IN does. GitHub builds a SYNTHETIC
        # merge for `refs/pull/N/merge` whose entire message is
        # "Merge <head> into <base>": it names neither the row branch nor the PR,
        # and it never lands on main. CI checks out exactly that commit, so every
        # feature PR failed a gate about MAIN's history, on a commit that is not
        # main's history. The reviewed content is the SECOND parent, the PR head,
        # so a merge of a branch whose own commits name the row IS arrival
        # through a row PR, one hop away. A merge of a branch that names neither
        # still fails, which is the case this gate exists for.
        return any(
            bool(ROW_BRANCH.search(m) or PR_REFERENCE.search(m)) for m in merged_messages
        )
    # GitHub squash-merges land a single commit carrying "(#N)".
    return bool(ROW_BRANCH.search(message) or PR_REFERENCE.search(subject))


def commit_violations(
    commit: str,
    parents: list[str],
    subject: str,
    body: str,
    paths: list[str],
    merged_messages: tuple[str, ...] = (),
) -> list[str]:
    """Return the reasons this commit breaks role discipline (empty if fine)."""
    features = sorted(p for p in paths if is_feature_path(p))
    if not features:
        return []
    if arrives_via_row_pr(parents, subject, body, merged_messages):
        return []
    preview = ", ".join(features[:4])
    if len(features) > 4:
        preview += f", ... (+{len(features) - 4})"
    return [
        f"{commit}: feature code ({preview}) reached main without a reviewed "
        "row/* PR. Feature work goes through a helper session or a sub-agent on "
        "a `row/<ROW-ID>` branch; the operator merges it"
    ]


def policy_commit_violations(
    commit: str,
    parents: list[str],
    subject: str,
    body: str,
    paths: list[str],
    merged_messages: tuple[str, ...] = (),
) -> list[str]:
    """Enforce POL-PR-REQUIRED for every tracked repository change."""

    governed = sorted(path for path in paths if path)
    if not governed or arrives_via_row_pr(parents, subject, body, merged_messages):
        return []
    preview = ", ".join(governed[:4])
    if len(governed) > 4:
        preview += f", ... (+{len(governed) - 4})"
    return [
        f"{commit}: repository change ({preview}) reached main without a reviewed "
        "row/* PR"
    ]


def commit_paths(commit: str) -> list[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        out = git("diff", "--name-only", parents[0], commit)
    else:
        out = git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
    return [line for line in out.splitlines() if line]


def inspect(commit: str) -> list[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    subject = git("log", "-1", "--format=%s", commit)
    body = git("log", "-1", "--format=%b", commit)
    short = git("rev-parse", "--short", commit)
    # The messages of the branches this commit MERGES IN (parents[1:]), for the
    # synthetic-PR-merge case in arrives_via_row_pr.
    merged = tuple(git("log", "-1", "--format=%s%n%b", parent) for parent in parents[1:])
    return policy_commit_violations(
        short, parents, subject, body, commit_paths(commit), merged
    )


def enforced(commit: str) -> bool:
    """True when this commit is after the cutover."""
    if ROLE_DISCIPLINE_SINCE is None:
        return False
    try:
        git("merge-base", "--is-ancestor", ROLE_DISCIPLINE_SINCE, commit)
        return True
    except subprocess.CalledProcessError:
        return False


def has_reached_main(commit: str) -> bool:
    """Whether the commit is already contained by the local main ref."""

    try:
        git("merge-base", "--is-ancestor", commit, "refs/heads/main")
        return True
    except subprocess.CalledProcessError:
        pass
    try:
        branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    except subprocess.CalledProcessError:
        # CI checks out main and synthetic PR merges detached. A synthetic merge
        # already passes arrives_via_row_pr; a violating detached commit must
        # fail closed as landed/integration history.
        return True
    # An unmerged row head is pending PR disposition, not main history yet.
    return not branch.startswith("row/")


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    return [c for c in git("rev-list", "--reverse", f"{base}..{head}").splitlines() if c]


def pending_pr_commits(base: str, head: str, pending_pr_head: str) -> frozenset[str]:
    """Return the exact unmerged PR range named by a trusted PR event."""

    if re.fullmatch(r"[0-9a-f]{40}", pending_pr_head) is None:
        raise ValueError("--pending-pr-head must be one lowercase 40-byte commit SHA")
    try:
        resolved_head = git("rev-parse", "--verify", f"{head}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise ValueError("--head must resolve to one commit") from exc
    if resolved_head != pending_pr_head:
        raise ValueError("--pending-pr-head must exactly match --head")
    return frozenset(commits_in_range(base, resolved_head))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", help="check one commit")
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    parser.add_argument(
        "--pending-pr-head",
        help="exact PR-event head SHA; marks only base..head as not yet landed",
    )
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    if args.pending_pr_head is not None and args.base is None:
        parser.error("--pending-pr-head requires --base and --head")

    if args.base is not None:
        commits = commits_in_range(args.base, args.head)
    elif args.commit:
        commits = [args.commit]
    else:
        commits = ["HEAD"]

    try:
        pending = (
            pending_pr_commits(args.base, args.head, args.pending_pr_head)
            if args.pending_pr_head is not None
            else frozenset()
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures, reported = [], []
    for commit in commits:
        for problem in inspect(commit):
            # A row head has not reached main yet, so it is reportable pending
            # integration rather than a false claim that unmerged work already
            # violated the arrival rule. Main history and recognized synthetic
            # PR merges remain strict.
            landed = commit not in pending and enforced(commit) and has_reached_main(commit)
            (failures if landed else reported).append(problem)

    for problem in reported:
        print(f"REPORT: {problem}", file=sys.stderr)
    for problem in failures:
        print(f"ERROR: {problem}", file=sys.stderr)

    if failures:
        return 1
    if ROLE_DISCIPLINE_SINCE is None:
        print(
            "OK: role discipline is REPORT-ONLY "
            f"({len(reported)} commit(s) would fail once ROLE_DISCIPLINE_SINCE "
            "names the cutover commit)."
        )
    else:
        print("OK: feature code on main arrived through reviewed row/* PRs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
