#!/usr/bin/env python3
"""Keep .agents/NOW.md a short, current, one-Read resume surface.

The canonical state record is a bounded manifest, sharded CSV indexes, and
immutable event evidence. NOW.md is the live projection of that record.

NOW.md is the fix: the single small file a cold session reads first to become
productive. It is a SNAPSHOT, never a log -- it is rewritten in place, and the
detail it summarises stays in the append-only record.

Two obligations are enforced:
  * structure and budget, so it cannot decay into another status log;
  * freshness, so it cannot silently go stale. A newly indexed event refreshes
    NOW.md when its outcome changes the live project position. Migration-only,
    forensic corrections, and already-terminal forensic events are exempt.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.state_record import EVENT_HEADER, Event


NOW = ROOT / ".agents/NOW.md"
NOW_PATH = ".agents/NOW.md"
STATE_INDEX_PREFIX = ".agents/state-index/"
TERMINAL_OUTCOMES = frozenset({"landed", "closed"})
TERMINAL_NEXT_ACTIONS = frozenset(
    {"", "-", "—", "none", "no further action", "terminal"}
)
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Budgets. NOW.md exists to be read in full, every session, by every agent. The
# moment it stops fitting in one screenful of attention it has become the thing
# it was meant to replace.
MAX_LINES = 100
MAX_CHARS = 6000
MAX_ENTRY_CHARS = 400

REQUIRED_HEADINGS = (
    "live claims",
    "current gate",
    "next actions",
)

STAMP = re.compile(r"^<!--\s*now-updated:\s*(\d{4}-\d{2}-\d{2})\s*-->$", re.MULTILINE)


def structure_errors(text: str) -> list[str]:
    """Return budget/shape problems with the NOW digest."""
    errors: list[str] = []

    if not STAMP.search(text):
        errors.append(
            "missing the freshness stamp <!-- now-updated: YYYY-MM-DD -->; it "
            "records when this snapshot was last known true"
        )

    lowered = text.lower()
    for heading in REQUIRED_HEADINGS:
        if f"## {heading}" not in lowered:
            errors.append(
                f"missing the '## {heading}' section; a cold session needs all "
                f"of {', '.join(REQUIRED_HEADINGS)} to resume without reading "
                "the full record"
            )

    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        errors.append(
            f"is {len(lines)} lines, over the {MAX_LINES}-line budget; move "
            "detail to structured state evidence and keep only the live position here"
        )
    if len(text) > MAX_CHARS:
        errors.append(
            f"is {len(text)} characters, over the {MAX_CHARS}-character budget; "
            "this is a digest, not a status log"
        )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "|")) and len(stripped) > MAX_ENTRY_CHARS:
            errors.append(
                f"an entry is {len(stripped)} characters, over the "
                f"{MAX_ENTRY_CHARS}-character budget: {stripped[:60]!r}...; "
                "link the spec or state entry instead of inlining the narrative"
            )

    return errors


def _state_index_paths(paths: set[str]) -> list[str]:
    return sorted(
        path
        for path in paths
        if path.startswith(STATE_INDEX_PREFIX) and path.endswith(".csv")
    )


def _normalize_next_action(value: str) -> str:
    return value.strip().rstrip(".").strip().lower()


def event_requires_refresh(event: Event, now_text: str) -> bool:
    """Whether a newly indexed event changes the live NOW projection."""
    if event.kind == "legacy_import":
        return False
    subjects = [subject.lower() for subject in event.subject_ids.split(";") if subject]
    subject_is_live = any(subject in now_text.lower() for subject in subjects)
    has_follow_up = _normalize_next_action(event.next_action) not in TERMINAL_NEXT_ACTIONS
    if event.kind == "correction" and event.outcome == "superseded":
        return subject_is_live or has_follow_up
    if event.outcome not in TERMINAL_OUTCOMES:
        return True
    return subject_is_live or has_follow_up


def freshness_errors(
    paths: set[str],
    appended_events: list[Event] | tuple[Event, ...] = (),
    *,
    now_text: str | None = None,
) -> list[str]:
    """Return staleness problems for one atomic structured-record change."""
    triggers = _state_index_paths(paths)
    if not triggers or not appended_events:
        return []
    digest = NOW.read_text(encoding="utf-8") if now_text is None else now_text
    live_events = [event for event in appended_events if event_requires_refresh(event, digest)]
    if live_events and NOW_PATH not in paths:
        event_ids = ", ".join(event.event_id for event in live_events)
        return [
            f"{', '.join(triggers)} appended live event(s) {event_ids} but "
            f"{NOW_PATH} did not change; refresh the digest in the same change "
            "(live claims, current gate, next actions, stamp)"
        ]
    return []


def _events_from_csv(text: str) -> list[Event]:
    if not text:
        return []
    reader = csv.reader(io.StringIO(text), strict=True)
    header = tuple(next(reader, ()))
    if header != EVENT_HEADER:
        raise ValueError(f"expected state-index header {EVENT_HEADER!r}, found {header!r}")
    return [Event(*row) for row in reader if row]


def _git_blob(revision: str, path: str) -> str:
    try:
        return git("show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return ""


def appended_events(paths: set[str], before: str, after: str) -> list[Event]:
    """Return rows introduced between two trees (``:`` means the staged index)."""
    appended: list[Event] = []
    for path in _state_index_paths(paths):
        old_events = _events_from_csv(_git_blob(before, path))
        old_ids = {event.event_id for event in old_events}
        appended.extend(
            event
            for event in _events_from_csv(_git_blob(after, path))
            if event.event_id not in old_ids
        )
    return appended


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def commit_paths(commit: str) -> set[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        output = git("diff", "--name-only", parents[0], commit)
    else:
        output = git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit
        )
    return {line for line in output.splitlines() if line}


def commit_events(commit: str, paths: set[str]) -> list[Event]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    before = parents[0] if parents else EMPTY_TREE
    return appended_events(paths, before, commit)


def now_at(revision: str) -> str:
    return _git_blob(revision, NOW_PATH) or NOW.read_text(encoding="utf-8")


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    output = git("rev-list", "--reverse", "--no-merges", f"{base}..{head}")
    return [line for line in output.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--commit", default=None, help="check one commit")
    source.add_argument(
        "--staged", action="store_true", help="check the current staged change"
    )
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    if args.base is not None and (args.commit is not None or args.staged):
        parser.error("a revision range cannot be combined with --commit/--staged")
    return args


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    if not NOW.exists():
        print(f"ERROR: {NOW_PATH} does not exist", file=sys.stderr)
        return 1

    failures.extend(
        f"{NOW_PATH} {error}"
        for error in structure_errors(NOW.read_text(encoding="utf-8"))
    )

    if args.staged:
        paths = set(git("diff", "--cached", "--name-only").splitlines())
        staged_events = appended_events(paths, "HEAD", "")
        failures.extend(
            f"staged change: {error}"
            for error in freshness_errors(
                paths, staged_events, now_text=now_at("")
            )
        )
    elif args.base is not None:
        for commit in commits_in_range(args.base, args.head):
            short = git("rev-parse", "--short", commit)
            failures.extend(
                f"commit {short}: {error}"
                for error in freshness_errors(
                    commit_paths(commit),
                    commit_events(commit, commit_paths(commit)),
                    now_text=now_at(commit),
                )
            )
    elif args.commit is not None:
        short = git("rev-parse", "--short", args.commit)
        failures.extend(
            f"commit {short}: {error}"
            for error in freshness_errors(
                commit_paths(args.commit),
                commit_events(args.commit, commit_paths(args.commit)),
                now_text=now_at(args.commit),
            )
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "NOW.md is the one-Read resume surface: the live claims, the gate "
            "being chased, and the next actions, rewritten in place. Detail "
            "belongs in structured state evidence and the area matrices.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {NOW_PATH} is a current, in-budget resume digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
