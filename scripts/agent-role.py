#!/usr/bin/env python3
"""Declare, materialize and re-derive an agent session's role. (W0)

The role CANNOT be derived at session start: the common case is one operator and
several helpers all launched from the SAME checkout, indistinguishable until a
role has already been taken. So the role is DECLARED, immediately MATERIALIZED
into a fact, and only then re-derived. See
.agents/specs/operator-helper-protocol.md.

A role keys on the **worktree**, never on the session (user-directed correction,
2026-08-06):

* **worktree** - `git rev-parse --absolute-git-dir`, which is per-worktree
  (`.git/worktrees/<name>`), so a materialized helper is distinguishable from
  the primary checkout without any bookkeeping. The marker lives inside it, so
  one worktree is one role and that role survives a new shell, a new process
  and a lost environment.
* **session** - `VLLM_CPP_AGENT_SESSION` when set, else the parent process id.
  Recorded as PROVENANCE only, and nothing gates on it. An earlier version of
  this file called it "stable across tool calls within a session (measured)".
  That was DISPROVEN: at least one real harness gives every tool call a fresh
  shell and does not persist environment variables, so a role claimed in one
  call resolved as UNDECLARED in the next and `agent-preflight.sh` exited 1 --
  making a default-on role gate unpassable rather than strict, which is how a
  gate teaches people to disable it.

The accepted cost is explicit: two sessions sharing one checkout share a role.
Every unit of work takes its own worktree, so that case is rare and it is the
right trade. See .agents/specs/session-onboarding.md, "Correction: a role keys
on the WORKTREE, not the session".

The coordinator RECORDS live in the git COMMON dir, not the working tree: they
are shared by every worktree of the repo (the right scope for "who is
coordinating where") and can never be committed by accident.

They are a record and NEVER a refusal (user-directed, issue #285,
.agents/specs/operator-record.md). This file used to argue for one exclusive
operator per repo because "the shared case is the operator's primary checkout,
where one role is the correct answer anyway". That premise is gone: AGENTS.md
now requires EVERY unit of work to take its own worktree and to reach `main`
from a task branch, so there is no shared checkout to protect.

What an operator is, is a COORDINATOR. Its powers are merging PRs and
dispatching sub-agents into separate worktrees; it never rewrites shared
history, and `main` is never force-pushed. A plain `git push` therefore refuses
any non-fast-forward, so git itself is the interlock and concurrent
coordinators serialise on it -- the loser fetches, re-merges, re-gates and
pushes again. A JSON file in `.git/` never provided that guarantee and could
not. What it did provide was two hours of blocked coordination every time a
session died mid-flight, with no remedy but hand-deleting a file.

So the representation is one record per worktree,
`<git-common-dir>/vllm-cpp-operators/<sha256(worktree)[:16]>.json`, rather than
one shared file. A writer only ever touches the path derived from its OWN
worktree -- the identity ownership already keys on -- so two claimants racing
write two different paths and neither can lose the other's record. Each publish
is `write temp + os.replace` in the same directory, which is atomic on POSIX, so
a reader sees the old record or the new one, never half of one. There is no
read-modify-write of a shared file anywhere here, which is exactly what a single
JSON array or an append log would have needed to release or prune.

    scripts/agent-role.py show                  # resolve; exit 3 if undeclared
    scripts/agent-role.py claim operator        # records; never refused
    scripts/agent-role.py claim helper --row ENG-FOO
    scripts/agent-role.py claim read-only          # declares no claim at all
    scripts/agent-role.py claim helper --row ENG-FOO --headless
    scripts/agent-role.py heartbeat
    scripts/agent-role.py release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# read-only is a declared ABSENCE of claim, not a third role: it records nothing
# and creates no worktree. Without it, a session that only reads must either
# record itself as a coordinator or create a throwaway worktree, and faced
# with that people reach for --no-require-role until the gate means nothing.
#
# CLAIMABLE_ROLES is the vocabulary a "may this session write?" test SHOULD key
# on, and it is kept at exactly two so that such a test stays correct when one
# is written. Today it has no consumer outside this file and its suite: the one
# write refusal that exists is `agent-preflight.sh --staged`, which matches on
# the rendered `role=read-only` line. Nothing else refuses a read-only session
# (see AGENTS.md and .agents/specs/session-onboarding.md, which say so).
CLAIMABLE_ROLES = ("operator", "helper")
DECLARABLE = (*CLAIMABLE_ROLES, "read-only")
ROLES = CLAIMABLE_ROLES  # alias kept as the "may write" name for future callers


def mode_from_marker(marker: dict) -> str:
    """Interactive unless headless was DECLARED. Never inferred."""
    return "headless" if marker.get("mode") == "headless" else "interactive"


# A record older than this with no heartbeat is stale: it describes a session
# that stopped coordinating, and showing it would make the record lie. The value
# is unchanged from when this was a lock; only the consequence changed. A stale
# record is filtered out of every display, and ANOTHER worktree's is unlinked by
# the next `claim`; OURS is replaced by it and never unlinked -- see
# `prune_stale_records`' `keep_canonical`, which exists because our own record is
# stale on any ordinary re-claim and unlinking it before republishing it is the
# one window that leaves this worktree an operator marker with no record. A stale
# record can no longer refuse anybody either, so breaking one is not an event
# worth announcing any more.
RECORD_TTL_SECONDS = 2 * 60 * 60

# One directory of per-worktree records. See the module docstring for why this
# shape and not one shared file.
RECORDS_DIRNAME = "vllm-cpp-operators"

# The pre-#285 single-file lock. Still READ, so a session that claimed operator
# before this change keeps resolving instead of turning UNDECLARED mid-flight
# and failing its next preflight; its next claim or release removes it. Delete
# this once no pre-#285 file can exist.
LEGACY_RECORD_NAME = "vllm-cpp-operator.lock"

UNDECLARED_EXIT = 3


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def session_id() -> str:
    """Provenance only: who declared this. NOT stable across tool calls."""
    explicit = os.environ.get("VLLM_CPP_AGENT_SESSION")
    return explicit if explicit else f"ppid:{os.getppid()}"


def worktree_id() -> str:
    """The identity a role keys on. One worktree is one role."""
    return git("rev-parse", "--absolute-git-dir")


def marker_path() -> Path:
    """Per-worktree, so a materialized helper carries its own role."""
    return Path(worktree_id()) / "vllm-cpp-agent-role"


def common_dir() -> Path:
    """Shared by every worktree of this repo, and never inside a work tree."""
    return Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))


def records_dir() -> Path:
    return common_dir() / RECORDS_DIRNAME


def record_path(worktree: str | None = None) -> Path:
    """This worktree's OWN record file, and no one else's.

    The name is derived from the worktree, which is the identity ownership keys
    on, so two coordinators claiming at the same instant address two different
    paths. That -- not a lock -- is what makes concurrent claims safe. The digest
    keeps a path that may contain separators or exotic characters usable as one
    filename; the record itself carries the readable worktree path.
    """
    key = hashlib.sha256((worktree or worktree_id()).encode("utf-8")).hexdigest()[:16]
    return records_dir() / f"{key}.json"


def legacy_record_path() -> Path:
    return common_dir() / LEGACY_RECORD_NAME


def record_is_ours(record: dict | None) -> bool:
    """Does this coordinator record belong to THIS worktree?

    Ownership follows the same identity as the role. A record written before the
    2026-08-06 correction carries no worktree, so it falls back to its recorded
    session: that keeps such a record removable by the session that wrote it
    instead of lingering in everyone's display until the TTL expires.
    """
    if not record:
        return False
    if record.get("worktree"):
        return record["worktree"] == worktree_id()
    return record.get("session") == session_id()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_is_stale(record: dict) -> bool:
    beat = record.get("heartbeat", record.get("claimed_at", 0))
    try:
        return (time.time() - float(beat)) > RECORD_TTL_SECONDS
    except (TypeError, ValueError):
        # A record whose heartbeat is unreadable describes nothing usable. Treat
        # it as stale so it is pruned, never as fresh so it lingers forever.
        return True


def read_records() -> list[dict]:
    """Every readable coordinator record, each carrying the file it came from.

    A record being unlinked by its owner WHILE this reads is ordinary, not an
    error: read_json answers None and it is simply not listed. Nothing here
    writes, because `show` and `resolve` run inside agent-preflight.sh, which
    documents itself as never writing anything.
    """
    found: list[dict] = []
    try:
        entries = sorted(records_dir().iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if entry.suffix != ".json":
            continue
        record = read_json(entry)
        if record:
            found.append({**record, "path": str(entry)})
    legacy = read_json(legacy_record_path())
    if legacy:
        found.append({**legacy, "path": str(legacy_record_path()), "legacy": True})
    return found


def our_record() -> dict | None:
    return next((record for record in read_records() if record_is_ours(record)), None)


def peer_records() -> list[dict]:
    """The live coordinators that are NOT this worktree. The display's subject."""
    return [
        record
        for record in read_records()
        if not record_is_ours(record) and not record_is_stale(record)
    ]


def write_our_record(claimed_at: float | None = None) -> dict:
    """Publish THIS worktree's record atomically, and heal any legacy file.

    temp + os.replace inside the same directory: a concurrent reader sees the
    previous record or this one and never a partial write. `claimed_at` is
    carried over by `heartbeat` and reset by `claim`.
    """
    now = time.time()
    record = {
        "session": session_id(),
        "worktree": worktree_id(),
        "claimed_at": now if claimed_at is None else claimed_at,
        "heartbeat": now,
        "host": os.uname().nodename,
        "pid": os.getpid(),
    }
    target = record_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, target)
    return record


def drop_our_record(keep_canonical: bool = False) -> bool:
    """Remove THIS worktree's record -- including a legacy file it owns.

    `keep_canonical` leaves our own `record_path()` alone and drops only the
    other files this worktree owns (in practice the pre-#285 single-file lock).
    The claim path needs it: unlinking our canonical record and re-creating it
    is an unlink-then-create where the design promises replace, and a session
    killed inside that window leaves an operator marker with NO record -- the
    exact state issue #285 exists to remove, and the one state that still
    refuses to resolve. `write_our_record`'s `os.replace` publishes over the
    path without ever removing it, so nothing has to be unlinked first.
    """
    canonical = record_path() if keep_canonical else None
    removed = False
    for record in read_records():
        if not record_is_ours(record):
            continue
        path = Path(record["path"])
        if canonical is not None and path == canonical:
            continue
        try:
            path.unlink(missing_ok=True)
            removed = True
        except OSError:
            pass
    return removed


def prune_stale_records(keep_canonical: bool = False) -> list[dict]:
    """Drop records past the TTL. Called from `claim`, which already writes.

    `keep_canonical` skips THIS worktree's own `record_path()`. The claim path
    needs it, and only the claim path calls this: our own record is stale on
    every ordinary re-claim -- the TTL is two hours and a session re-claims at
    the top of its next tool call -- and pruning it is an unlink-then-create
    that this path has already been fixed once to avoid (see `drop_our_record`).
    It is safe because `write_our_record` republishes that exact path
    immediately afterwards, and it is not a leak: `resolve` matches our own
    record by ownership with no staleness filter, so an aged own record was
    never being displayed as a live coordinator anyway.

    For every OTHER record this targets a PATH, not the inode it read, so a peer
    that was silent for more than the TTL and republishes inside this window
    loses the record it just wrote. Left as is deliberately: the loser's remedy
    is `claim operator`, which is never refused and which every session runs at
    the top of its next tool call, so the cost is one display cycle. Re-reading
    before the unlink would narrow the window without closing it and would add a
    branch no test can reach. That declination covers peers ONLY -- our own
    record is not exposed to it, because `keep_canonical` never unlinks it.
    """
    canonical = record_path() if keep_canonical else None
    pruned = []
    for record in read_records():
        if canonical is not None and Path(record["path"]) == canonical:
            continue
        if record_is_stale(record):
            try:
                Path(record["path"]).unlink(missing_ok=True)
                pruned.append(record)
            except OSError:
                pass
    return pruned


def describe_record(record: dict) -> str:
    """Who, which worktree, and how long since the heartbeat -- ASCII only."""
    try:
        beat = float(record.get("heartbeat", record.get("claimed_at", 0)))
    except (TypeError, ValueError):
        beat = 0.0
    age = int(max(0.0, time.time() - beat))
    return (
        f"{record.get('worktree') or 'unknown worktree'} - "
        f"session {record.get('session') or 'unknown'} "
        f"on {record.get('host') or 'unknown host'} "
        f"(pid {record.get('pid', '?')}, last heartbeat {age}s ago)"
    )


def render_peers(peers: list[dict]) -> list[str]:
    """The whole point of keeping the file: who else is coordinating, and where.

    Empty when nobody else is, so a solo session reads no conflict where there
    is none.
    """
    if not peers:
        return []
    return [f"other coordinators recorded: {len(peers)}"] + [
        f"  - {describe_record(record)}" for record in peers
    ]


def current_branch() -> str:
    try:
        return git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return ""


def resolve() -> dict:
    """Return the resolved role for THIS WORKTREE, or {'role': None, ...}."""
    me = session_id()
    marker = read_json(marker_path())
    # One read of the records, split two ways: ours decides the role, the live
    # rest are reported as peers. Peers never gate anything -- they are
    # information, and every path carries them so any session can see who is
    # coordinating where.
    records = read_records()
    mine = next((record for record in records if record_is_ours(record)), None)
    peers = [
        record
        for record in records
        if not record_is_ours(record) and not record_is_stale(record)
    ]

    # The marker is keyed on the WORKTREE, which is where it lives, and NOT on
    # the session: a session id is not stable across tool calls, so requiring it
    # made a declared role invisible one call later. `session` is carried
    # through as `declared_by` provenance and gates nothing.
    #
    # DECLARABLE, not ROLES: read-only is declarable but records nothing, so it
    # must resolve here while still never counting as "may write".
    if marker and marker.get("role") in DECLARABLE:
        declared = marker["role"]
        if declared == "operator" and mine is None:
            return {
                "role": None,
                "session": me,
                "mode": "interactive",
                # The one path that still refuses to resolve, and it is
                # REACHABLE: a host cleanup deleted exactly this file on
                # 2026-08-10. It must stay distinguishable from "never
                # declared", because the remedy is named here and now always
                # works -- `claim operator` records and succeeds whoever else
                # is recorded.
                "reason": "operator marker without a coordinator record; re-claim",
                "operator_peers": peers,
                "branch": current_branch(),
            }
        return {
            "role": declared,
            "row": marker.get("row"),
            "session": me,
            "declared_by": marker.get("session"),
            "branch": current_branch(),
            "mode": mode_from_marker(marker),
            "operator_peers": peers,
            "reason": "declared",
        }

    # Not declared here. Report what else is going on so the caller can decide.
    return {
        "role": None,
        "session": me,
        "branch": current_branch(),
        "mode": "interactive",
        "operator_peers": peers,
        "reason": "undeclared",
    }


def cmd_show(args: argparse.Namespace) -> int:
    state = resolve()
    if args.json:
        print(json.dumps(state))
        return 0 if state["role"] else UNDECLARED_EXIT
    if state["role"]:
        row = f" row={state['row']}" if state.get("row") else ""
        print(f"role={state['role']}{row} session={state['session']} branch={state['branch']}")
    else:
        print(f"role=UNDECLARED session={state['session']} branch={state['branch']}")
        if state.get("reason", "").startswith("operator marker"):
            print("  note: this worktree's coordinator record is gone; "
                  "re-run `claim operator` (it is never refused)")
    # Printed for EVERY role, declared or not: any session may need to know who
    # else is coordinating, and this is the only place that says so.
    for line in render_peers(state.get("operator_peers") or []):
        print(line)
    return 0 if state["role"] else UNDECLARED_EXIT


def cmd_claim(args: argparse.Namespace) -> int:
    me = session_id()
    role = args.role
    if role == "helper" and not args.row:
        print("ERROR: a helper claims one row: --row <ROW-ID>", file=sys.stderr)
        return 2

    peers: list[dict] = []
    if role == "operator":
        # No refusal exists here any more (issue #285). A second coordinator is
        # RECORDED: it merges PRs and dispatches sub-agents into worktrees, and
        # since `main` is never force-pushed, git's non-fast-forward refusal is
        # the interlock this file was pretending to be.
        #
        # Prune first, then read the peers, so a record left by a session that
        # died mid-flight neither lingers in the display nor is reported as a
        # live coordinator. This is a write path, which is why pruning happens
        # here and never in `show`.
        #
        # keep_canonical: the prune is about OTHER sessions. Our own record is
        # stale on any ordinary re-claim, and unlinking it here would put back
        # exactly the window `drop_our_record(keep_canonical=True)` closes
        # below -- an operator marker with no record, manufactured out of a
        # state that resolved fine a moment earlier.
        prune_stale_records(keep_canonical=True)
        peers = peer_records()
        # Rewriting our own record is the renewal path too: a live coordinator
        # re-claims more often than it beats, and a record that ages out while
        # its owner is alive disappears from everyone else's display. It also
        # replaces a legacy single-file lock this worktree owned, so that file
        # cannot linger and be counted twice.
        #
        # keep_canonical: our own `record_path()` is left alone here, so only
        # the other files this worktree owns (in practice the pre-#285 legacy
        # lock) are unlinked. Together with the scoped prune above, nothing on
        # this path removes our record: it is REPLACED by the publish below, so
        # a re-claim has no window in which this worktree has an operator marker
        # and no record -- at any age of the record it started from.
        drop_our_record(keep_canonical=True)
        write_our_record()
    else:
        # Downgrading OUT of the operator role must not orphan the record.
        # `claim read-only` ("just looking") is the exact command a coordinator
        # types next, and leaving the record behind makes it lie: heartbeat
        # answers "not the operator; nothing to heartbeat", so nothing renews
        # it, and everyone else sees a coordinator that stopped coordinating
        # until the TTL. Removing here is the same ownership test `release` uses.
        if drop_our_record():
            print(f"removed this worktree's coordinator record (now {role})")

    marker_path().write_text(
        json.dumps({
            "role": role,
            "row": args.row,
            "session": me,
            # Declared with the role, so it is a fact rather than a guess: no
            # later code has to infer headless from the hour or from silence.
            "mode": "headless" if args.headless else "interactive",
            "at": time.time(),
        }),
        encoding="utf-8",
    )
    print(f"claimed role={role}" + (f" row={args.row}" if args.row else ""))
    # Information, never an obstacle: several coordinators may run at once.
    for line in render_peers(peers):
        print(line)
    return 0


def cmd_heartbeat(_: argparse.Namespace) -> int:
    state = resolve()
    if state["role"] != "operator":
        print("not the operator; nothing to heartbeat")
        return 0
    existing = our_record() or {}
    # Only ever this worktree's own file, and `claimed_at` is carried over so
    # the record still says when this coordinator started.
    write_our_record(claimed_at=existing.get("claimed_at"))
    print("heartbeat updated")
    return 0


def cmd_release(_: argparse.Namespace) -> int:
    if drop_our_record():
        print("removed this worktree's coordinator record")
    marker_path().unlink(missing_ok=True)
    print("released the role marker")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="resolve this session's role")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show)

    claim = sub.add_parser("claim", help="declare and materialize a role")
    claim.add_argument("role", choices=DECLARABLE)
    claim.add_argument("--row", help="the row ID a helper claims")
    claim.add_argument(
        "--headless",
        action="store_true",
        help="unattended run: decide and record rather than ask (never inferred)",
    )
    claim.set_defaults(func=cmd_claim)

    sub.add_parser("heartbeat", help="keep this coordinator record fresh").set_defaults(
        func=cmd_heartbeat
    )
    sub.add_parser("release", help="drop the role and this worktree's record").set_defaults(
        func=cmd_release
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
