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
Helpers take their own worktree by construction, so the shared case is the
operator's primary checkout, where one role is the correct answer anyway. See
.agents/specs/session-onboarding.md, "Correction: a role keys on the WORKTREE,
not the session".

The operator lock lives in the git COMMON dir, not the working tree: it is
shared by every worktree of the repo (the correct scope for "one operator per
repo") and can never be committed by accident. Its ownership keys on the
worktree too, so the operator survives the same call boundary while a second
worktree is still refused.

    scripts/agent-role.py show                  # resolve; exit 3 if undeclared
    scripts/agent-role.py claim operator
    scripts/agent-role.py claim helper --row ENG-FOO
    scripts/agent-role.py claim read-only          # declares no claim at all
    scripts/agent-role.py claim helper --row ENG-FOO --headless
    scripts/agent-role.py heartbeat
    scripts/agent-role.py release
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# read-only is a declared ABSENCE of claim, not a third role: it takes no lock
# and creates no worktree. Without it, a session that only reads must either
# take the repo-wide operator lock or create a throwaway worktree, and faced
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


# A lock older than this with no heartbeat is stale: a crashed operator must not
# block everyone forever. Breaking one is always logged, never silent.
LOCK_TTL_SECONDS = 2 * 60 * 60

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


def lock_path() -> Path:
    """Shared by every worktree of this repo, and never inside the work tree."""
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "vllm-cpp-operator.lock"


def lock_is_ours(lock: dict | None) -> bool:
    """Does the operator lock belong to THIS worktree?

    Ownership follows the same identity as the role. A lock written before the
    2026-08-06 correction carries no worktree, so it falls back to its recorded
    session: that keeps such a lock releasable and re-claimable by the session
    that took it instead of wedging the repo until the TTL expires.

    That fallback is WIDER than the worktree rule -- another worktree running
    under the same session id also reads it as ours -- so `claim` rewrites the
    record instead of passing, stamping the worktree on and healing the
    ambiguity the first time it is used. Delete this branch once no
    pre-correction lock can exist.
    """
    if not lock:
        return False
    if lock.get("worktree"):
        return lock["worktree"] == worktree_id()
    return lock.get("session") == session_id()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def lock_is_stale(record: dict) -> bool:
    beat = record.get("heartbeat", record.get("claimed_at", 0))
    return (time.time() - float(beat)) > LOCK_TTL_SECONDS


def current_branch() -> str:
    try:
        return git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return ""


def resolve() -> dict:
    """Return the resolved role for THIS WORKTREE, or {'role': None, ...}."""
    me = session_id()
    marker = read_json(marker_path())
    lock = read_json(lock_path())

    # The marker is keyed on the WORKTREE, which is where it lives, and NOT on
    # the session: a session id is not stable across tool calls, so requiring it
    # made a declared role invisible one call later. `session` is carried
    # through as `declared_by` provenance and gates nothing.
    #
    # DECLARABLE, not ROLES: read-only is declarable but holds no lock, so it
    # must resolve here while still never counting as "may write".
    if marker and marker.get("role") in DECLARABLE:
        declared = marker["role"]
        if declared == "operator" and not lock_is_ours(lock):
            return {
                "role": None,
                "session": me,
                "mode": "interactive",
                "reason": "operator marker without a held lock; re-claim",
                # Carried on THIS path too, and for the same reason the
                # undeclared path below carries it: a worktree keyed lock made
                # this branch reachable from a LIVE RIVAL, not only from a
                # self-lost lock. Without the key, "locked out by another
                # operator" renders identically to "never declared", and every
                # reader -- render_probe's NOTE, cmd_show's note, the probe's
                # JSON -- tells the session to `claim operator`, which exits 1.
                "operator_held_by_other": bool(lock and not lock_is_stale(lock)),
                "branch": current_branch(),
            }
        return {
            "role": declared,
            "row": marker.get("row"),
            "session": me,
            "declared_by": marker.get("session"),
            "branch": current_branch(),
            "mode": mode_from_marker(marker),
            "reason": "declared",
        }

    # Not declared here. Report what else is going on so the caller can decide.
    held_by_other = bool(lock and not lock_is_ours(lock) and not lock_is_stale(lock))
    return {
        "role": None,
        "session": me,
        "branch": current_branch(),
        "mode": "interactive",
        "operator_held_by_other": held_by_other,
        "reason": "undeclared",
    }


def cmd_show(args: argparse.Namespace) -> int:
    state = resolve()
    if args.json:
        print(json.dumps(state))
    elif state["role"]:
        row = f" row={state['row']}" if state.get("row") else ""
        print(f"role={state['role']}{row} session={state['session']} branch={state['branch']}")
    else:
        print(f"role=UNDECLARED session={state['session']} branch={state['branch']}")
        if state.get("operator_held_by_other"):
            print("  note: the operator lock is held by another live session")
    return 0 if state["role"] else UNDECLARED_EXIT


def cmd_claim(args: argparse.Namespace) -> int:
    me = session_id()
    role = args.role
    if role == "helper" and not args.row:
        print("ERROR: a helper claims one row: --row <ROW-ID>", file=sys.stderr)
        return 2

    if role == "operator":
        path = lock_path()
        record = {"session": me, "worktree": worktree_id(),
                  "claimed_at": time.time(), "heartbeat": time.time(),
                  "host": os.uname().nodename, "pid": os.getpid()}
        try:
            # Create-exclusive: a second self-declared operator FAILS here rather
            # than racing on main, which is the whole point of the lock.
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle)
        except FileExistsError:
            existing = read_json(path) or {}
            if lock_is_ours(existing):
                # Already this worktree's. REWRITE rather than pass: it renews
                # the heartbeat (ownership keyed on the worktree means a live
                # operator re-claims more often than it beats, and a lock that
                # ages out while its owner is alive gets broken by someone
                # else), and it stamps `worktree` onto a pre-correction lock,
                # which is what stops the legacy session fallback below from
                # leaving two worktrees resolving as operator at once.
                path.write_text(json.dumps(record), encoding="utf-8")
            elif lock_is_stale(existing):
                age = int(time.time() - float(existing.get("heartbeat", 0)))
                print(
                    f"NOTE: breaking a STALE operator lock held by "
                    f"{existing.get('session')} on {existing.get('host')} "
                    f"({age}s without heartbeat, TTL {LOCK_TTL_SECONDS}s)",
                    file=sys.stderr,
                )
                path.write_text(json.dumps(record), encoding="utf-8")
            else:
                print(
                    f"ERROR: the operator role is already held by session "
                    f"{existing.get('session')} on {existing.get('host')}. "
                    "This session cannot be the operator; take the helper role "
                    "instead (scripts/agent-role.py claim helper --row <ROW-ID>).",
                    file=sys.stderr,
                )
                return 1
    else:
        # Downgrading OUT of the operator role must not orphan the lock.
        # `claim read-only` ("just looking") is the exact command an operator
        # types next, and leaving the lock behind is worse than leaving no role
        # at all: heartbeat answers "not the operator; nothing to heartbeat", so
        # nothing renews it, while a second worktree's `claim operator` is
        # refused for the full TTL by a session that holds no role. Releasing
        # here is the same ownership test `release` uses.
        if lock_is_ours(read_json(lock_path())):
            lock_path().unlink(missing_ok=True)
            print(f"released the operator lock (this worktree is now {role})")

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
    return 0


def cmd_heartbeat(_: argparse.Namespace) -> int:
    state = resolve()
    if state["role"] != "operator":
        print("not the operator; nothing to heartbeat")
        return 0
    path = lock_path()
    record = read_json(path) or {}
    record["heartbeat"] = time.time()
    path.write_text(json.dumps(record), encoding="utf-8")
    print("heartbeat updated")
    return 0


def cmd_release(_: argparse.Namespace) -> int:
    lock = read_json(lock_path())
    if lock_is_ours(lock):
        lock_path().unlink(missing_ok=True)
        print("released the operator lock")
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

    sub.add_parser("heartbeat", help="keep the operator lock alive").set_defaults(
        func=cmd_heartbeat
    )
    sub.add_parser("release", help="drop the role and any held lock").set_defaults(
        func=cmd_release
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
