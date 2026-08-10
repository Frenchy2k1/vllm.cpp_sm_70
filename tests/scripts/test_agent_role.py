#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-role.py (W0) and
scripts/check-role-discipline.py (W1).

The behaviours that matter are the ones the protocol rests on: a coordinator
must be RECORDED and never refused (issue #285), a session sharing a checkout
must NOT inherit another session's role, a stale record must be pruned without
blocking anyone, and feature code must not reach main without a row/* PR.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


discipline = _load("role_discipline", "scripts/check-role-discipline.py")
role = _load("agent_role", "scripts/agent-role.py")
ROLE_SCRIPT = ROOT / "scripts/agent-role.py"


def run_role(repo: Path, session: str, *args: str):
    env = dict(os.environ, VLLM_CPP_AGENT_SESSION=session)
    return subprocess.run(
        [sys.executable, str(ROLE_SCRIPT), *args],
        cwd=repo, env=env, capture_output=True, text=True,
    )


class _TempRepo:
    """A throwaway git repo per test. The real checkout is never touched."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "root", "--allow-empty"],
                       cwd=self.repo, check=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def worktree(self, name: str) -> Path:
        """A real second worktree of the throwaway repo.

        A role keys on the worktree, so everything this suite proves --
        coordinator records being per-worktree, and helper isolation -- can only
        be proven with a genuine second worktree rather than a second session id.
        """
        path = self.repo / f".{name}"
        subprocess.run(["git", "worktree", "add", "-q", str(path), "-b", name],
                       cwd=self.repo, check=True, capture_output=True)
        return path

    def common(self) -> Path:
        return Path(subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip())

    def records_dir(self) -> Path:
        """Where coordinator records live: shared by every worktree, never
        inside a work tree, and one file per worktree so no two claimants ever
        write the same path."""
        return self.common() / "vllm-cpp-operators"

    def record_files(self) -> list[Path]:
        directory = self.records_dir()
        return sorted(directory.glob("*.json")) if directory.is_dir() else []

    def records(self) -> list[dict]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.record_files()]

    def record_of(self, worktree: Path) -> dict:
        """The record whose worktree is `worktree`'s git dir. Fails if absent."""
        wanted = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=worktree, text=True).strip()
        for record in self.records():
            if record.get("worktree") == wanted:
                return record
        raise AssertionError(f"no coordinator record for {wanted}: {self.records()}")

    def backdate(self, worktree: Path, seconds: float) -> Path:
        """Age one worktree's record past the TTL, as a crashed session leaves it."""
        wanted = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=worktree, text=True).strip()
        for path in self.record_files():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("worktree") == wanted:
                record["heartbeat"] = time.time() - seconds
                path.write_text(json.dumps(record), encoding="utf-8")
                return path
        raise AssertionError(f"no coordinator record for {wanted}")


class RoleLifecycle(_TempRepo, unittest.TestCase):
    """Exercised against a throwaway repo, never the real one."""

    def test_undeclared_session_exits_3(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)

    def test_claim_then_resolve(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        out = run_role(self.repo, "a", "show")
        self.assertEqual(out.returncode, 0)
        self.assertIn("role=operator", out.stdout)

    def test_a_second_coordinator_is_recorded_not_refused(self) -> None:
        """Issue #285, user-directed: the record must never refuse.

        This test asserted the OPPOSITE until 2026-08-10 -- a second worktree's
        `claim operator` exited 1 with "already held". An operator is a
        coordinator: it merges PRs and dispatches sub-agents into worktrees, and
        it never force-pushes `main`, so git's non-fast-forward refusal is the
        interlock and a JSON file in `.git/` never was one. The refusal cost two
        hours of blocked coordination every time a session died mid-flight.
        """
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        rival = self.worktree("rival")
        second = run_role(rival, "b", "claim", "operator")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotIn("already held", second.stderr)
        # Both are operators, and both are recorded -- neither displaced the other.
        for where in (self.repo, rival):
            self.assertIn("role=operator", run_role(where, "z", "show").stdout)
        self.assertEqual(len(self.record_files()), 2, self.records())

    def test_another_session_in_the_same_worktree_shares_the_role(self) -> None:
        """The accepted cost of keying on the worktree, made explicit.

        This test asserted the opposite until 2026-08-06. The session id was
        measured NOT to be stable across tool calls in a real harness, so
        requiring it made a declared role invisible one call later and turned
        --require-role default-on into an unpassable gate rather than a strict
        one. Isolation is preserved where it is real -- see
        test_helper_marker_does_not_leak_into_another_worktree -- and
        .agents/specs/session-onboarding.md records the trade.
        """
        run_role(self.repo, "a", "claim", "operator")
        other = run_role(self.repo, "b", "show")
        self.assertEqual(other.returncode, 0)
        self.assertIn("role=operator", other.stdout)

    def test_helper_requires_a_row(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "claim", "helper").returncode, 2)
        ok = run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        self.assertEqual(ok.returncode, 0)
        self.assertIn("row=ENG-FOO", run_role(self.repo, "a", "show").stdout)

    def test_release_removes_this_worktrees_record(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        self.assertEqual(len(self.record_files()), 1)
        run_role(self.repo, "a", "release")
        self.assertEqual(self.record_files(), [])
        self.assertEqual(run_role(self.repo, "b", "claim", "operator").returncode, 0)

    def test_operator_marker_without_a_record_does_not_resolve(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        for path in self.record_files():
            path.unlink()
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)


class WorktreeKeyedRole(_TempRepo, unittest.TestCase):
    """A role keys on the WORKTREE, not the session (user-directed 2026-08-06).

    `.agents/specs/session-onboarding.md`, "Correction: a role keys on the
    WORKTREE, not the session". Every test here dies if `resolve()` goes back to
    comparing `marker['session']` to the current process.
    """

    def test_helper_role_survives_a_new_session_id(self) -> None:
        # THE regression this correction exists to prevent: claim in one tool
        # call, resolve in the next, where the parent pid has already changed.
        self.assertEqual(
            run_role(self.repo, "call-1", "claim", "helper", "--row", "ENG-FOO").returncode,
            0,
        )
        later = run_role(self.repo, "call-2-different-pid", "show")
        self.assertEqual(later.returncode, 0)
        self.assertIn("role=helper", later.stdout)
        self.assertIn("row=ENG-FOO", later.stdout)

    def test_operator_role_survives_a_new_session_id(self) -> None:
        # The lock is what makes an operator an operator, so lock OWNERSHIP has
        # to key on the worktree too. Key only the marker and the operator alone
        # still dies at the call boundary, which is the failure that matters
        # most: the operator is the role that lands on main.
        run_role(self.repo, "call-1", "claim", "operator")
        later = run_role(self.repo, "call-2-different-pid", "show")
        self.assertEqual(later.returncode, 0)
        self.assertIn("role=operator", later.stdout)

    def test_resolve_ignores_the_marker_session(self) -> None:
        # The same pin driven through resolve() itself rather than the CLI: a
        # marker whose recorded session is NOT this process's must still
        # resolve, and the recorded session must survive as provenance.
        run_role(self.repo, "some-other-session", "claim", "helper", "--row", "ENG-BAR")
        marker = json.loads(
            (self.repo / ".git/vllm-cpp-agent-role").read_text(encoding="utf-8"))
        self.assertEqual(marker["session"], "some-other-session")

        saved_env = os.environ.get("VLLM_CPP_AGENT_SESSION")
        saved_cwd = os.getcwd()
        os.environ["VLLM_CPP_AGENT_SESSION"] = "a-completely-different-session"
        os.chdir(self.repo)
        try:
            state = role.resolve()
        finally:
            os.chdir(saved_cwd)
            if saved_env is None:
                os.environ.pop("VLLM_CPP_AGENT_SESSION", None)
            else:
                os.environ["VLLM_CPP_AGENT_SESSION"] = saved_env

        self.assertEqual(state["role"], "helper")
        self.assertEqual(state["row"], "ENG-BAR")
        self.assertEqual(state["declared_by"], "some-other-session")

    def test_the_records_are_shared_by_every_worktree(self) -> None:
        # Keying on the worktree must not NARROW the records: they live in the
        # git common dir, shared by every worktree, which is what makes "who is
        # coordinating where" answerable from any of them.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        rival = self.worktree("rival")
        self.assertEqual(run_role(rival, "b", "claim", "operator").returncode, 0)
        # The same two records are visible from either side, and neither lives
        # in a work tree where a commit could pick it up.
        self.assertEqual(len(self.record_files()), 2)
        for where in (self.repo, rival):
            self.assertNotIn("vllm-cpp-operators", subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=where, text=True))

    def test_helper_marker_does_not_leak_into_another_worktree(self) -> None:
        # Isolation, asserted where it is now real. The SAME session id in
        # another worktree must resolve as undeclared: a helper materializes its
        # own worktree, so its marker cannot reach anyone else's.
        run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        out = run_role(self.worktree("elsewhere"), "a", "show")
        self.assertEqual(out.returncode, 3)
        self.assertIn("UNDECLARED", out.stdout)

    def _legacy(self) -> Path:
        """The pre-#285 single-file lock. Still read, so a session that claimed
        before the change does not silently become UNDECLARED mid-flight."""
        return self.common() / "vllm-cpp-operator.lock"

    def _resolve_in(self, where: Path) -> dict:
        """resolve()'s OWN return value, read in `where`.

        The CLI prints a rendering; several keys of the resolved state never
        reach stdout, so a dict-level assertion is the only way to pin them.
        """
        saved = os.getcwd()
        os.chdir(where)
        try:
            return role.resolve()
        finally:
            os.chdir(saved)

    def test_reclaiming_your_own_record_refreshes_the_heartbeat(self) -> None:
        # A live coordinator is likelier to re-claim than to heartbeat, and a
        # record that ages past the TTL while its owner is alive drops out of
        # everyone else's display of who is working where.
        run_role(self.repo, "a", "claim", "operator")
        self.backdate(self.repo, 10 * 60 * 60)
        run_role(self.repo, "b", "claim", "operator")
        self.assertGreater(
            self.record_of(self.repo)["heartbeat"], time.time() - 60)

    def test_a_legacy_lock_file_is_adopted_as_this_worktrees_record(self) -> None:
        # A pre-#285 session holds the single-file lock. It must keep resolving
        # as operator -- turning a live coordinator UNDECLARED mid-flight fails
        # its next preflight -- and the next claim must heal the repo by moving
        # it into the records directory, so the legacy file cannot linger and be
        # counted twice.
        legacy = self._legacy()
        worktree = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=self.repo, text=True).strip()
        legacy.write_text(json.dumps({
            "session": "pre-285", "worktree": worktree,
            "claimed_at": time.time(), "heartbeat": time.time(),
            "host": "somewhere", "pid": 1}), encoding="utf-8")
        (Path(worktree) / "vllm-cpp-agent-role").write_text(
            json.dumps({"role": "operator", "session": "pre-285", "at": time.time()}),
            encoding="utf-8")

        self.assertIn("role=operator", run_role(self.repo, "pre-285", "show").stdout)
        self.assertEqual(run_role(self.repo, "later", "claim", "operator").returncode, 0)
        self.assertFalse(legacy.exists(), "the legacy lock file was not healed away")
        self.assertEqual(len(self.record_files()), 1, self.records())

    def test_an_operator_whose_record_vanished_is_told_to_re_claim(self) -> None:
        # The one path that still refuses to resolve: an operator marker with no
        # record of its own. It is REACHABLE -- a host cleanup deleted exactly
        # this file on 2026-08-10 -- and it must be distinguishable from "never
        # declared", because the remedy is `claim operator`, which now always
        # succeeds. A live coordinator elsewhere changes none of it.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        rival = self.worktree("live-rival")
        self.assertEqual(run_role(rival, "b", "claim", "operator").returncode, 0)
        self.record_of(self.repo)  # precondition: ours exists before we delete it
        for path in self.record_files():
            if json.loads(path.read_text(encoding="utf-8")).get("worktree") == \
                    subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                            cwd=self.repo, text=True).strip():
                path.unlink()

        state = self._resolve_in(self.repo)
        self.assertIsNone(state["role"])
        self.assertEqual(
            state["reason"],
            "operator marker without a coordinator record; re-claim")
        shown = run_role(self.repo, "a", "show")
        self.assertEqual(shown.returncode, 3)
        # No refusal language survives anywhere: the rival is reported as a
        # peer, never as a blocker, and re-claiming works on the spot.
        self.assertNotIn("cannot be the operator", shown.stdout + shown.stderr)
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        self.assertIn("role=operator", run_role(self.repo, "a", "show").stdout)

    def test_downgrading_out_of_operator_releases_the_record(self) -> None:
        # `claim read-only` ("just looking") is the exact next command an
        # operator types, and leaving the record behind is worse than holding no
        # role: heartbeat answers "not the operator; nothing to heartbeat", so
        # nothing renews it, and the display then shows a coordinator that is
        # not coordinating for the full 2h TTL.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        self.assertEqual(len(self.record_files()), 1)
        downgrade = run_role(self.repo, "a", "claim", "read-only")
        self.assertEqual(downgrade.returncode, 0, downgrade.stderr)
        self.assertEqual(self.record_files(), [])
        self.assertIn("role=read-only", run_role(self.repo, "a", "show").stdout)

    def test_downgrading_to_helper_releases_the_record_too(self) -> None:
        # Same rule, the other non-operator answer: the release keys on "the
        # claimed role is not operator", not on read-only specifically.
        run_role(self.repo, "a", "claim", "operator")
        self.assertEqual(
            run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO").returncode, 0)
        self.assertEqual(self.record_files(), [])

    def test_a_downgrade_never_removes_ANOTHER_worktrees_record(self) -> None:
        # The removal must use the same ownership test `release` does. A
        # read-only claim in one worktree erasing another worktree's record
        # would make the record lie about who is working where.
        run_role(self.repo, "a", "claim", "operator")
        elsewhere = self.worktree("bystander")
        self.assertEqual(run_role(elsewhere, "b", "claim", "read-only").returncode, 0)
        self.assertEqual(len(self.record_files()), 1)
        self.assertIn("role=operator", run_role(self.repo, "a", "show").stdout)

    def test_release_from_a_new_session_removes_the_record(self) -> None:
        # Release has to cross the call boundary as well, or a record written in
        # one call is unreleasable in the next and shows a phantom coordinator
        # until the TTL.
        run_role(self.repo, "call-1", "claim", "operator")
        run_role(self.repo, "call-2-different-pid", "release")
        self.assertEqual(self.record_files(), [])


class CoordinatorRecords(_TempRepo, unittest.TestCase):
    """Issue #285: the file records who is coordinating where; it never refuses.

    Everything here is stated across REAL worktrees, because a record keys on
    the worktree and one file per worktree is what makes concurrent claimants
    safe: two claims write two different paths, so neither can lose the other's
    record, and each publish is a rename over its own path.
    """

    def peers_shown(self, where: Path) -> str:
        shown = run_role(where, "watcher", "show")
        return shown.stdout

    def test_show_lists_the_other_live_coordinators(self) -> None:
        # The point of keeping the file at all: who, which worktree, and how
        # long since the heartbeat. Diagnosing a dead holder needed exactly this
        # on 2026-08-10 and the tool would not say it.
        run_role(self.repo, "primary", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "rival-session", "claim", "operator")
        rival_git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=rival, text=True).strip()

        shown = self.peers_shown(self.repo)
        self.assertIn("other coordinators", shown)
        self.assertIn(rival_git_dir, shown)          # which worktree
        self.assertIn("rival-session", shown)        # who
        self.assertIn("heartbeat", shown)            # how long since
        self.assertRegex(shown, r"\d+s ago")

    def test_show_never_lists_this_worktree_as_a_peer(self) -> None:
        # A record that counted itself would report a coordinator conflict with
        # nobody, which is how a record starts reading like a lock again.
        run_role(self.repo, "solo", "claim", "operator")
        own_git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=self.repo, text=True).strip()
        shown = run_role(self.repo, "solo", "show").stdout
        self.assertIn("role=operator", shown)
        self.assertNotIn("other coordinators", shown)
        self.assertNotIn(own_git_dir, shown)

    def test_release_removes_only_the_callers_record(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "b", "claim", "operator")

        run_role(self.repo, "a", "release")
        self.assertEqual(len(self.record_files()), 1, self.records())
        self.record_of(rival)  # the survivor is the OTHER worktree's
        self.assertIn("role=operator", run_role(rival, "b", "show").stdout)
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)

    def test_a_stale_record_is_pruned_and_never_blocks(self) -> None:
        # A session killed mid-flight leaves a dead pid and a frozen heartbeat.
        # The TTL stays; what changes is the consequence -- the stale record
        # drops out of the display instead of refusing everyone for two hours.
        run_role(self.repo, "crashed", "claim", "operator")
        self.backdate(self.repo, 10 * 60 * 60)
        successor = self.worktree("successor")

        took = run_role(successor, "b", "claim", "operator")
        self.assertEqual(took.returncode, 0, took.stderr)
        self.assertNotIn("crashed", took.stdout + took.stderr)
        shown = run_role(successor, "b", "show").stdout
        self.assertIn("role=operator", shown)
        self.assertNotIn("other coordinators", shown)
        self.assertNotIn("crashed", shown)
        self.assertEqual(len(self.record_files()), 1, self.records())
        self.record_of(successor)

    def test_a_stale_record_is_hidden_before_anything_prunes_it(self) -> None:
        # Pruning is a WRITE, so it happens on claim and never in `show`:
        # agent-preflight.sh documents itself as never writing anything. The
        # display must therefore filter on the TTL itself, not rely on a
        # previous claim having cleaned up.
        run_role(self.repo, "crashed", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "live", "claim", "operator")
        self.backdate(self.repo, 10 * 60 * 60)

        shown = run_role(rival, "live", "show").stdout
        self.assertNotIn("crashed", shown)
        self.assertNotIn("other coordinators", shown)
        self.assertEqual(len(self.record_files()), 2, "show must not delete anything")

    def test_concurrent_claims_lose_no_record(self) -> None:
        # The representation exists for this case. Eight worktrees claim at
        # once; every one of them must end up recorded, and every record must be
        # readable JSON -- a shared file rewritten by eight writers loses some.
        worktrees = [self.worktree(f"coord{index}") for index in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda pair: run_role(pair[1], f"s{pair[0]}", "claim", "operator"),
                list(enumerate(worktrees))))
        for index, result in enumerate(results):
            self.assertEqual(result.returncode, 0, f"claim {index}: {result.stderr}")
        self.assertEqual(len(self.record_files()), 8, self.records())
        for worktree in worktrees:
            self.record_of(worktree)  # every one survived, none was overwritten
        # And every claimant sees the other seven.
        shown = run_role(worktrees[0], "s0", "show").stdout
        self.assertIn("other coordinators recorded: 7", shown)

    def test_a_claim_never_rewrites_another_worktrees_record(self) -> None:
        # The atomicity argument in one assertion: a writer only ever touches
        # the path derived from its OWN worktree, so a peer's bytes are
        # untouched. If claims ever shared a file this is the first thing to go.
        run_role(self.repo, "a", "claim", "operator")
        before = self.record_files()[0].read_bytes()
        rival = self.worktree("rival")
        run_role(rival, "b", "claim", "operator")
        run_role(rival, "b", "heartbeat")
        run_role(rival, "b", "release")
        self.assertEqual(self.record_files()[0].read_bytes(), before)

    def test_heartbeat_renews_only_this_worktrees_record(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "b", "claim", "operator")
        self.backdate(self.repo, 30 * 60)
        self.backdate(rival, 30 * 60)

        beat = run_role(rival, "b", "heartbeat")
        self.assertEqual(beat.returncode, 0, beat.stderr)
        self.assertGreater(self.record_of(rival)["heartbeat"], time.time() - 60)
        self.assertLess(self.record_of(self.repo)["heartbeat"], time.time() - 60)

    def test_a_record_is_never_written_inside_a_work_tree(self) -> None:
        # It must stay uncommittable. `git status` in the worktree that claimed
        # is the check that matters, because that is where a `git add` runs.
        run_role(self.repo, "a", "claim", "operator")
        self.assertTrue(self.records_dir().is_dir())
        self.assertEqual(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repo, text=True).strip(),
            "")


class RecordPublishAndBadInput(_TempRepo, unittest.TestCase):
    """How a record reaches its own path, and what a bad file in the directory
    does to `show`.

    `test_a_claim_never_rewrites_another_worktrees_record` pins the DISJOINT
    PATHS half of the concurrency argument: it fails on shared-path symptoms and
    says nothing about the publish itself. A fresh review (2026-08-10) replaced
    `write temp + os.replace` with an in-place, byte-at-a-time flushing write --
    deliberately torn publishes -- and all three suites stayed green, so the
    mechanism the module docstring calls atomic was pinned by nothing. The same
    review found the re-claim path unlinking its own record before rewriting it,
    and both defensive branches (`record_is_stale`'s unreadable heartbeat,
    `read_records`'s suffix filter) reachable by no test at all.

    A third round then showed both of those repairs still short. The re-claim
    test only ever started from a FRESH record, so it never reached the prune
    that runs first and unlinked our own record once it was stale -- the common
    case, since the TTL is two hours. And both publish tests survive
    `unlink(target); target.write_text(new)`: a new inode leaves the hardlink
    witness intact and no temp is left, but the NAME is transiently absent,
    which is neither the old record nor the new one.
    """

    def test_a_publish_replaces_the_record_and_never_rewrites_it_in_place(self) -> None:
        # A hardlink is the inode a concurrent reader is holding. `os.replace`
        # publishes a NEW inode over the name, so the reader keeps seeing whole
        # old bytes; any in-place rewrite reaches through the link, which is
        # exactly how a reader gets half a record.
        run_role(self.repo, "a", "claim", "operator")
        published = self.record_files()[0]
        before = published.read_bytes()
        witness = published.with_name("witness-hardlink")
        os.link(published, witness)

        beat = run_role(self.repo, "a", "heartbeat")
        self.assertEqual(beat.returncode, 0, beat.stderr)
        self.assertNotEqual(published.read_bytes(), before, "nothing was published")
        self.assertEqual(
            witness.read_bytes(), before,
            "the record was rewritten IN PLACE: a reader holding the previous "
            "inode sees the new bytes, so it can observe a partial record")

    def test_a_publish_leaves_no_temporary_file_behind(self) -> None:
        # The other half of rename-publish: the temp file is CONSUMED by the
        # rename. A publish that copies instead leaves it, and a leftover temp
        # is a record-shaped file nobody owns.
        run_role(self.repo, "a", "claim", "operator")
        run_role(self.repo, "a", "heartbeat")
        run_role(self.repo, "a", "claim", "operator")
        residue = sorted(entry.name for entry in self.records_dir().iterdir()
                         if entry.suffix != ".json")
        self.assertEqual(residue, [], f"publish residue left behind: {residue}")

    def _killed_reclaim(self) -> None:
        """`claim operator` whose publish dies mid-flight. What survives is the
        point: whatever is on disk at that instant is what a killed session
        leaves behind."""
        saved = os.getcwd()
        os.chdir(self.repo)
        try:
            with mock.patch.object(role, "write_our_record",
                                   side_effect=RuntimeError("killed mid-publish")):
                with self.assertRaises(RuntimeError):
                    role.cmd_claim(argparse.Namespace(
                        role="operator", row=None, headless=False))
        finally:
            os.chdir(saved)

    def test_a_publish_never_leaves_the_record_NAME_absent(self) -> None:
        # The two tests above both survive `target.unlink(); target.write_text()`
        # (review mutation MINE-B, 2026-08-10): a fresh inode leaves the hardlink
        # witness reading the old bytes, and no temp file is left behind. What
        # that publish does do is make the NAME transiently absent, which is
        # neither "the old record" nor "the new one" -- a `show` landing in the
        # window reports an operator marker with no record and exits 3.
        #
        # So the NAME is watched rather than the bytes. Polling for the window
        # would be a race; instead the publish is observed from inside: at the
        # instant any file content is written, the published path must already
        # resolve, and nothing may unlink it. Both hold for temp + os.replace,
        # and neither holds for unlink-then-create. The record has to exist
        # first -- "old or new, never absent" says nothing about the first
        # publish, which has no old.
        run_role(self.repo, "a", "claim", "operator")

        absent_when_writing: list[str] = []
        unlinked: list[str] = []
        real_write_text = Path.write_text
        real_path_unlink = Path.unlink
        real_os_unlink = os.unlink
        real_os_remove = os.remove

        saved = os.getcwd()
        os.chdir(self.repo)
        try:
            target = role.record_path()

            def watched_write_text(path, *args, **kwargs):
                if not target.exists():
                    absent_when_writing.append(str(path))
                return real_write_text(path, *args, **kwargs)

            def watched_path_unlink(path, *args, **kwargs):
                if Path(path) == target:
                    unlinked.append(str(path))
                return real_path_unlink(path, *args, **kwargs)

            def watched_os_unlink(path, *args, **kwargs):
                if Path(path) == target:
                    unlinked.append(str(path))
                return real_os_unlink(path, *args, **kwargs)

            def watched_os_remove(path, *args, **kwargs):
                if Path(path) == target:
                    unlinked.append(str(path))
                return real_os_remove(path, *args, **kwargs)

            with mock.patch.object(Path, "write_text", watched_write_text), \
                    mock.patch.object(Path, "unlink", watched_path_unlink), \
                    mock.patch.object(os, "unlink", watched_os_unlink), \
                    mock.patch.object(os, "remove", watched_os_remove):
                role.write_our_record()
        finally:
            os.chdir(saved)

        self.assertEqual(
            unlinked, [],
            "the publish UNLINKED the record name; a reader in that window sees "
            "an operator marker with no record, not the old record")
        self.assertEqual(
            absent_when_writing, [],
            "the new bytes were written while the record name did not exist, so "
            f"the name was transiently absent: {absent_when_writing}")

    def test_a_reclaim_never_unlinks_its_own_record(self) -> None:
        # Re-claim must REPLACE, never unlink-then-create. With the publish
        # killed mid-flight, the record from the previous claim has to survive
        # byte-for-byte -- an operator marker with no record is the one state
        # that still refuses to resolve, and it is what this change exists to
        # remove.
        #
        # Both ages, because they take DIFFERENT code paths and only the fresh
        # one was covered: `cmd_claim` prunes before it publishes, and a prune
        # that is not scoped to skip our own path unlinks our record whenever it
        # is the stale one. Stale is the COMMON case -- the TTL is two hours and
        # a session re-claims at the top of its next tool call -- and `resolve`
        # matches our own record with no staleness filter, so the state RESOLVES
        # FINE until the re-claim destroys it. One backdate is the whole
        # difference between the two legs.
        for age, backdate_by in (("fresh", None),
                                 ("stale", role.RECORD_TTL_SECONDS + 60)):
            with self.subTest(own_record=age):
                run_role(self.repo, "a", "claim", "operator")
                if backdate_by is not None:
                    self.backdate(self.repo, backdate_by)
                before = self.record_files()[0].read_bytes()
                # Whatever its age, this worktree resolves BEFORE the re-claim.
                # So any refusal afterwards was manufactured by the re-claim.
                self.assertEqual(run_role(self.repo, "a", "show").returncode, 0)

                self._killed_reclaim()

                self.assertEqual(
                    len(self.record_files()), 1,
                    "the re-claim unlinked this worktree's own record before "
                    f"republishing it: {self.records()}")
                self.assertEqual(self.record_files()[0].read_bytes(), before)
                # The state the whole change exists to remove: an operator
                # marker with no record, created out of one that resolved.
                self.assertEqual(run_role(self.repo, "a", "show").returncode, 0)

    def test_show_survives_a_corrupt_record_a_bad_heartbeat_and_a_stray_temp(self) -> None:
        # Nothing exercised either defensive branch. A record whose heartbeat is
        # unreadable must read as STALE, not raise out of `show`; a `.tmp` file
        # caught mid-publish must not parse as a coordinator. And `show` must
        # still write nothing: agent-preflight.sh documents itself as never
        # writing, and it calls exactly this path.
        run_role(self.repo, "a", "claim", "operator")
        directory = self.records_dir()
        (directory / "corrupt.json").write_text("{ not json at all", encoding="utf-8")
        (directory / "bad-heartbeat.json").write_text(json.dumps({
            "session": "bad-beat-session", "worktree": "/nowhere/.git",
            "claimed_at": "not-a-number", "heartbeat": "not-a-number",
            "host": "somewhere", "pid": 4321}), encoding="utf-8")
        (directory / ".half-published.999.tmp").write_text(json.dumps({
            "session": "stray-temp-session", "worktree": "/torn/.git",
            "claimed_at": time.time(), "heartbeat": time.time(),
            "host": "somewhere", "pid": 999}), encoding="utf-8")
        before = sorted((entry.name, entry.read_bytes()) for entry in directory.iterdir())

        shown = run_role(self.repo, "a", "show")

        self.assertEqual(shown.returncode, 0, shown.stdout + shown.stderr)
        self.assertIn("role=operator", shown.stdout)
        self.assertNotIn("Traceback", shown.stderr)
        # Neither bad file may become a coordinator: one is past no TTL it can
        # state, the other is a temp file mid-publish.
        self.assertNotIn("other coordinators", shown.stdout)
        self.assertNotIn("bad-beat-session", shown.stdout)
        self.assertNotIn("stray-temp-session", shown.stdout)
        self.assertEqual(
            sorted((entry.name, entry.read_bytes()) for entry in directory.iterdir()),
            before, "`show` wrote to the records directory")


class RoleDiscipline(unittest.TestCase):
    def test_feature_path_classification(self) -> None:
        for path in ("src/vllm/a.cpp", "include/vt/b.h", "tests/vt/c.cpp",
                     "CMakeLists.txt", "cmake/x.cmake"):
            self.assertTrue(discipline.is_feature_path(path), path)
        for path in ("scripts/check-x.py", "tests/scripts/test_x.py",
                     ".agents/state.md", ".agents/state.csv",
                     ".agents/state-index/2026-08-001.csv",
                     ".agents/state-events/2026-08/STATE-20260808T120000-001.md",
                     ".agents/completed/state-migration-manifest.csv", "docs/STATUS.md",
                     ".github/workflows/ci.yml"):
            self.assertFalse(discipline.is_feature_path(path), path)

    def test_direct_feature_push_is_a_violation(self) -> None:
        problems = discipline.commit_violations(
            "abc1234", ["p1"], "perf: faster kernel", "", ["src/vllm/a.cpp"]
        )
        self.assertTrue(problems)
        self.assertIn("without a reviewed", problems[0])

    def test_row_pr_merge_is_accepted(self) -> None:
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["p1", "p2"],
                "Merge pull request #12 from mudler/row/ENG-FOO", "",
                ["src/vllm/a.cpp"]),
            [],
        )

    def test_githubs_synthetic_pr_merge_is_accepted(self) -> None:
        """`refs/pull/N/merge` names neither the branch nor the PR.

        GitHub builds it as "Merge <head> into <base>" and CI checks out exactly
        that commit, so before this every feature PR failed a gate about MAIN's
        history on a commit that never lands on main. The reviewed content is the
        SECOND parent: the PR head.
        """
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["base", "head"],
                "Merge 01cf15a1 into 4cfeee13", "",
                ["src/vllm/a.cpp"],
                ("feat(videos): a thing\n\nbranch `row/SERVE-VIDEOS-OAI`.",)),
            [],
        )

    def test_a_merge_naming_no_row_anywhere_still_FAILS(self) -> None:
        """The hole the case above must not open: a merge of a NON-row branch."""
        problems = discipline.commit_violations(
            "abc1234", ["base", "head"],
            "Merge 01cf15a1 into 4cfeee13", "",
            ["src/vllm/a.cpp"],
            ("perf: hand-edit a kernel\n\nno branch, no PR",))
        self.assertTrue(problems)
        self.assertIn("without a reviewed", problems[0])
        # And with no merged-branch messages at all (a plain local merge).
        self.assertTrue(
            discipline.commit_violations(
                "abc1234", ["base", "head"], "Merge branch 'wip'", "",
                ["src/vllm/a.cpp"])
        )

    def test_squash_merge_with_pr_number_is_accepted(self) -> None:
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["p1"], "feat: thing (#12)", "", ["src/vllm/a.cpp"]),
            [],
        )

    def test_integration_only_commit_is_exempt(self) -> None:
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["p1"], "docs: record", "",
                ["scripts/check-x.py", ".agents/state.md", "docs/STATUS.md"]),
            [],
        )

    def test_mixed_commit_is_judged_on_its_feature_paths(self) -> None:
        self.assertTrue(
            discipline.commit_violations(
                "abc1234", ["p1"], "chore", "",
                ["docs/STATUS.md", "src/vllm/a.cpp"])
        )

    def test_enforcement_is_live_and_anchored_to_a_real_commit(self) -> None:
        """Enabled 2026-08-05. The cutover must be a commit that exists."""
        self.assertIsNotNone(discipline.ROLE_DISCIPLINE_SINCE)
        import subprocess
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{discipline.ROLE_DISCIPLINE_SINCE}^{{commit}}"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_the_cutover_commit_itself_is_exempt(self) -> None:
        """History created under the previous direct-push policy stays green."""
        self.assertTrue(discipline.enforced(discipline.ROLE_DISCIPLINE_SINCE))
        first = discipline.git("rev-list", "--max-parents=0", "HEAD").split()[0]
        self.assertFalse(discipline.enforced(first))

    def test_a_direct_feature_push_after_cutover_now_FAILS(self) -> None:
        """The whole point of enabling it: this is an error, not a report."""
        problems = discipline.commit_violations(
            "deadbee", ["p1"], "perf: hand-edit a kernel", "", ["src/vt/cuda/x.cu"])
        self.assertTrue(problems)
        self.assertTrue(discipline.enforced("HEAD"))

    def test_exact_pending_pr_range_is_reportable_in_any_checkout(self) -> None:
        head = discipline.git("rev-parse", "HEAD")
        base = discipline.git("rev-parse", "HEAD^")
        saved = sys.argv
        sys.argv = [
            saved[0],
            "--base",
            base,
            "--head",
            head,
            "--pending-pr-head",
            head,
        ]
        try:
            self.assertEqual(discipline.main(), 0)
        finally:
            sys.argv = saved

    def test_landed_detached_commit_remains_strict_without_pending_evidence(self) -> None:
        # The subject here is main()'s DECISION: a violation on a commit that
        # has landed, with no --pending-pr-head evidence, is strict (1), not a
        # REPORT (0). Feed it a fixed violation instead of relying on the real
        # HEAD to be one. It did rely on that, and the coupling was live: under
        # a pull_request event CI checks out the SYNTHETIC merge, whose
        # merged_messages are the PR's own commit bodies, so any PR whose
        # message cites an issue or PR number matched PR_REFERENCE, HEAD stopped
        # being a violation, main() returned 0, and this test failed for a
        # reason that had nothing to do with what it asserts.
        saved = sys.argv
        sys.argv = [saved[0], "--commit", "HEAD"]
        try:
            with mock.patch.object(discipline, "has_reached_main", return_value=True), \
                 mock.patch.object(discipline, "enforced", return_value=True), \
                 mock.patch.object(discipline, "inspect", return_value=["x: landed without a row PR"]):
                self.assertEqual(discipline.main(), 1)
        finally:
            sys.argv = saved

    def test_the_real_push_that_reddened_main_now_passes(self) -> None:
        """`3bbee96e..0cf3dbbb` is the exact CI range that failed for PR #178.

        The unit checks above own the rule; this one owns the fact that the rule
        answers THE push CI ran. Skipped rather than failed where the history is
        absent (a shallow clone), because the checkers themselves need depth.

        `has_reached_main` and `enforced` are pinned TRUE on purpose: run from a
        `row/*` worktree they report every commit as pending PR disposition, so
        main() would return 0 without judging arrival at all and this test would
        pass against the very defect it exists to catch.
        """
        base, head = "3bbee96ea8649cefd748bf3b979f91ae4f31d08b", "0cf3dbbb"
        try:
            discipline.git("cat-file", "-e", f"{base}^{{commit}}")
            discipline.git("cat-file", "-e", f"{head}^{{commit}}")
        except subprocess.CalledProcessError:
            self.skipTest("history for the #178 push range is not present")
        saved = sys.argv
        sys.argv = [saved[0], "--base", base, "--head", head]
        try:
            with mock.patch.object(discipline, "has_reached_main", return_value=True), \
                 mock.patch.object(discipline, "enforced", return_value=True):
                self.assertEqual(discipline.main(), 0)
        finally:
            sys.argv = saved

    def test_a_pr_number_in_the_body_does_not_decide_this_gate(self) -> None:
        """Regression: the case that made the test above fail in CI.

        A commit message that merely MENTIONS `#123` must not change main()'s
        landed-vs-pending decision. This pins the decision to the inputs it is
        about, so a message quoting PR numbers cannot flip the outcome again.
        """
        saved = sys.argv
        sys.argv = [saved[0], "--commit", "HEAD"]
        try:
            with mock.patch.object(discipline, "has_reached_main", return_value=True), \
                 mock.patch.object(discipline, "enforced", return_value=True), \
                 mock.patch.object(discipline, "inspect", return_value=["x: see (#157) and #174"]):
                self.assertEqual(discipline.main(), 1)
        finally:
            sys.argv = saved


class MergeLandedPrContent(unittest.TestCase):
    """A PR landed with a REAL merge commit pushes the branch commits too.

    Those commits were never required to name the PR -- the merge above them
    does -- so judging each one on its own message called every merge-landed PR
    a direct push. Every main push that merged a PR was red for it (#178's
    `6603356a`, #204's `e73cbbae`, #196's `1a02ab4f`). These build real git
    history rather than hand-fed parents, because the defect was in WHICH
    commits get judged, not in how one commit's message reads.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "T")
        self.commit("docs: seed", "docs/STATUS.md")

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL
        ).strip()

    def commit(self, message: str, path: str) -> str:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{message}\n{path}\n")
        self.git("add", path)
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def land_via_merge(self, merge_message: str, branch_message: str) -> tuple[str, str]:
        """Build `main -- merge(row branch)` and return (branch head, merge)."""
        self.git("checkout", "-q", "-b", "row/ENG-FOO")
        head = self.commit(branch_message, "src/vllm/a.cpp")
        self.git("checkout", "-q", "main")
        self.git("merge", "-q", "--no-ff", "-m", merge_message, "row/ENG-FOO")
        return head, self.git("rev-parse", "HEAD")

    def content(self, *commits: str) -> frozenset[str]:
        with mock.patch.object(discipline, "ROOT", self.repo):
            return discipline.merged_pr_content(list(commits))

    def test_a_row_pr_merge_exempts_the_branch_commits_it_brings_in(self) -> None:
        head, merge = self.land_via_merge(
            "Merge pull request #12 from mudler/row/ENG-FOO", "perf: faster kernel"
        )
        self.assertIn(head, self.content(merge))

    def test_the_exemption_does_not_reach_mains_own_history(self) -> None:
        """`--not parents[0]`: the first-parent side is main, not PR content."""
        seed = self.git("rev-parse", "HEAD")
        _, merge = self.land_via_merge(
            "Merge pull request #12 from mudler/row/ENG-FOO", "perf: faster kernel"
        )
        self.assertNotIn(seed, self.content(merge))

    def test_a_direct_push_is_not_laundered_by_a_later_row_pr_merge(self) -> None:
        """The hole this must not open: merging a PR on top of a direct push."""
        pushed = self.commit("perf: hand-edit a kernel", "src/vt/cuda/x.cu")
        _, merge = self.land_via_merge(
            "Merge pull request #12 from mudler/row/ENG-FOO", "perf: faster kernel"
        )
        self.assertNotIn(pushed, self.content(pushed, merge))

    def test_a_merge_naming_no_row_anywhere_exempts_NOTHING(self) -> None:
        head, merge = self.land_via_merge("Merge branch 'wip'", "perf: hand-edit")
        self.assertEqual(self.content(merge), frozenset())
        self.assertNotIn(head, self.content(merge))


class ReadOnlyAndModeTests(unittest.TestCase):
    def test_claimable_roles_stay_exactly_two(self):
        # read-only must never become a third claimable role: it takes no lock
        # and no worktree. CLAIMABLE_ROLES is the vocabulary a "may this session
        # write?" test is meant to key on; it has no consumer outside
        # agent-role.py and this suite today, so this pin protects the
        # constant's meaning rather than a live refusal.
        self.assertEqual(role.CLAIMABLE_ROLES, ("operator", "helper"))
        self.assertIn("read-only", role.DECLARABLE)
        self.assertNotIn("read-only", role.CLAIMABLE_ROLES)

    def test_read_only_is_declarable(self):
        self.assertIn("read-only", role.DECLARABLE)

    def test_the_roles_alias_is_not_widened(self):
        # ROLES is the alias a write-gating call site would import; no such
        # call site exists yet, so keeping it the CLAIMABLE pair is what stops
        # the first one from being born wrong. Mutating it to DECLARABLE leaves every other
        # assertion in this suite green, so the constraint that keeps read-only
        # out of "may this session write?" would be enforced by comment only.
        self.assertEqual(role.ROLES, role.CLAIMABLE_ROLES)
        self.assertNotIn("read-only", role.ROLES)

    def test_mode_defaults_to_interactive(self):
        # Headless is DECLARED, never inferred. Absent an explicit flag the
        # session is interactive.
        self.assertEqual(role.mode_from_marker({}), "interactive")
        self.assertEqual(role.mode_from_marker({"mode": "headless"}), "headless")
        self.assertEqual(role.mode_from_marker({"mode": "nonsense"}), "interactive")


class ReadOnlyAndModeResolved(_TempRepo, unittest.TestCase):
    """Drives resolve() itself, not only the pure helpers above.

    mode_from_marker() can be perfectly correct while resolve() never calls it:
    the key would simply be absent from the resolved state and a test that only
    exercised the helper would stay green. So these claim through the real CLI
    and read the mode back out of resolve()'s OWN return value.
    """

    def _resolve_as(self, session: str, where: Path | None = None) -> dict:
        cwd = os.getcwd()
        saved = os.environ.get("VLLM_CPP_AGENT_SESSION")
        os.chdir(where or self.repo)
        os.environ["VLLM_CPP_AGENT_SESSION"] = session
        try:
            return role.resolve()
        finally:
            os.chdir(cwd)
            if saved is None:
                del os.environ["VLLM_CPP_AGENT_SESSION"]
            else:
                os.environ["VLLM_CPP_AGENT_SESSION"] = saved

    def test_resolve_carries_a_declared_headless_mode(self) -> None:
        claimed = run_role(self.repo, "a", "claim", "read-only", "--headless")
        self.assertEqual(claimed.returncode, 0, claimed.stderr)
        state = self._resolve_as("a")
        self.assertEqual(state["role"], "read-only")
        self.assertEqual(state["mode"], "headless")

    def test_resolve_reports_interactive_unless_headless_was_declared(self) -> None:
        run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        self.assertEqual(self._resolve_as("a")["mode"], "interactive")
        # An UNDECLARED context is interactive too: silence is never headless.
        # Genuinely undeclared means another WORKTREE since the 2026-08-06
        # correction; another session id in THIS one resolves to the role that
        # was declared here.
        undeclared = self._resolve_as("b", where=self.worktree("undeclared"))
        self.assertIsNone(undeclared["role"])
        self.assertEqual(undeclared["mode"], "interactive")

    def test_read_only_writes_no_coordinator_record(self) -> None:
        # The whole reason read-only exists: a session that only reads is not
        # coordinating, so it must not appear in the record of who is.
        self.assertEqual(run_role(self.repo, "a", "claim", "read-only").returncode, 0)
        self.assertEqual(self.record_files(), [])
        self.assertIn("role=read-only", run_role(self.repo, "a", "show").stdout)
        # ... and a real coordinator elsewhere records itself normally.
        self.assertEqual(
            run_role(self.worktree("real-operator"), "b", "claim", "operator").returncode, 0)
        self.assertEqual(len(self.record_files()), 1)


if __name__ == "__main__":
    unittest.main()
