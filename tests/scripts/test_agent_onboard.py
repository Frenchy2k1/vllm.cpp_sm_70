#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-onboard.py.

The probe exists to report what is unresolved. Its one job is to be honest
about absence: a missing .env and an unreadable .env must not look the same as
a complete one, and an undeclared role must never render as a declared one.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


onboard = _load("agent_onboard", "scripts/agent-onboard.py")


class EnvStateTests(unittest.TestCase):
    def test_env_keys_match_the_tracked_example(self):
        # The probe must never invent a key. .env.example is the only source.
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        declared = {
            line.split("=", 1)[0]
            for line in example.splitlines()
            if line and not line.startswith("#") and "=" in line
        }
        self.assertEqual(set(onboard.ENV_KEYS), declared)

    def test_missing_file_reports_missing_not_incomplete(self):
        status, missing = onboard.env_state(ROOT / "does-not-exist-.env")
        self.assertEqual(status, "missing")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))

    def test_blank_value_counts_as_missing_that_key(self):
        # An empty value is a legitimate "unavailable", but the probe still
        # has to report it so the agent knows what it may ask for.
        text = "\n".join(f"{k}=" for k in onboard.ENV_KEYS)
        status, missing = onboard.env_state_from_text(text)
        self.assertEqual(status, "incomplete")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))

    def test_all_values_present_reports_present(self):
        text = "\n".join(f"{k}=/some/path" for k in onboard.ENV_KEYS)
        status, missing = onboard.env_state_from_text(text)
        self.assertEqual(status, "present")
        self.assertEqual(missing, [])

    def test_unreadable_file_is_neither_present_nor_absent(self):
        # Beyond the brief. A .env that exists but cannot be read must not
        # crash the probe, must not read as complete, and must not read as
        # absent either: "create one" is the wrong instruction for a file that
        # is already there.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.mkdir()  # exists(), but read_text() raises IsADirectoryError
            status, missing = onboard.env_state(path)
        self.assertEqual(status, "unreadable")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))


class ProbeRenderTests(unittest.TestCase):
    UNDECLARED = {
        "role": None, "row": None, "mode": "interactive",
        "env": "missing", "env_missing": ["VLLM_ORACLE"], "queue": ["ENG-FOO"],
    }

    def test_undeclared_role_renders_as_undeclared(self):
        out = onboard.render_probe(self.UNDECLARED)
        self.assertIn("UNDECLARED", out)
        self.assertNotIn("operator", out.split("queue")[0])

    def test_declared_role_renders_with_its_row(self):
        # The row id must NOT be one the fixture queue already contains, or the
        # queue line satisfies the assertion and deleting row rendering stays
        # green. Assert the `row=` prefix, not the bare id.
        out = onboard.render_probe(dict(self.UNDECLARED, role="helper", row="KERNEL-BAR"))
        self.assertIn("helper", out)
        self.assertIn("row=KERNEL-BAR", out)

    def test_undeclared_render_carries_the_interview_hint(self):
        # The hint is the whole point of the probe: without it an agent sees a
        # state line and no instruction. Deleting the block must go red.
        out = onboard.render_probe(self.UNDECLARED)
        self.assertIn("claim", out)
        self.assertIn("read-only", out)
        self.assertNotIn("claim", onboard.render_probe(
            dict(self.UNDECLARED, role="helper", row="KERNEL-BAR")))
        # Added: an absent mode is an absence too. resolve() has no mode until
        # step 2, so it must render as a default and never as a declaration.
        # Assert the whole mode field: the hint line already contains "not
        # declared", so a looser assertion would pass on a fabricated mode.
        undeclared_mode = {k: v for k, v in self.UNDECLARED.items() if k != "mode"}
        self.assertIn(
            "mode: interactive (default, not declared)",
            onboard.render_probe(undeclared_mode),
        )

    def test_live_coordinators_are_rendered_and_never_as_a_blocker(self):
        # Beyond the brief. Until issue #285 this asserted a NOTE saying the
        # lock was "held by another live session" so the reader would not run a
        # claim "that will fail". The claim no longer fails, so the peer is
        # rendered as information -- who, where, how long since the heartbeat --
        # and the interview hint still offers every role.
        peer = {"worktree": "/repo/.git/worktrees/other", "session": "peer-1",
                "host": "box", "pid": 42, "heartbeat": 0}
        out = onboard.render_probe(dict(self.UNDECLARED, operator_peers=[peer]))
        self.assertIn("other coordinators recorded: 1", out)
        self.assertIn("/repo/.git/worktrees/other", out)
        self.assertIn("peer-1", out)
        self.assertNotIn("will fail", out)
        self.assertIn("operator", out)  # the role is still on the table
        # and with nobody else recorded, no coordinator line is invented
        self.assertNotIn("other coordinators", onboard.render_probe(self.UNDECLARED))

    def test_probe_never_exits_nonzero(self):
        # The probe reports; it does not gate. Preflight gates.
        # The render itself goes to a buffer only so the suite's output stays
        # clean; the assertion is unchanged.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(onboard.main(["--probe"]), 0)


ROLE_SCRIPT = ROOT / "scripts/agent-role.py"
ONBOARD_SCRIPT = ROOT / "scripts/agent-onboard.py"


class ProbeFieldsComeFromResolve(unittest.TestCase):
    """probe()'s RETURNED DICT, against a role claimed through the real CLI.

    Every other probe test in this file feeds render_probe an already-populated
    fixture, so nothing asserted that probe() actually reads its fields out of
    agent-role.py. Five hardcodes therefore left the whole suite green:
    `operator_peers: []`, `reason: None`, `mode: "headless"`,
    `role: "operator"` and `env: "present"`. `--probe` is the front door
    .agents/workflow.md sends every session to, and a probe that answers
    `role: operator` to a session that never declared one is worse than no
    probe: it reports a claim that was never made.

    So these claim in a THROWAWAY repo and read the values back out of probe().
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "root", "--allow-empty"], cwd=self.repo,
            check=True,
            env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                     GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))

    def claim(self, where: Path, session: str, *args: str):
        result = subprocess.run(
            [sys.executable, str(ROLE_SCRIPT), "claim", *args], cwd=where,
            env=dict(os.environ, VLLM_CPP_AGENT_SESSION=session),
            capture_output=True, text=True)
        return result

    def worktree(self, name: str) -> Path:
        path = self.repo / f".{name}"
        subprocess.run(["git", "worktree", "add", "-q", str(path), "-b", name],
                       cwd=self.repo, check=True, capture_output=True)
        return path

    def probe_in(self, where: Path) -> dict:
        saved = os.getcwd()
        os.chdir(where)
        try:
            return onboard.probe()
        finally:
            os.chdir(saved)

    def test_probe_reports_the_role_and_row_that_were_claimed(self) -> None:
        # Kills `role: "operator"`, `row: None`, `mode: "headless"` and
        # `reason: None` in one assertion block: a helper claim with no
        # --headless flag disagrees with every one of them.
        claimed = self.claim(self.repo, "a", "helper", "--row", "PROBE-WIRING")
        self.assertEqual(claimed.returncode, 0, claimed.stderr)
        state = self.probe_in(self.repo)
        self.assertEqual(state["role"], "helper")
        self.assertEqual(state["row"], "PROBE-WIRING")
        self.assertEqual(state["mode"], "interactive")
        self.assertEqual(state["reason"], "declared")
        self.assertEqual(state["operator_peers"], [])
        self.assertEqual(state["branch"], "master")
        self.assertEqual(
            state["worktree"],
            subprocess.check_output(
                ["git", "rev-parse", "--absolute-git-dir"],
                cwd=self.repo,
                text=True,
            ).strip(),
        )

    def test_probe_projects_real_helper_worktree_identity_and_branch(self) -> None:
        helper = self.worktree("probe-helper")
        claimed = self.claim(
            helper, "a", "helper", "--row", "PROBE-REAL-WORKTREE"
        )
        self.assertEqual(claimed.returncode, 0, claimed.stderr)

        state = self.probe_in(helper)

        self.assertEqual(state["branch"], "probe-helper")
        self.assertEqual(
            state["worktree"],
            subprocess.check_output(
                ["git", "rev-parse", "--absolute-git-dir"],
                cwd=helper,
                text=True,
            ).strip(),
        )
        self.assertNotEqual(state["worktree"], str(helper))

    def test_probe_reports_a_mode_that_was_declared(self) -> None:
        # The other half of the mode pin: a hardcoded "interactive" survives the
        # test above and dies here. Headless is DECLARED, never inferred, so
        # both directions have to come out of the marker.
        self.assertEqual(
            self.claim(self.repo, "a", "read-only", "--headless").returncode, 0)
        state = self.probe_in(self.repo)
        self.assertEqual(state["role"], "read-only")
        self.assertEqual(state["mode"], "headless")

    def test_probe_carries_the_live_coordinators_out_of_resolve(self) -> None:
        # Kills `operator_peers: []` and `reason: None`, against records written
        # by the REAL CLI in two real worktrees. Until issue #285 this asserted
        # `blocked_by_other_operator: True` and a "held by another live session"
        # NOTE; a peer is now reported and blocks nothing.
        self.assertEqual(self.claim(self.repo, "a", "operator").returncode, 0)
        rival = self.worktree("rival")
        self.assertEqual(self.claim(rival, "rival-session", "operator").returncode, 0)
        rival_git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=rival, text=True).strip()

        state = self.probe_in(self.repo)
        self.assertEqual(state["role"], "operator")
        self.assertEqual(state["reason"], "declared")
        self.assertEqual(
            [record.get("worktree") for record in state["operator_peers"]],
            [rival_git_dir])
        rendered = onboard.render_probe(state)
        self.assertIn("other coordinators recorded: 1", rendered)
        self.assertIn(rival_git_dir, rendered)
        self.assertIn("rival-session", rendered)

    def test_probe_reports_a_vanished_record_as_re_claimable(self) -> None:
        # The one state that still fails to resolve, and it is REACHABLE: a
        # host cleanup deleted this worktree's record on 2026-08-10. It must
        # stay distinguishable from "never declared", because the remedy is
        # `claim operator` and it now always succeeds.
        self.assertEqual(self.claim(self.repo, "a", "operator").returncode, 0)
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip()
        for record in (Path(common) / "vllm-cpp-operators").glob("*.json"):
            record.unlink()

        state = self.probe_in(self.repo)
        self.assertIsNone(state["role"])
        self.assertEqual(
            state["reason"],
            "operator marker without a coordinator record; re-claim")
        self.assertEqual(state["operator_peers"], [])

    def test_probe_json_peer_records_carry_exactly_these_fields(self) -> None:
        # `--probe --json` used to emit one bool for this. It now emits whole
        # peer records, and nothing pinned what may appear in them. This is the
        # machine-readable front door .agents/workflow.md sends sessions to, so
        # the surface is fixed here: a new field in a coordinator record has to
        # be a deliberate edit of this list, not a silent widening. Run through
        # the CLI, because the emitted JSON is the surface, not probe()'s dict.
        self.assertEqual(self.claim(self.repo, "a", "operator").returncode, 0)
        rival = self.worktree("shape-rival")
        self.assertEqual(self.claim(rival, "rival-session", "operator").returncode, 0)

        emitted = subprocess.run(
            [sys.executable, str(ONBOARD_SCRIPT), "--probe", "--json"],
            cwd=self.repo, env=dict(os.environ, VLLM_CPP_AGENT_SESSION="a"),
            capture_output=True, text=True)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        state = json.loads(emitted.stdout)

        self.assertEqual(len(state["operator_peers"]), 1, state["operator_peers"])
        self.assertEqual(
            sorted(state["operator_peers"][0]),
            ["claimed_at", "heartbeat", "host", "path", "pid", "session", "worktree"])

        # A pre-#285 peer widens that set by one: `read_records` tags the legacy
        # single-file lock with `legacy` on the way through, and it reaches this
        # surface like any other peer. Pinned rather than left implicit, because
        # the untagged case above passes either way. This leg goes away with
        # LEGACY_RECORD_NAME, which agent-role.py already schedules for deletion.
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip()
        (Path(common) / onboard.role_mod.LEGACY_RECORD_NAME).write_text(json.dumps({
            "session": "legacy-session", "worktree": "/elsewhere/.git",
            "claimed_at": time.time(), "heartbeat": time.time(),
            "host": "somewhere", "pid": 77}), encoding="utf-8")

        emitted = subprocess.run(
            [sys.executable, str(ONBOARD_SCRIPT), "--probe", "--json"],
            cwd=self.repo, env=dict(os.environ, VLLM_CPP_AGENT_SESSION="a"),
            capture_output=True, text=True)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        peers = {record["session"]: sorted(record)
                 for record in json.loads(emitted.stdout)["operator_peers"]}
        self.assertEqual(
            peers,
            {"rival-session": ["claimed_at", "heartbeat", "host", "path", "pid",
                               "session", "worktree"],
             "legacy-session": ["claimed_at", "heartbeat", "host", "legacy",
                                "path", "pid", "session", "worktree"]})

    def test_probe_reports_the_env_state_it_was_given(self) -> None:
        # Kills `env: "present"`. It cannot be pinned against the real tree --
        # a developer with a complete .env would make the hardcode true -- so
        # env_state is substituted with a value no hardcode can be, and probe()
        # must carry BOTH of its outputs through.
        saved = onboard.env_state
        onboard.env_state = lambda *a, **k: ("stubbed-status", ["STUB_KEY"])
        try:
            state = self.probe_in(self.repo)
        finally:
            onboard.env_state = saved
        self.assertEqual(state["env"], "stubbed-status")
        self.assertEqual(state["env_missing"], ["STUB_KEY"])


class QueueTests(unittest.TestCase):
    def test_queue_is_the_checkers_own_computation_and_failure_is_visible(self):
        # Beyond the brief, two properties of the same function.
        #
        # 1. The queue must come from ready-for-helper.py's queue(), not from a
        #    re-parse of its prose: an uppercase-token filter over that stdout
        #    reads the "READY-FOR-HELPER queue: N row(s)" header as a row (so an
        #    EMPTY queue reports one phantom row) and drops every mixed-case row
        #    id such as MODEL-TEXT-glm4-glm4-for-causal-lm.
        # 2. A queue that could not be computed is not an empty queue. Swallowing
        #    the failure into [] is the queue-side twin of reporting an unreadable
        #    .env as a complete one.
        rows = onboard.ready_rows()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/ready-for-helper.py")],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        header = result.stdout.splitlines()[0]
        self.assertIn("READY-FOR-HELPER queue:", header)
        self.assertEqual(len(rows), int(header.split(":")[1].split()[0]))
        self.assertNotIn("READY-FOR-HELPER", rows)

        def explode():
            raise RuntimeError("record is broken")

        saved = sys.modules.get("ready_for_helper")
        sys.modules["ready_for_helper"] = types.SimpleNamespace(queue=explode)
        try:
            broken_rows, error = onboard.queue_state()
        finally:
            if saved is None:
                del sys.modules["ready_for_helper"]
            else:
                sys.modules["ready_for_helper"] = saved
        self.assertEqual(broken_rows, [])
        self.assertIn("record is broken", error)
        self.assertIn("UNAVAILABLE", onboard.render_probe(
            dict(ProbeRenderTests.UNDECLARED, queue=[], queue_error=error)))


class PreflightWiringTests(unittest.TestCase):
    TEXT = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")

    def test_require_role_defaults_on(self):
        # Anchored to the DEFAULT assignment itself -- the line with no
        # indentation and nothing else on it. A bare assertIn("REQUIRE_ROLE=1")
        # is satisfied by the --require-role arm of the arg loop all by itself,
        # so the default could be flipped back and the whole deliverable of this
        # change would go unprotected. Any line-anchored assignment of zero is
        # refused, quoted or not, because that is what a silent revert looks
        # like however it is spelled.
        self.assertRegex(self.TEXT, r"(?m)^REQUIRE_ROLE=1$")
        self.assertNotRegex(self.TEXT, r"""(?m)^REQUIRE_ROLE=['"]?0['"]?$""")

    def test_opt_out_flag_exists(self):
        self.assertIn("--no-require-role", self.TEXT)

    def test_failure_text_points_to_the_canonical_entrypoint(self):
        # Preflight remains a backstop, but the role interview has one owner.
        self.assertIn("scripts/agent-start.py", self.TEXT)

    def test_staged_refuses_read_only(self):
        self.assertIn("read-only", self.TEXT)
        self.assertIn("STAGED", self.TEXT)

    def test_the_gate_records_a_failure_and_not_only_a_print(self):
        # Every other assertion in this class inspects the text that EXPLAINS
        # the gate, so deleting the one line that enforces it -- the failed+=()
        # inside the REQUIRE_ROLE branch -- leaves them all green while
        # preflight exits 0 on an undeclared role. Pin the enforcing line, and
        # pin that it is inside the branch: outside it, --no-require-role would
        # stop working instead.
        branch = re.search(
            r'if \[ "\$REQUIRE_ROLE" -eq 1 \]; then(.*?)\n  fi', self.TEXT, re.S)
        self.assertIsNotNone(branch, "the REQUIRE_ROLE branch is gone")
        self.assertIn('failed+=("role-undeclared")', branch.group(1))

    def test_onboard_suite_is_registered(self):
        self.assertIn("test_agent_onboard", self.TEXT)

    def test_agent_start_suite_is_registered(self):
        self.assertIn("test_agent_start", self.TEXT)

    def test_read_only_alone_does_not_satisfy_a_write_gate(self):
        # agent-role.py show exits 0 for read-only, so --require-role is
        # satisfied by a declared ABSENCE of claim. That is correct for a plain
        # run and wrong for --staged; the refusal must be explicit.
        self.assertIn("read-only-cannot-stage", self.TEXT)

    def test_preflight_actually_fails_on_an_undeclared_role(self):
        # Every other assertion in this class greps text, so a REWRITE of the
        # default is caught while an OVERRIDE is not: keep `REQUIRE_ROLE=1` and
        # add `REQUIRE_ROLE=0 ` (one trailing space, so the ^...$ anchor misses)
        # on a later line and the whole class stays green while the gate stops
        # failing. Only executing the script closes that hole.
        #
        # Two mechanics this test cannot do the obvious way:
        #
        # * VLLM_CPP_AGENT_SESSION cannot make a role unresolvable. A role keys
        #   on the WORKTREE (agent-role.py marker_path/lock_path, corrected
        #   2026-08-06), so this tree's own marker answers whatever the session
        #   id says. GIT_DIR is the one knob that moves the marker and the lock
        #   together, so the run is pointed at an empty git dir instead.
        # * The run is --role-only, not a full preflight: agent-preflight.sh
        #   runs THIS suite, and a nested full run would recurse without bound.
        #   --role-only executes the same role block and the same REQUIRE_ROLE
        #   branch, which is the code under test.
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty-git-dir"
            subprocess.run(
                ["git", "init", "--quiet", "--bare", str(empty)], check=True)
            env = dict(os.environ, GIT_DIR=str(empty))
            env.pop("GIT_WORK_TREE", None)

            # PRECONDITION, asserted and never skipped: if a role still
            # resolves here the run below proves nothing and must go red.
            probe = subprocess.run(
                [sys.executable, str(ROOT / "scripts/agent-role.py"), "show"],
                cwd=ROOT, capture_output=True, text=True, check=False, env=env,
            )
            self.assertEqual(
                probe.returncode, 3,
                "precondition failed: a role still resolves under the empty "
                f"GIT_DIR, so this test asserts nothing. show said: "
                f"{probe.stdout.strip()!r}")
            self.assertIn("UNDECLARED", probe.stdout)

            result = subprocess.run(
                ["bash", str(ROOT / "scripts/agent-preflight.sh"), "--role-only"],
                cwd=ROOT, capture_output=True, text=True, check=False, env=env,
            )
        self.assertNotEqual(
            result.returncode, 0,
            "preflight exited 0 with no resolvable role; the REQUIRE_ROLE "
            "default is off however it is spelled")
        self.assertIn("role-undeclared", result.stdout + result.stderr)

    def test_role_only_is_not_mistakable_for_a_full_preflight(self):
        # --role-only exists so a test can execute the gate without recursing.
        # It must never read as a green preflight, or it becomes the opt-out
        # --no-require-role was demoted from.
        self.assertIn("--role-only", self.TEXT)
        run = subprocess.run(
            ["bash", str(ROOT / "scripts/agent-preflight.sh"), "--role-only"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        output = run.stdout + run.stderr
        self.assertIn("NOT a full preflight", output)
        self.assertNotIn("All gates green", output)


class EnvSetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".env"
        self._real = onboard.ENV_FILE
        onboard.ENV_FILE = self.env
        # cmd_env_set reports on stdout and refuses on stderr. Both go to a
        # buffer so the suite's own output stays clean -- but the buffers are
        # KEPT and asserted on, because a refusal that returns 2 with an empty
        # explanation would otherwise pass every test in this class.
        self.out, self.err = io.StringIO(), io.StringIO()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(self.out))
        stack.enter_context(contextlib.redirect_stderr(self.err))
        self.addCleanup(stack.close)

    def tearDown(self):
        onboard.ENV_FILE = self._real
        shutil.rmtree(self.tmp)

    def test_unknown_key_is_refused(self):
        # Never invent a key: a typo'd name would sit in .env doing nothing
        # while the gate that wanted it stays mysteriously PENDING.
        self.assertEqual(onboard.cmd_env_set("NOT_A_REAL_KEY=/x"), 2)
        self.assertFalse(self.env.exists())
        # An exit code alone gets routed around, and every refusal here is a
        # human's typo. Name the offending key and the legal ones.
        self.assertIn("NOT_A_REAL_KEY", self.err.getvalue())
        self.assertIn(onboard.ENV_KEYS[0], self.err.getvalue())

    def test_missing_pair_is_refused(self):
        self.assertEqual(onboard.cmd_env_set("VLLM_ORACLE"), 2)
        self.assertFalse(self.env.exists())
        self.assertIn("KEY=VALUE", self.err.getvalue())

    def test_first_write_seeds_from_the_example(self):
        self.assertEqual(onboard.cmd_env_set(f"{onboard.ENV_KEYS[0]}=/oracle"), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{onboard.ENV_KEYS[0]}=/oracle", text)
        # every other declared key survives, so nothing is silently dropped
        for key in onboard.ENV_KEYS:
            self.assertIn(key, text)

    def test_second_write_updates_in_place_without_duplicating(self):
        key = onboard.ENV_KEYS[0]
        onboard.cmd_env_set(f"{key}=/first")
        onboard.cmd_env_set(f"{key}=/second")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{key}=/second", text)
        self.assertNotIn("/first", text)
        self.assertEqual(sum(1 for l in text.splitlines() if l.startswith(f"{key}=")), 1)

    def test_other_keys_are_not_disturbed(self):
        a, b = onboard.ENV_KEYS[0], onboard.ENV_KEYS[1]
        onboard.cmd_env_set(f"{a}=/aaa")
        onboard.cmd_env_set(f"{b}=/bbb")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{a}=/aaa", text)
        self.assertIn(f"{b}=/bbb", text)

    def test_the_flag_is_wired_into_main(self):
        # Beyond the brief. Every test above calls cmd_env_set() directly, so
        # deleting --env-set or its dispatch in main() leaves all five green
        # while the only entry point an agent actually types either dies in
        # argparse or silently prints a probe and records nothing.
        key = onboard.ENV_KEYS[0]
        self.assertEqual(onboard.main(["--env-set", f"{key}=/wired"]), 0)
        self.assertIn(f"{key}=/wired", self.env.read_text(encoding="utf-8"))
        self.assertEqual(onboard.main(["--env-set", "NOT_A_REAL_KEY=/x"]), 2)
        # An empty argument is a malformed write, not "no flag given": a
        # truthiness dispatch falls through to the probe and exits 0 having
        # recorded nothing, which is the silent no-op this command must not do.
        self.assertEqual(onboard.main(["--env-set", ""]), 2)

    def test_an_empty_value_is_accepted_and_still_reads_as_unset(self):
        # Beyond the brief, and the rule the whole command serves: unanswered
        # means EMPTY, and empty means the gates that need it stay PENDING.
        # A writer that refused an empty value would push its caller toward
        # inventing one from a path, a username or another developer's setup,
        # which is the failure this spec exists to prevent. Clearing a value
        # must also be possible, or a wrong answer is unretractable.
        key = onboard.ENV_KEYS[0]
        self.assertEqual(onboard.cmd_env_set(f"{key}=/somewhere"), 0)
        self.assertEqual(onboard.cmd_env_set(f"{key}="), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"\n{key}=\n", f"\n{text}")
        self.assertIn(key, onboard.env_state_from_text(text)[1])

    def test_no_line_separator_in_a_value_can_forge_a_second_line(self):
        # Beyond the brief. The value is whatever the interview answer was, and
        # it is the one field no check looks at. A separator inside it appends a
        # whole extra .env line that no key check ever saw -- the silent clobber
        # this task exists to rule out, arriving through the unvalidated field.
        #
        # EVERY separator, not just "\n": str.splitlines() is what the reader
        # and the rewrite both use, and it breaks on ten. Guarding two of them
        # let a "\v" or a U+2028 smuggle a forged pair past the key check, after
        # which the probe reported the forged key as SET. This list IS the
        # separator set; a guard that enumerates characters again fails here.
        key, victim = onboard.ENV_KEYS[0], onboard.ENV_KEYS[3]
        for separator in ("\n", "\r", "\r\n", "\v", "\f",
                          "\x1c", "\x1d", "\x1e", "\x85", " ", " "):
            with self.subTest(separator=repr(separator)):
                pair = f"{key}=/ok{separator}{victim}=/forged"
                self.assertEqual(onboard.cmd_env_set(pair), 2)
                self.assertFalse(self.env.exists())
        # The property, stated the way the guard states it: what is written
        # must survive the round trip through the reader that parses it back.
        self.assertEqual(onboard.cmd_env_set(f"{key}=/ok {victim}=/forged"), 2)
        self.assertIn("line separator", self.err.getvalue())

    def test_a_trailing_duplicate_key_cannot_survive_the_write(self):
        # Beyond the brief. A hand-maintained .env routinely carries an override
        # appended at the bottom, and both this reader and the documented
        # `set -a; . ./.env` loader take the LAST assignment. Updating only the
        # first match left the file changed, the exit code 0 and the message
        # reassuring while the EFFECTIVE value never moved -- the silent no-op
        # this command exists to rule out, arriving from the other side.
        key, other = onboard.ENV_KEYS[0], onboard.ENV_KEYS[5]
        self.env.write_text(
            f"{key}=/one\n{other}=h\n{key}=/override\n", encoding="utf-8")
        self.assertEqual(onboard.cmd_env_set(f"{key}=/new"), 0)
        lines = self.env.read_text(encoding="utf-8").splitlines()
        effective = [l.split("=", 1)[1] for l in lines if l.startswith(f"{key}=")][-1]
        self.assertEqual(effective, "/new")
        self.assertEqual(sum(1 for l in lines if l.startswith(f"{key}=")), 1)
        self.assertIn(f"{other}=h", lines)  # the unrelated line keeps its place

    def test_an_unreadable_env_is_refused_and_not_a_traceback(self):
        # Beyond the brief. env_state treats an existing-but-unreadable .env as
        # its own third case; the WRITER has to agree, or the one command an
        # agent is told to run dies in a bare traceback that names no fix.
        self.env.mkdir()  # exists(), but read_text() raises IsADirectoryError
        self.assertEqual(onboard.cmd_env_set(f"{onboard.ENV_KEYS[0]}=/x"), 2)
        self.assertIn("cannot be read", self.err.getvalue())

    def test_a_value_needing_quotes_reads_the_same_to_both_readers(self):
        # Beyond the brief, same family as the separator finding: .env.example
        # documents the loader as `set -a; . ./.env`, so the file is shell as
        # well as data. An unquoted "/pa th" makes that loader run `th` and
        # leave the variable EMPTY while this probe's parser reports it PRESENT
        # -- a gate that stays PENDING with the surface insisting it is set.
        key = onboard.ENV_KEYS[0]
        self.assertEqual(onboard.cmd_env_set(f"{key}=/pa th"), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertNotIn(key, onboard.env_state_from_text(text)[1])  # probe: set
        sourced = subprocess.run(
            ["sh", "-c", f'set -a; . "$1"; set +a; printf %s "${key}"', "sh",
             str(self.env)],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(sourced.stdout, "/pa th")  # loader: the SAME value
        self.assertEqual(sourced.stderr, "")
        # and an empty value must still round-trip as UNSET, not as the literal
        # two-character '' that shlex.quote would otherwise produce.
        self.assertEqual(onboard.cmd_env_set(f"{key}="), 0)
        self.assertIn(key, onboard.env_state_from_text(
            self.env.read_text(encoding="utf-8"))[1])


if __name__ == "__main__":
    unittest.main()
