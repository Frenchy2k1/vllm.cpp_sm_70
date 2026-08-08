#!/usr/bin/env python3
"""Behavior checks for the mechanically proven helper queue."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ready = _load("ready_for_helper_tests", "scripts/ready-for-helper.py")


class Row:
    def __init__(
        self,
        root: Path,
        *,
        item_id: str = "ENG-FOO",
        state: str = "READY",
        dependencies: str = "None",
        spec: str = "[spec](specs/foo.md)",
    ) -> None:
        self.path = root / ".agents/engine-matrix.md"
        self.item_id = item_id
        self.state = state
        self.header = ("id", "dependencies", "spike spec", "state")
        self.cells = (item_id, dependencies, spec, state)

    def field(self, name: str) -> str:
        aliases = {"spec": "spike spec", "dependencies": "dependencies"}
        key = aliases.get(name, name)
        try:
            return self.cells[self.header.index(key)]
        except ValueError:
            return ""


class RepositoryFixture:
    def __init__(self, hardware: str = "CPU") -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".agents/specs").mkdir(parents=True)
        (self.root / ".agents").mkdir(exist_ok=True)
        (self.root / ".agents/engine-matrix.md").write_text("matrix\n", encoding="utf-8")
        (self.root / "tests").mkdir()
        (self.root / "tests/gate.py").write_text(
            "#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8"
        )
        (self.root / "tests/mutation.py").write_text(
            "#!/usr/bin/env python3\nraise SystemExit(9)\n", encoding="utf-8"
        )
        self.write_spec(hardware=hardware)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=self.root, check=True)

    def write_spec(
        self,
        *,
        hardware: str = "CPU",
        gate: list[str] | str | None = None,
        mutation: list[str] | str | None = None,
    ) -> None:
        contract = {
            "gate": gate if gate is not None else ["python3", "tests/gate.py"],
            "mutation": mutation
            if mutation is not None
            else ["python3", "tests/mutation.py"],
        }
        (self.root / ".agents/specs/foo.md").write_text(
            "# Gates\n\n"
            "<!-- helper-readiness:v1\n"
            + json.dumps(contract, separators=(",", ":"))
            + "\n-->\n\n"
            f"Gate hardware: {hardware}.\n",
            encoding="utf-8",
        )

    def commit(self, message: str = "fixture") -> None:
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=self.root, check=True)

    def close(self) -> None:
        self.temp.cleanup()


class ReadinessProof(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = RepositoryFixture()
        self.row = Row(self.repo.root)
        self.states = {"ENG-FOO": "READY", "KV-BAR": "DONE"}

    def tearDown(self) -> None:
        self.repo.close()

    def evaluate(self, row: Row | None = None, **kwargs) -> list[str]:
        return ready.evaluate(
            row or self.row,
            live_claims=kwargs.pop("live_claims", set()),
            known_tasks=kwargs.pop("known_tasks", set(self.states)),
            task_states=kwargs.pop("task_states", self.states),
            root=self.repo.root,
            base="HEAD",
            **kwargs,
        )

    def test_complete_proof_is_pickable(self) -> None:
        self.assertEqual(self.evaluate(), [])

    def test_each_lifecycle_state_is_enforced(self) -> None:
        for state in ("INVENTORIED", "SPIKE", "ACTIVE", "GATING", "BLOCKED", "DONE"):
            with self.subTest(state=state):
                errors = self.evaluate(Row(self.repo.root, state=state))
                self.assertTrue(any("lifecycle" in error for error in errors), errors)

    def test_unknown_task_uses_the_same_shared_adapter_boundary(self) -> None:
        errors = self.evaluate(known_tasks={"KV-BAR"})
        self.assertIn("unknown task ID", errors)

    def test_spec_must_be_committed_and_reachable_from_base(self) -> None:
        path = self.repo.root / ".agents/specs/foo.md"
        path.write_text("# rewritten worktree file without a contract\n", encoding="utf-8")
        self.assertEqual(self.evaluate(), [])
        subprocess.run(["git", "rm", "--cached", "-q", ".agents/specs/foo.md"], cwd=self.repo.root, check=True)
        subprocess.run(["git", "commit", "-qm", "remove spec from base"], cwd=self.repo.root, check=True)
        errors = self.evaluate()
        self.assertTrue(any("base-reachable committed spec" in error for error in errors), errors)

    def test_spec_must_be_a_regular_blob_and_base_must_resolve_to_a_commit(self) -> None:
        spec = self.repo.root / ".agents/specs/foo.md"
        spec.unlink()
        os.symlink("../../tests/gate.py", spec)
        self.repo.commit("symlink spec")
        self.assertIn("no base-reachable committed spec", self.evaluate())
        errors = ready.evaluate(
            self.row,
            live_claims=set(),
            known_tasks=set(self.states),
            task_states=self.states,
            root=self.repo.root,
            base="refs/heads/does-not-exist",
        )
        self.assertTrue(any("base commit" in error for error in errors), errors)

    def test_gate_needs_executable_command_and_failing_mutation(self) -> None:
        cases = (
            ({"gate": "python3 tests/gate.py", "mutation": ["python3", "tests/mutation.py"]}, "structured"),
            ({"gate": ["python3", "-c", "pass"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["python3", "-cpass"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["python3", "--eval=pass"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["/usr/bin/python3", "tests/gate.py"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["python3", "../gate.py"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["python3", "--config=../gate.py"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["python3", "--config=/etc/passwd"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["MALICE=1", "tests/gate.py"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["python3", "tests/gate.py;echo"], "mutation": ["python3", "tests/mutation.py"]}, "unsafe gate"),
            ({"gate": ["missing-program", "tests/gate.py"], "mutation": ["python3", "tests/mutation.py"]}, "gate executable"),
        )
        spec = self.repo.root / ".agents/specs/foo.md"
        for contract, expected in cases:
            with self.subTest(contract=contract):
                spec.write_text(
                    "# Gates\n\n<!-- helper-readiness:v1\n"
                    + json.dumps(contract, separators=(",", ":"))
                    + "\n-->\n\nGate hardware: CPU.\n",
                    encoding="utf-8",
                )
                self.repo.commit("bad contract")
                errors = self.evaluate()
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_gate_and_mutation_are_executed_from_exact_base_not_worktree(self) -> None:
        gate = self.repo.root / "tests/gate.py"
        mutation = self.repo.root / "tests/mutation.py"
        gate.write_text("raise SystemExit(7)\n", encoding="utf-8")
        mutation.write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.assertEqual(self.evaluate(), [])

        self.repo.commit("make exact base fail readiness")
        errors = self.evaluate()
        self.assertTrue(any("gate exited 7" in error for error in errors), errors)
        self.assertTrue(any("mutation unexpectedly exited 0" in error for error in errors), errors)

    def test_gate_runs_with_sanitized_environment_and_timeout(self) -> None:
        gate = self.repo.root / "tests/gate.py"
        gate.write_text(
            "import os, time\n"
            "assert 'HOST_SECRET_FOR_TEST' not in os.environ\n"
            "assert set(os.environ) <= {'HOME','LANG','LC_ALL','PATH','PYTHONHASHSEED','TMPDIR'}\n"
            "time.sleep(2)\n",
            encoding="utf-8",
        )
        self.repo.commit("slow sanitized gate")
        with mock.patch.dict(os.environ, {"HOST_SECRET_FOR_TEST": "must-not-leak"}), mock.patch.object(
            ready, "RUN_TIMEOUT_SECONDS", 0.05
        ):
            errors = self.evaluate()
        self.assertTrue(any("gate could not execute" in error and "timed out" in error for error in errors), errors)

    def test_missing_or_nonregular_base_argv_paths_are_rejected(self) -> None:
        for argument in ("tests/missing.py", "tests"):
            with self.subTest(argument=argument):
                self.repo.write_spec(gate=["python3", argument])
                self.repo.commit("bad argv path")
                self.assertTrue(
                    any("regular base file" in error for error in self.evaluate())
                )

    def test_hardware_must_say_cpu_or_name_exact_hardware(self) -> None:
        self.repo.write_spec(hardware="GPU")
        self.repo.commit("generic hardware")
        self.assertTrue(any("exact gate hardware" in error for error in self.evaluate()))
        self.repo.write_spec(hardware="NVIDIA GB10 sm_121")
        self.repo.commit("exact hardware")
        self.assertEqual(self.evaluate(), [])

    def test_dependencies_must_parse_and_be_done(self) -> None:
        self.assertEqual(self.evaluate(Row(self.repo.root, dependencies="`KV-BAR`")), [])
        pending_states = {**self.states, "KV-BAR": "ACTIVE"}
        errors = self.evaluate(Row(self.repo.root, dependencies="`KV-BAR`"), task_states=pending_states)
        self.assertIn("dependency KV-BAR is ACTIVE, not satisfied", errors)
        errors = self.evaluate(Row(self.repo.root, dependencies="after somebody reviews it"))
        self.assertIn("dependencies are not fully parsed", errors)

    def test_live_claim_excludes_task(self) -> None:
        self.assertIn("already claimed by an open PR", self.evaluate(live_claims={"ENG-FOO"}))


class QueueAndCli(unittest.TestCase):
    def test_queue_consumes_claim_views_shared_known_task_adapter(self) -> None:
        row = Row(ROOT)
        with mock.patch.object(
            ready.claim_view, "canonical_task_rows", return_value=[row]
        ) as rows:
            pickable, reasons = ready.queue([], root=ROOT, base="HEAD")
        rows.assert_called_once_with(ROOT)
        self.assertNotIn("unknown task ID", reasons)

    def test_print_queue_rejects_remote_unavailability(self) -> None:
        with mock.patch.object(ready.claim_view, "fetch_prs", side_effect=ready.claim_view.RemoteUnverified("offline")):
            self.assertEqual(ready.main(["--check-live"]), ready.claim_view.REMOTE_UNVERIFIED_EXIT)

    def test_bare_check_is_explicitly_local_only_and_never_advertises_queue(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            ready.claim_view, "fetch_prs", side_effect=AssertionError("network")
        ), mock.patch.object(ready, "queue", side_effect=AssertionError("queue")), redirect_stdout(output):
            self.assertEqual(ready.main(["--check"]), 0)
        self.assertIn("LOCAL_ONLY", output.getvalue())
        self.assertIn("no helper queue was advertised", output.getvalue())

    def test_fixture_claims_are_consumed_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "prs.json"
            fixture.write_text(
                json.dumps(
                    {"expected": {"repository": "owner/repo", "base": "HEAD"}, "prs": []}
                ),
                encoding="utf-8",
            )
            with mock.patch.object(ready, "queue", return_value=([], {})) as queue:
                self.assertEqual(
                    ready.main(["--check-live", "--pr-json", str(fixture)]), 0
                )
            queue.assert_called_once()

    def test_deprecated_check_rejects_fixture_and_live_options(self) -> None:
        for args in (
            ["--check", "--pr-json", "claims.json"],
            ["--check", "--check-live"],
            ["--check", "--base", "HEAD"],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                ready.main(args)

    def test_explicit_local_check_is_available_and_never_fetches(self) -> None:
        with mock.patch.object(
            ready.claim_view, "fetch_prs", side_effect=AssertionError("network")
        ):
            self.assertEqual(ready.main(["--check-local"]), 0)

    def test_duplicate_row_ids_are_rejected_before_state_indexing(self) -> None:
        with mock.patch.object(
            ready.claim_view,
            "canonical_task_rows",
            side_effect=ValueError("duplicate task ID ENG-FOO at a:1 and b:2"),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate task ID ENG-FOO"):
                ready._rows(ROOT)


if __name__ == "__main__":
    unittest.main()
