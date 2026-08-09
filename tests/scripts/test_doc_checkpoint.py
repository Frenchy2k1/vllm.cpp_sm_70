#!/usr/bin/env python3
"""Mutation tests for scripts/check-doc-checkpoint.py.

The rewritten gate asks "did a row change lifecycle state", not "which directory
did you touch". These tests pin BOTH halves of that:

  * the obligation still fires on a real claim (a lifecycle move, a measurement),
    so a capability cannot land undocumented;
  * it does NOT fire on ordinary code edits, which is the defect that produced
    16 of the last 20 red CI runs and six hardcoded escape-hatch path sets.

The second half matters as much as the first. A gate that over-fires gets
worked around, and the workarounds were what made the old gate unrepairable.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "check_doc_checkpoint", ROOT / "scripts/check-doc-checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


ROW_TABLE = """# Kernel matrix

| ID | Item | State | Owner |
|---|---|---|---|
| `KERNEL-ALPHA` | alpha | `{alpha}` | ops |
| `KERNEL-BETA` | beta | `{beta}` | ops |
"""


class RowStateParsing(unittest.TestCase):
    def test_row_states_are_read_from_the_keyed_table(self):
        states = checker.row_states(ROW_TABLE.format(alpha="READY", beta="DONE"))
        self.assertEqual(states, {"KERNEL-ALPHA": "READY", "KERNEL-BETA": "DONE"})

    def test_prose_states_in_evidence_columns_do_not_win(self):
        """The state cell is the LAST backticked state on the row."""
        line = "| `KERNEL-X` | supersedes the `DONE` attempt | `ACTIVE` | ops |\n"
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "ACTIVE"})

    def test_non_row_lines_are_ignored(self):
        self.assertEqual(checker.row_states("| not a row | `DONE` |\n"), {})


class LifecycleTrigger(unittest.TestCase):
    """classify() via a stubbed blob reader, so no git fixture is needed."""

    def classify(self, paths, before_text, after_text):
        original = checker.blob
        checker.blob = lambda rev, path: before_text if rev == "BEFORE" else after_text
        try:
            return checker.classify(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_lifecycle_move_is_a_claim(self):
        classes, reasons = self.classify(
            [".agents/kernel-matrix.md"],
            ROW_TABLE.format(alpha="READY", beta="DONE"),
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
        )
        self.assertIn("lifecycle", classes)
        self.assertTrue(any("READY -> DONE" in r for r in reasons), reasons)

    def test_editing_a_row_without_moving_its_state_is_not_a_claim(self):
        table = ROW_TABLE.format(alpha="READY", beta="DONE")
        classes, _ = self.classify(
            [".agents/kernel-matrix.md"], table, table.replace("alpha", "alpha prime")
        )
        self.assertNotIn("lifecycle", classes)

    def test_a_new_row_only_counts_once_it_claims_something(self):
        one_row = "| `KERNEL-ALPHA` | alpha | `{}` | ops |\n"
        for state, expected in (("TODO", False), ("READY", False), ("DONE", True)):
            with self.subTest(state=state):
                classes, _ = self.classify(
                    [".agents/kernel-matrix.md"], "", one_row.format(state)
                )
                self.assertEqual("lifecycle" in classes, expected)

    def test_a_measurement_is_a_claim(self):
        classes, reasons = self.classify([".agents/benchmark-record.md"], "", "")
        self.assertIn("lifecycle", classes)
        self.assertTrue(any("measurement" in r for r in reasons), reasons)


class ObligationsFire(unittest.TestCase):
    def errors(self, paths, before_text="", after_text=""):
        original = checker.blob
        checker.blob = lambda rev, path: before_text if rev == "BEFORE" else after_text
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    MOVED = (
        ROW_TABLE.format(alpha="READY", beta="DONE"),
        ROW_TABLE.format(alpha="DONE", beta="DONE"),
    )

    def test_lifecycle_move_demands_status_benchmarks_and_now(self):
        errors = self.errors([".agents/kernel-matrix.md"], *self.MOVED)
        self.assertTrue(errors)
        for surface in ("docs/STATUS.md", "docs/BENCHMARKS.md", ".agents/NOW.md"):
            self.assertIn(surface, errors[0])

    def test_each_required_surface_is_individually_load_bearing(self):
        required = ["docs/STATUS.md", "docs/BENCHMARKS.md", ".agents/NOW.md"]
        for omitted in required:
            with self.subTest(omitted=omitted):
                paths = [".agents/kernel-matrix.md"] + [
                    s for s in required if s != omitted
                ]
                errors = self.errors(paths, *self.MOVED)
                self.assertTrue(errors, f"omitting {omitted} was not caught")
                self.assertIn(omitted, errors[0])

    def test_a_complete_checkpoint_passes(self):
        paths = [
            ".agents/kernel-matrix.md",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
            ".agents/NOW.md",
        ]
        self.assertEqual(self.errors(paths, *self.MOVED), [])


class ObligationsDoNotOverfire(unittest.TestCase):
    """The regression this rewrite exists to prevent."""

    def errors(self, paths):
        original = checker.blob
        checker.blob = lambda rev, path: ""
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_editing_source_owes_nothing(self):
        for path in (
            "src/vt/cuda/cuda_backend.cu",
            "tests/vt/test_backend.cpp",
            "scripts/check-agent-record.py",
            "tools/bench/serve.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.errors([path]), [])

    def test_a_governance_change_is_not_a_feature_checkpoint(self):
        """Staging this very design must not demand a benchmark update."""
        self.assertEqual(
            self.errors(
                [
                    "AGENTS.md",
                    ".agents/workflow.md",
                    "scripts/check-doc-checkpoint.py",
                    "tests/scripts/test_doc_checkpoint.py",
                    "docs/superpowers/specs/2026-08-09-policy-simplification-design.md",
                ]
            ),
            [],
        )

    def test_no_hardcoded_escape_hatches_remain(self):
        """Six exact-path-set exemptions were fossils of the wrong trigger."""
        source = (ROOT / "scripts/check-doc-checkpoint.py").read_text(encoding="utf-8")
        for fossil in (
            "POLICY_CONSOLIDATION_FILES",
            "POLICY_CUTOVER_FILES",
            "PR_SIZE_BOOTSTRAP_FILES",
            "PENDING_PR_RANGE_FILES",
            "SYNTHETIC_MERGE_RANGE_FILES",
            "CLAIM_CUTOVER_FILES",
        ):
            self.assertNotIn(fossil, source)


class SupportSurfaces(unittest.TestCase):
    def errors(self, paths):
        original = checker.blob
        checker.blob = lambda rev, path: ""
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_a_new_model_owes_the_feature_surface(self):
        errors = self.errors(["src/vllm/model_executor/models/newmodel.cpp"])
        self.assertTrue(errors)
        self.assertIn("docs/FEATURES.md", errors[0])

    def test_a_user_entrypoint_owes_usage(self):
        errors = self.errors(["include/vllm.h"])
        self.assertTrue(errors)
        self.assertIn("docs/USAGE.md", errors[0])

    def test_readme_needs_a_landing_source(self):
        errors = self.errors(["README.md"])
        self.assertTrue(errors)
        self.assertIn("landing source", errors[0])

    def test_readme_with_a_landing_source_passes(self):
        self.assertEqual(self.errors(["README.md", ".agents/mission.md"]), [])

    def test_a_coedited_projection_never_licenses_readme_churn(self):
        """Deliberate: STATUS.md changing is not permission to rewrite README."""
        errors = self.errors(["README.md", "docs/STATUS.md"])
        self.assertTrue(errors)
        self.assertIn("landing source", errors[0])

    def test_a_landing_source_permits_but_does_not_demand_readme(self):
        self.assertEqual(self.errors([".agents/mission.md"]), [])


if __name__ == "__main__":
    unittest.main()
