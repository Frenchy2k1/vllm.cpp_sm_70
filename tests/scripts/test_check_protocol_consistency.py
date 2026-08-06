#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-protocol-consistency.py.

The gate exists to catch prose drifting away from the checker that enforces it,
so the mutations here are the real historical failure: a document that still
names README.md as a checkpoint surface, and a document whose contract silently
disagrees with scripts/check-doc-checkpoint.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import shutil
import sys
import tempfile
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


consistency = _load("protocol_consistency", "scripts/check-protocol-consistency.py")

EXPECTED = ("docs/STATUS.md", "docs/BENCHMARKS.md", "docs/FEATURES.md")


def document(*paths: str) -> str:
    rows = "\n".join(f"| `{path}` | every checkpoint |" for path in paths)
    return "\n".join(
        [
            "# Some normative document",
            "",
            consistency.BEGIN,
            "| Public surface | Owed by |",
            "|---|---|",
            rows,
            consistency.END,
            "",
            "Trailing prose.",
        ]
    )


class ContractParsing(unittest.TestCase):
    def test_extracts_paths_in_order(self) -> None:
        self.assertEqual(
            consistency.contract_paths(document(*EXPECTED)), list(EXPECTED)
        )

    def test_absent_block_is_none(self) -> None:
        self.assertIsNone(consistency.contract_paths("# No contract here"))

    def test_end_before_begin_is_none(self) -> None:
        text = f"{consistency.END}\n| `docs/STATUS.md` |\n{consistency.BEGIN}"
        self.assertIsNone(consistency.contract_paths(text))


class Mutations(unittest.TestCase):
    def test_baseline_passes(self) -> None:
        self.assertEqual(
            consistency.document_errors("doc", document(*EXPECTED), EXPECTED), []
        )

    def test_missing_block_is_rejected(self) -> None:
        errors = consistency.document_errors("doc", "# nothing", EXPECTED)
        self.assertTrue(any("missing the doc-obligation contract" in e for e in errors))

    def test_readme_in_contract_is_rejected_by_name(self) -> None:
        """The exact historical regression: README named as a checkpoint."""
        text = document("README.md", "docs/BENCHMARKS.md", "docs/FEATURES.md")
        errors = consistency.document_errors("doc", text, EXPECTED)
        self.assertTrue(any("README.md" in e and "landing page" in e for e in errors))

    def test_dropped_surface_is_rejected(self) -> None:
        text = document("docs/STATUS.md", "docs/BENCHMARKS.md")
        errors = consistency.document_errors("doc", text, EXPECTED)
        self.assertTrue(any("enforces" in e for e in errors))

    def test_reordered_surfaces_are_rejected(self) -> None:
        text = document("docs/BENCHMARKS.md", "docs/STATUS.md", "docs/FEATURES.md")
        self.assertNotEqual(
            consistency.document_errors("doc", text, EXPECTED), []
        )

    def test_extra_surface_is_rejected(self) -> None:
        text = document(*EXPECTED, "docs/USAGE.md")
        self.assertNotEqual(consistency.document_errors("doc", text, EXPECTED), [])


class LiveTree(unittest.TestCase):
    def test_expected_surfaces_come_from_the_checker(self) -> None:
        self.assertEqual(consistency.obligated_surfaces(), EXPECTED)

    def test_repository_contract_is_consistent(self) -> None:
        self.assertEqual(consistency.main(), 0)

    def test_every_contract_document_exists(self) -> None:
        for name in consistency.CONTRACT_DOCUMENTS:
            self.assertTrue((ROOT / name).exists(), name)


class InterviewBlockTests(unittest.TestCase):
    def test_workflow_carries_the_role_interview(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn(consistency.INTERVIEW_MARKER, text)
        self.assertIn("read-only", text)
        self.assertIn("claim helper --row", text)

    def test_checker_rejects_a_workflow_without_the_interview(self):
        # The mutation this gate exists to catch: the gate ships, the prose
        # does not, and agents never learn the precondition.
        errors = consistency.interview_errors("# workflow\n\nno interview here\n")
        self.assertTrue(errors)

    def test_every_declarable_role_is_named_in_the_interview(self):
        # INTERVIEW_REQUIRED is a hand-written tuple, so emptying or narrowing
        # it would leave every other assertion in this class green while the
        # checker quietly stopped looking at the answers. Bind it to the roles
        # agent-role.py actually accepts instead: a fourth answer must reach the
        # prose, and dropping one from the checker is a red build.
        role = _load("agent_role_for_interview", "scripts/agent-role.py")
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        for name in role.DECLARABLE:
            with self.subTest(role=name):
                self.assertIn(f"claim {name}", text)
                self.assertTrue(
                    any(f"claim {name}" in n for n in consistency.INTERVIEW_REQUIRED),
                    f"INTERVIEW_REQUIRED does not cover 'claim {name}'",
                )

    def test_each_required_answer_is_pinned_individually(self):
        # A block that exists but has lost one of the three answers is the
        # likelier drift, and the marker alone would not see it.
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        for needle in consistency.INTERVIEW_REQUIRED:
            with self.subTest(needle=needle):
                self.assertTrue(consistency.interview_errors(text.replace(needle, "")))


class InterviewWiring(unittest.TestCase):
    """The checker must CALL interview_errors, not merely define it.

    Every assertion above exercises the function directly, so a `main()` that
    never calls it leaves them all green while the gate enforces nothing --
    which is the exact shape of the drift this file exists to catch.
    """

    STRIP = re.compile(
        r"<!-- role-interview:begin -->.*?<!-- role-interview:end -->\n?", re.S
    )

    @contextlib.contextmanager
    def _tree(self, workflow_text: str):
        """Run consistency.main() against a copy of the repo's own documents."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / ".agents").mkdir()
            shutil.copy(
                ROOT / "scripts/check-doc-checkpoint.py",
                root / "scripts/check-doc-checkpoint.py",
            )
            shutil.copy(ROOT / "AGENTS.md", root / "AGENTS.md")
            (root / ".agents/workflow.md").write_text(workflow_text, encoding="utf-8")
            saved, consistency.ROOT = consistency.ROOT, root
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    yield lambda: (consistency.main(), out.getvalue(), err.getvalue())
            finally:
                consistency.ROOT = saved

    def test_faithful_copy_passes(self):
        """Positive control: the temp tree itself is not what fails below."""
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn(consistency.INTERVIEW_MARKER, text)
        with self._tree(text) as run:
            code, _, err = run()
        self.assertEqual(code, 0, err)

    def test_main_fails_when_the_interview_is_deleted(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        stripped = self.STRIP.sub("", text)
        self.assertNotIn(consistency.INTERVIEW_MARKER, stripped)
        self.assertNotEqual(stripped, text, "the strip pattern matched nothing")
        with self._tree(stripped) as run:
            code, _, err = run()
        self.assertEqual(code, 1)
        self.assertIn("role-interview", err)


if __name__ == "__main__":
    unittest.main()
