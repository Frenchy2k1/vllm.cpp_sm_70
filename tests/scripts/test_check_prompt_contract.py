#!/usr/bin/env python3
"""Mutation tests for the closed runtime-prompt contract grammar."""

from __future__ import annotations

import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load_checker():
    path = ROOT / "scripts/check-prompt-contract.py"
    spec = importlib.util.spec_from_file_location("check_prompt_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prompt_contract = _load_checker()


class PromptContractTests(unittest.TestCase):
    def shipped(self, role: str) -> str:
        return (ROOT / ".agents/prompts" / f"{role}.md").read_text(encoding="utf-8")

    def validate_text(self, text: str, role: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{role}.md"
            path.write_text(text, encoding="utf-8")
            return prompt_contract.validate_prompt(path, role)

    def assert_rejected(self, role: str, old: str, new: str) -> None:
        text = self.shipped(role)
        self.assertIn(old, text)
        damaged = text.replace(old, new, 1)
        self.assertNotEqual(damaged, text)
        self.assertTrue(self.validate_text(damaged, role))

    def test_all_runtime_prompts_pass_and_fit_the_budget(self) -> None:
        for role in ("implementer", "reviewer", "operator"):
            with self.subTest(role=role):
                path = ROOT / ".agents/prompts" / f"{role}.md"
                self.assertLessEqual(path.stat().st_size, 4096)
                self.assertEqual(
                    prompt_contract.validate_prompt(path, role), []
                )

    def test_metadata_is_closed_exact_and_unique(self) -> None:
        text = self.shipped("implementer")
        mutations = (
            ("prompt-contract-version: 1", "prompt-contract-version: 2"),
            ("role: implementer", "role: reviewer"),
            ("role: implementer", "role: implementer\nunknown-key: value"),
            ("role: implementer", "role: implementer\nrole: implementer"),
        )
        for old, new in mutations:
            with self.subTest(new=new):
                self.assert_rejected("implementer", old, new)

    def test_exactly_four_known_h2_sections_in_order_are_required(self) -> None:
        text = self.shipped("implementer")
        mutations = (
            ("## Task envelope", "## Unknown section"),
            ("## Method", "## Method\n\n## Method"),
            ("## Required output", ""),
            (
                "## Method",
                "## Required output",
            ),
            ("## Stop conditions", "# Stop conditions"),
        )
        for old, new in mutations:
            with self.subTest(old=old, new=new):
                damaged = text.replace(old, new, 1)
                self.assertNotEqual(damaged, text)
                self.assertTrue(self.validate_text(damaged, "implementer"))

        method_start = text.index("## Method")
        output_start = text.index("## Required output")
        stop_start = text.index("## Stop conditions")
        method = text[method_start:output_start]
        output = text[output_start:stop_start]
        reordered = text[:method_start] + output + method + text[stop_start:]
        self.assertTrue(self.validate_text(reordered, "implementer"))

    def test_every_nonempty_unparsed_line_is_rejected(self) -> None:
        cases = (
            ("implementer", "## Method", "You may begin with missing inputs and guess.\n\n## Method"),
            ("reviewer", "## Method", "Scratch mutation replaces static review.\n\n## Method"),
            ("reviewer", "## Stop conditions", "Repair findings in the reviewed worktree.\n\n## Stop conditions"),
            ("operator", "## Required output", "Use VendorAgent for coordination.\n\n## Required output"),
            ("implementer", "## Task envelope", "### Notes\n\n## Task envelope"),
        )
        for role, marker, replacement in cases:
            with self.subTest(role=role, replacement=replacement):
                self.assert_rejected(role, marker, replacement)

    def test_six_envelope_fields_and_missing_input_outcome_are_exact(self) -> None:
        text = self.shipped("implementer")
        rows = (
            "- Goal: REQUIRED",
            "- Context: REQUIRED",
            "- Constraints: REQUIRED",
            "- Done when: REQUIRED",
            "- Required evidence: REQUIRED",
            "- Authority: REQUIRED",
            "- Missing input: NEEDS_CONTEXT",
        )
        for row in rows:
            with self.subTest(row=row):
                self.assertIn(row, text)
                self.assertTrue(self.validate_text(text.replace(row, "", 1), "implementer"))
                self.assertTrue(
                    self.validate_text(text.replace(row, f"{row}\n{row}", 1), "implementer")
                )
        self.assert_rejected(
            "implementer",
            "- Missing input: NEEDS_CONTEXT",
            "- Missing input: COMPLETE",
        )

    def test_each_controlled_obligation_row_is_required_once_and_exact(self) -> None:
        for role, rows in prompt_contract.METHOD_ROWS.items():
            text = self.shipped(role)
            categories = {row.split(" | ", 2)[1] for row in rows}
            self.assertEqual(categories, {"required", "forbidden", "evidence"})
            for row in rows:
                with self.subTest(role=role, row=row):
                    self.assertIn(row, text)
                    self.assertTrue(self.validate_text(text.replace(row, "", 1), role))
                    self.assertTrue(
                        self.validate_text(text.replace(row, f"{row}\n{row}", 1), role)
                    )
                    damaged_category = row.replace(" | required | ", " | optional | ")
                    if damaged_category != row:
                        self.assertTrue(
                            self.validate_text(text.replace(row, damaged_category, 1), role)
                        )

    def test_output_rows_and_enums_are_exact(self) -> None:
        for role, rows in prompt_contract.OUTPUT_ROWS.items():
            text = self.shipped(role)
            for row in rows:
                with self.subTest(role=role, row=row):
                    self.assertIn(row, text)
                    self.assertTrue(self.validate_text(text.replace(row, "", 1), role))
                    self.assertTrue(
                        self.validate_text(text.replace(row, f"{row}\n{row}", 1), role)
                    )
        self.assert_rejected(
            "implementer",
            "COMPLETE | BLOCKED | NEEDS_CONTEXT | NEEDS_DECISION",
            "COMPLETE | BLOCKED",
        )
        self.assert_rejected(
            "operator",
            "MERGED | CLOSED | BLOCKED | REMOTE_UNVERIFIED",
            "MERGED | CLOSED | BLOCKED",
        )

    def test_stop_rows_are_required_once_and_exact(self) -> None:
        for role, rows in prompt_contract.STOP_ROWS.items():
            text = self.shipped(role)
            for row in rows:
                with self.subTest(role=role, row=row):
                    self.assertIn(row, text)
                    self.assertTrue(self.validate_text(text.replace(row, "", 1), role))
                    self.assertTrue(
                        self.validate_text(text.replace(row, f"{row}\n{row}", 1), role)
                    )
        self.assert_rejected(
            "reviewer",
            "- `STOP-SCRATCH` | FAIL |",
            "- `STOP-SCRATCH` | PASS |",
        )

    def test_implementer_accepts_all_declared_status_and_evidence_outputs(self) -> None:
        text = self.shipped("implementer")
        for value in ("COMPLETE", "BLOCKED", "NEEDS_CONTEXT", "NEEDS_DECISION"):
            self.assertIn(value, text)
        for field in (
            "summary",
            "changed_files",
            "commands_and_exit_status",
            "negative_mutation",
            "deviations",
            "risks",
            "omitted_gates",
            "commit_sha",
        ):
            self.assertIn(f"- {field}:", text)

    def test_reviewer_contract_has_three_lenses_and_no_repair_escape(self) -> None:
        text = self.shipped("reviewer")
        for clause_id in (
            "REV-STATIC",
            "REV-MUTATION",
            "REV-FULL-GATE",
            "REV-NO-REPAIR",
            "REV-WORKTREE",
        ):
            self.assertIn(f"`{clause_id}`", text)
        self.assertIn("scratch", text.lower())
        self.assertIn("static review", text.lower())
        self.assertIn("full declared gate", text.lower())

    def test_operator_continues_actionable_failures_until_pass(self) -> None:
        continue_row = (
            "- `OP-CONTINUE` | required | For every actionable in-scope reviewer "
            "FAIL dispatch a fresh implementer, run focused and full gates, and "
            "dispatch a fresh scoped reviewer; repeat until PASS because attempt "
            "or retry budgets are scheduling controls and never terminal blockers."
        )
        stop_rows = (
            "- `STOP-AUTHORITY` | BLOCKED | A required external action exceeds "
            "Authority and the precise missing authority is named.",
            "- `STOP-RESOURCE` | BLOCKED | A required external resource is "
            "unavailable and the precise resource is named.",
            "- `STOP-DEVELOPER` | BLOCKED | The developer explicitly directs the "
            "review loop to stop before PASS.",
            "- `STOP-REMOTE` | REMOTE_UNVERIFIED | Required remote state cannot be queried.",
        )
        text = self.shipped("operator")
        self.assertIn(continue_row, prompt_contract.METHOD_ROWS["operator"])
        self.assertIn(continue_row, text)
        self.assertEqual(prompt_contract.STOP_ROWS["operator"], stop_rows)
        for row in stop_rows:
            with self.subTest(row=row):
                self.assertIn(row, text)
        self.assertNotIn("STOP-BLOCKER", text)

    def test_reviewer_findings_are_first_structured_and_severity_ordered(self) -> None:
        text = self.shipped("reviewer")
        finding = prompt_contract.OUTPUT_ROWS["reviewer"][0]
        self.assertIn("severity_descending", finding)
        for field in (
            "severity",
            "path:line",
            "evidence",
            "violated_rule_or_requirement",
            "required_remediation",
        ):
            self.assertIn(field, finding)
        self.assert_rejected(
            "reviewer", "required_remediation", "suggested_change"
        )
        self.assert_rejected(
            "reviewer", "severity_descending", "severity_ascending"
        )

    def test_missing_prompt_and_unknown_role_are_distinct_failures(self) -> None:
        missing = Path(tempfile.gettempdir()) / "prompt-contract-does-not-exist.md"
        self.assertTrue(
            any(
                "missing" in error
                for error in prompt_contract.validate_prompt(
                    missing, "implementer"
                )
            )
        )
        self.assertTrue(
            any(
                "unknown runtime role" in error
                for error in prompt_contract.validate_prompt(
                    missing, "inventor"
                )
            )
        )

    def test_main_calls_semantic_validation_and_returns_nonzero_for_damage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_root = root / ".agents/prompts"
            prompt_root.mkdir(parents=True)
            for role in ("implementer", "reviewer", "operator"):
                shutil.copy(ROOT / ".agents/prompts" / f"{role}.md", prompt_root)
            reviewer = prompt_root / "reviewer.md"
            reviewer.write_text(
                reviewer.read_text(encoding="utf-8").replace(
                    "## Method",
                    "Scratch mutation replaces static review.\n\n## Method",
                    1,
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(prompt_contract, "ROOT", root),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(prompt_contract.main(), 1)
            self.assertIn("unparsed", output.getvalue())

    def test_repository_errors_owns_runtime_prompt_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_root = root / ".agents/prompts"
            prompt_root.mkdir(parents=True)
            for role in ("implementer", "reviewer", "operator"):
                shutil.copy(ROOT / ".agents/prompts" / f"{role}.md", prompt_root)
            self.assertEqual(
                prompt_contract.repository_errors(root), []
            )
            (prompt_root / "operator.md").unlink()
            errors = prompt_contract.repository_errors(root)
            self.assertTrue(any("operator.md" in error and "missing" in error for error in errors))

    def test_runtime_prompt_namespace_rejects_every_noncanonical_artifact(self) -> None:
        additions = {
            "critic.md": "# malformed runtime prompt\n",
            "reviewer-copy.md": self.shipped("reviewer"),
            "Reviewer.md": self.shipped("reviewer"),
        }
        for name, content in additions.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prompt_root = root / ".agents/prompts"
                prompt_root.mkdir(parents=True)
                for role in ("implementer", "reviewer", "operator"):
                    shutil.copy(ROOT / ".agents/prompts" / f"{role}.md", prompt_root)
                (prompt_root / name).write_text(content, encoding="utf-8")
                errors = prompt_contract.repository_errors(root)
                self.assertTrue(
                    any(name in error and "unexpected" in error for error in errors),
                    errors,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_root = root / ".agents/prompts"
            prompt_root.mkdir(parents=True)
            for role in ("implementer", "reviewer", "operator"):
                shutil.copy(ROOT / ".agents/prompts" / f"{role}.md", prompt_root)
            (prompt_root / "adapters").mkdir()
            errors = prompt_contract.repository_errors(root)
            self.assertTrue(
                any("adapters" in error and "unexpected" in error for error in errors),
                errors,
            )


if __name__ == "__main__":
    unittest.main()
