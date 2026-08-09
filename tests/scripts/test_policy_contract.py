#!/usr/bin/env python3
"""Mutation tests for the authoritative policy and waiver registries."""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.policy_contract import (
    PolicyRule,
    Waiver,
    load_policy,
    load_waivers,
    validate_policy,
)


POLICY_HEADER = "rule_id,scope,trigger,requirement,enforcement,waiver_class,procedure\n"
WAIVER_HEADER = "waiver_id,rule_id,scope,owner,reason,evidence,expires\n"
ROOT = Path(__file__).resolve().parents[2]


class PolicyFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".agents").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/check-policy.py").write_text("#!/usr/bin/env python3\n")
        (self.root / ".agents/workflow.md").write_text("# Workflow\n")
        self.write_policy(
            "POL-TEST-001,all,always,Keep the contract testable.,"
            "scripts/check-policy.py,expiring,.agents/workflow.md\n"
        )
        self.write_waivers("")

    def close(self) -> None:
        self.tmp.cleanup()

    def write_policy(self, rows: str) -> None:
        (self.root / ".agents/policy.csv").write_text(POLICY_HEADER + rows)

    def write_waivers(self, rows: str) -> None:
        (self.root / ".agents/waivers.csv").write_text(WAIVER_HEADER + rows)


class LoadContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = PolicyFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_records_are_immutable_and_valid_registry_loads(self) -> None:
        rules = load_policy(self.fx.root)
        self.assertEqual(set(rules), {"POL-TEST-001"})
        self.assertEqual(rules["POL-TEST-001"].waiver_class, "expiring")
        self.assertEqual(load_waivers(self.fx.root, rules, today=dt.date(2026, 8, 7)), [])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rules["POL-TEST-001"].scope = "changed"  # type: ignore[misc]
        self.assertTrue(dataclasses.is_dataclass(PolicyRule))
        self.assertTrue(dataclasses.is_dataclass(Waiver))


class PolicyMutations(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = PolicyFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def errors(self) -> list[str]:
        return validate_policy(self.fx.root, schema_only=True)

    def test_valid_fixture_passes_schema_validation(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_exact_header_is_required(self) -> None:
        path = self.fx.root / ".agents/policy.csv"
        path.write_text(path.read_text().replace("rule_id,scope", "id,scope", 1))
        self.assertTrue(any("header" in error for error in self.errors()))

    def test_duplicate_and_malformed_rule_ids_are_rejected(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,First.,scripts/check-policy.py,never,.agents/workflow.md\n"
            "POL-TEST-001,all,always,Second.,scripts/check-policy.py,never,.agents/workflow.md\n"
            "test-2,all,always,Third.,scripts/check-policy.py,never,.agents/workflow.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("duplicate rule_id" in error for error in errors), errors)
        self.assertTrue(any("malformed rule_id" in error for error in errors), errors)

    def test_multiline_field_and_blank_physical_record_are_rejected(self) -> None:
        self.fx.write_policy(
            'POL-TEST-001,all,always,"First physical line\nsecond physical line",'
            "scripts/check-policy.py,never,.agents/workflow.md\n\n"
        )
        errors = self.errors()
        self.assertTrue(any("multiline" in error for error in errors), errors)
        self.assertTrue(any("blank physical" in error for error in errors), errors)

    def test_empty_fields_and_bad_waiver_class_are_rejected_in_bootstrap(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,,always,,scripts/check-policy.py,forever,.agents/workflow.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("scope is empty" in error for error in errors), errors)
        self.assertTrue(any("requirement is empty" in error for error in errors), errors)
        self.assertTrue(any("waiver_class" in error for error in errors), errors)

    def test_checker_and_procedure_paths_must_be_known_repo_files(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/check-missing.py,never,.agents/missing.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("unknown enforcement" in error for error in errors), errors)
        self.assertTrue(any("unknown procedure" in error for error in errors), errors)

    def test_absolute_traversing_and_non_checker_paths_are_rejected(self) -> None:
        (self.fx.root / "scripts/tool.py").write_text("pass\n")
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/tool.py;/tmp/check.py,never,../workflow.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("checker entrypoint" in error for error in errors), errors)
        self.assertTrue(any("repo-relative" in error for error in errors), errors)

    def test_policy_count_and_byte_budgets_are_enforced(self) -> None:
        rows = "".join(
            f"POL-TEST-{index:03d},all,always,Rule {index}.,scripts/check-policy.py,never,.agents/workflow.md\n"
            for index in range(61)
        )
        self.fx.write_policy(rows)
        self.assertTrue(any("more than 60" in error for error in self.errors()))
        self.fx.write_policy(
            "POL-TEST-001,all,always," + ("x" * 17000) + ",scripts/check-policy.py,never,.agents/workflow.md\n"
        )
        self.assertTrue(any("16384-byte" in error for error in self.errors()))

    def test_load_policy_refuses_an_invalid_registry(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/check-missing.py,never,.agents/workflow.md\n"
        )
        with self.assertRaises(ValueError):
            load_policy(self.fx.root)


class WaiverMutations(unittest.TestCase):
    TODAY = dt.date(2026, 8, 7)

    def setUp(self) -> None:
        self.fx = PolicyFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def load_errors(self) -> str:
        rules = load_policy(self.fx.root)
        with self.assertRaises(ValueError) as raised:
            load_waivers(self.fx.root, rules, today=self.TODAY)
        return str(raised.exception)

    def mark_used(self, waiver_id: str) -> None:
        (self.fx.root / ".agents/waiver-evidence.md").write_text(
            f"Applied exception: {waiver_id}\n"
        )

    def test_exact_header_and_valid_future_waiver(self) -> None:
        self.mark_used("WAIVER-TEST-001")
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,maintainer,"
            "Bounded migration,docs/evidence.md,2026-08-08\n"
        )
        waiver = load_waivers(
            self.fx.root, load_policy(self.fx.root), today=self.TODAY
        )[0]
        self.assertEqual(waiver.expires, dt.date(2026, 8, 8))
        self.assertEqual(waiver.scope, "task:POLICY-1")

        path = self.fx.root / ".agents/waivers.csv"
        path.write_text(path.read_text().replace("waiver_id,rule_id", "id,rule_id", 1))
        self.assertIn("header", self.load_errors())

    def test_unknown_and_never_waivable_rules_are_rejected(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/check-policy.py,never,.agents/workflow.md\n"
        )
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-MISSING,task:POLICY-1,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-002,POL-TEST-001,task:POLICY-2,owner,reason,evidence,2026-08-08\n"
        )
        errors = self.load_errors()
        self.assertIn("unknown rule", errors)
        self.assertIn("cannot be waived", errors)

    def test_scope_must_be_narrow_exact_and_unique(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,path:*,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-002,POL-TEST-001,repository:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-003,POL-TEST-001,task:POLICY-3,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-004,POL-TEST-001,task:POLICY-3,owner,reason,evidence,2026-08-09\n"
        )
        errors = self.load_errors()
        self.assertIn("wildcard", errors)
        self.assertIn("exact scope", errors)
        self.assertIn("duplicate rule/scope", errors)

    def test_repository_wide_and_traversing_scopes_are_not_exact(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,pr:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-002,POL-TEST-001,commit:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-003,POL-TEST-001,hardware:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-004,POL-TEST-001,path:foo/..,owner,reason,evidence,2026-08-08\n"
        )
        errors = self.load_errors()
        for scope in ("pr:all", "commit:all", "hardware:all", "path:foo/.."):
            with self.subTest(scope=scope):
                self.assertIn(repr(scope), errors)
                self.assertIn("exact scope", errors)

    def test_each_supported_scope_kind_accepts_a_concrete_target(self) -> None:
        waiver_ids = [f"WAIVER-TEST-{index:03d}" for index in range(1, 7)]
        (self.fx.root / ".agents/waiver-evidence.md").write_text(
            "Applied exceptions: " + " ".join(waiver_ids) + "\n"
        )
        scopes = (
            "task:POLICY-1",
            "pr:128",
            f"commit:{'a' * 40}",
            "path:.agents/workflow.md",
            "gate:policy-schema",
            "hardware:GB10-1",
        )
        self.fx.write_waivers(
            "".join(
                f"{waiver_id},POL-TEST-001,{scope},owner,reason,evidence,2026-08-08\n"
                for waiver_id, scope in zip(waiver_ids, scopes, strict=True)
            )
        )
        waivers = load_waivers(
            self.fx.root, load_policy(self.fx.root), today=self.TODAY
        )
        self.assertEqual([waiver.scope for waiver in waivers], list(scopes))

    def test_duplicate_ids_and_missing_attribution_are_rejected(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-2,owner,,evidence,2026-08-08\n"
            "WAIVER-TEST-003,POL-TEST-001,task:POLICY-3,owner,reason,,2026-08-08\n"
        )
        errors = self.load_errors()
        self.assertIn("duplicate waiver_id", errors)
        self.assertIn("owner is empty", errors)
        self.assertIn("reason is empty", errors)
        self.assertIn("evidence is empty", errors)

    def test_invalid_expired_and_today_expiry_are_rejected(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,owner,reason,evidence,not-a-date\n"
            "WAIVER-TEST-002,POL-TEST-001,task:POLICY-2,owner,reason,evidence,2026-08-06\n"
            "WAIVER-TEST-003,POL-TEST-001,task:POLICY-3,owner,reason,evidence,2026-08-07\n"
        )
        errors = self.load_errors()
        self.assertIn("ISO date", errors)
        self.assertIn("expired", errors)

    def test_unreferenced_waiver_is_rejected_as_unused(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,owner,reason,evidence,2026-08-08\n"
        )
        self.assertIn("unused waiver", self.load_errors())

    def test_schema_only_does_not_suppress_waiver_defects(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-MISSING,path:*,owner,reason,evidence,2026-08-06\n"
        )
        errors = validate_policy(self.fx.root, schema_only=True)
        self.assertTrue(any("unknown rule" in error for error in errors), errors)
        self.assertTrue(any("wildcard" in error for error in errors), errors)
        self.assertTrue(any("expired" in error for error in errors), errors)


class CommandLineContract(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = PolicyFixture()
        shutil.copy2(ROOT / "scripts/policy_contract.py", self.fx.root / "scripts")
        shutil.copy2(ROOT / "scripts/check-policy.py", self.fx.root / "scripts")

    def tearDown(self) -> None:
        self.fx.close()

    def run_checker(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", "scripts/check-policy.py", *arguments],
            cwd=self.fx.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_passes_valid_registry_and_fails_bootstrap_defect(self) -> None:
        self.assertEqual(self.run_checker("--schema-only").returncode, 0)
        path = self.fx.root / ".agents/policy.csv"
        path.write_text(path.read_text().replace("rule_id", "id", 1))
        result = self.run_checker("--schema-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("header", result.stdout + result.stderr)


class RepositoryRegistry(unittest.TestCase):
    def test_review_failure_policy_requires_continuation_until_pass(self) -> None:
        rule = load_policy(ROOT)["POL-REVIEW-NO-REPAIR"]
        for clause in (
            "actionable in-scope findings",
            "fresh implementer",
            "without repair in the coordinating session",
            "focused and full gates",
            "fresh scoped review",
            "until PASS",
            "attempt budgets never terminate correctable findings",
            "explicit developer direction",
            "precise external authority or resource blocker",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, rule.requirement)
        self.assertEqual(rule.waiver_class, "never")

    def test_accepted_design_inventory_passes_schema(self) -> None:
        rules = load_policy(ROOT)
        self.assertLessEqual(len(rules), 60)
        self.assertLessEqual((ROOT / ".agents/policy.csv").stat().st_size, 16 * 1024)
        self.assertEqual(validate_policy(ROOT, schema_only=True), [])

    def test_consolidated_policy_contract_passes_full_validation(self) -> None:
        self.assertEqual(validate_policy(ROOT), [])

    def test_historical_line_ranges_resolve_in_their_named_procedure(self) -> None:
        reference = re.compile(
            r"(?P<path>(?:completed/porting-discipline-legacy|workflow|verification|porting)\.md)"
            r"(?::| )"
            r"(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?"
        )
        state = ROOT / ".agents/state.md"
        failures: list[str] = []
        for match in reference.finditer(state.read_text(encoding="utf-8")):
            target = state.parent / match.group("path")
            line_count = len(target.read_text(encoding="utf-8").splitlines())
            end = int(match.group("end") or match.group("start"))
            if end > line_count:
                failures.append(
                    f"{match.group(0)!r} ends at line {end}, but {target.name} "
                    f"has {line_count} lines"
                )
        self.assertEqual(failures, [])


class ConsolidationMutations(unittest.TestCase):
    """Pin the compact bootstrap, active procedures, and archive cutover."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copy2(ROOT / "AGENTS.md", self.root / "AGENTS.md")
        shutil.copy2(ROOT / ".env.example", self.root / ".env.example")
        shutil.copytree(ROOT / ".agents", self.root / ".agents")
        shutil.copytree(ROOT / "scripts", self.root / "scripts")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def errors(self) -> list[str]:
        return validate_policy(self.root)

    def mutate(self, relative: str, old: str, new: str) -> None:
        path = self.root / relative
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))

    def restore(self, relative: str) -> None:
        shutil.copy2(ROOT / relative, self.root / relative)

    def test_agents_budget_boot_order_and_generated_t0_are_enforced(self) -> None:
        path = self.root / "AGENTS.md"
        path.write_text(path.read_text() + ("x" * 13_000))
        self.assertTrue(any("12 KiB" in error for error in self.errors()))

        shutil.copy2(ROOT / "AGENTS.md", path)
        self.mutate("AGENTS.md", "1. Run `scripts/agent-start.py`", "1. Read task state before running `scripts/agent-start.py`")
        self.assertTrue(any("boot block" in error for error in self.errors()))

        shutil.copy2(ROOT / "AGENTS.md", path)
        self.mutate("AGENTS.md", "Use policy.csv as", "Consult policy.csv as")
        self.assertTrue(any("T0 block" in error for error in self.errors()))

    def test_legacy_active_duplicate_and_ambiguous_archives_are_rejected(self) -> None:
        (self.root / ".agents/directives.md").write_text("duplicate active policy\n")
        self.assertTrue(any("retired active policy" in error for error in self.errors()))
        (self.root / ".agents/directives.md").unlink()
        (self.root / ".agents/completed/directives-copy.md").write_text("ambiguous\n")
        self.assertTrue(any("ambiguous policy archive" in error for error in self.errors()))

    def test_procedure_must_be_regular_nonempty_and_inside_repository(self) -> None:
        procedure = self.root / ".agents/verification.md"
        procedure.unlink()
        procedure.symlink_to(self.root / ".agents/workflow.md")
        self.assertTrue(any("symlink" in error for error in self.errors()))

        procedure.unlink()
        procedure.write_text("")
        self.assertTrue(any("empty procedure" in error for error in self.errors()))

        self.mutate(".agents/policy.csv", ".agents/verification.md", "../verification.md")
        self.assertTrue(any("repo-relative" in error for error in self.errors()))

    def test_normative_paragraph_requires_one_applicable_policy_id(self) -> None:
        self.mutate(".agents/workflow.md", "[POL-BOOT-NOW]", "[NO-POLICY]")
        self.assertTrue(any("exactly one policy reference" in error for error in self.errors()))

        self.restore(".agents/workflow.md")
        self.mutate(".agents/workflow.md", "[POL-BOOT-NOW]", "[POL-BOOT-NOW] [POL-BOOT-TASK]")
        self.assertTrue(any("exactly one policy reference" in error for error in self.errors()))

        self.restore(".agents/workflow.md")
        self.mutate(".agents/workflow.md", "[POL-BOOT-NOW]", "[POL-MIRROR-VLLM]")
        self.assertTrue(any("belongs to" in error for error in self.errors()))

    def test_every_rule_has_one_procedure_back_reference(self) -> None:
        path = self.root / ".agents/workflow.md"
        text = path.read_text()
        paragraph = next(p for p in text.split("\n\n") if "[POL-BOOT-NOW]" in p)
        path.write_text(text.replace(paragraph + "\n\n", "", 1))
        self.assertTrue(any("missing procedure back-reference" in error for error in self.errors()))

    def test_retired_links_and_methodology_drift_are_rejected(self) -> None:
        self.mutate("AGENTS.md", "(.agents/workflow.md)", "(.agents/directives.md)")
        self.assertTrue(any("retired active path" in error for error in self.errors()))

        self.restore("AGENTS.md")
        self.mutate(".agents/workflow.md", "Write or port the smallest\ntest that fails", "Write a test near the fix")
        self.assertTrue(any("implementation phase" in error for error in self.errors()))

        self.restore(".agents/workflow.md")
        self.mutate(".agents/workflow.md", "static inspection and targeted scratch mutations", "static inspection")
        self.assertTrue(any("mutation review" in error for error in self.errors()))

        self.restore(".agents/workflow.md")
        self.mutate(".agents/workflow.md", "fresh implementer", "implementer")
        self.assertTrue(any("fresh-agent repair" in error for error in self.errors()))

    def test_public_document_purpose_drift_is_rejected(self) -> None:
        replacements = (
            ("every feature or iteration checkpoint", "occasional checkpoint"),
            ("feature, model, backend, or quantization surface", "feature surface"),
            ("commands, C API, configuration, installation, or user workflows", "usage"),
            ("only for a user-visible landing-page headline", "for any change"),
            ("same change as every qualifying appended structured event", "later change"),
        )
        for old, new in replacements:
            with self.subTest(old=old):
                self.restore(".agents/workflow.md")
                self.mutate(".agents/workflow.md", old, new)
                self.assertTrue(any("purpose contract" in error for error in self.errors()), self.errors())


if __name__ == "__main__":
    unittest.main()
