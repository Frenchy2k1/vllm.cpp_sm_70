#!/usr/bin/env python3
"""Behavior and mutation checks for purpose-specific public documentation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-doc-checkpoint.py"
SPEC = importlib.util.spec_from_file_location("doc_checkpoint", CHECKER)
assert SPEC is not None and SPEC.loader is not None
doc_checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = doc_checkpoint
SPEC.loader.exec_module(doc_checkpoint)


class RequirementParserTests(unittest.TestCase):
    def test_positive_action_and_repository_path_are_parsed(self) -> None:
        for requirement, expected in (
            ("Update docs/STATUS.md.", ("Update", "docs/STATUS.md")),
            ("Refresh .agents/NOW.md.", ("Refresh", ".agents/NOW.md")),
            (
                "Update docs/segment-/segment_.",
                ("Update", "docs/segment-/segment_"),
            ),
            (
                "Update docs/name_with-dash.v2.md.",
                ("Update", "docs/name_with-dash.v2.md"),
            ),
        ):
            with self.subTest(requirement=requirement):
                self.assertEqual(
                    doc_checkpoint.parse_requirement(requirement), expected
                )

    def test_malformed_positive_requirement_is_rejected_before_target_use(
        self,
    ) -> None:
        invalid = (
            "Update docs/STATUS.md..",
            "Update docs/STATUS.md...",
            "Update docs./STATUS.md.",
            "Update docs/STATUS./index.md.",
            "Update docs//STATUS.md.",
            "Update docs/./STATUS.md.",
            "Update docs/../STATUS.md.",
            "Update ../docs/STATUS.md.",
            "Update /docs/STATUS.md.",
            r"Update docs\STATUS.md.",
            "Update docs/STATUS file.md.",
            "Update docs/STATUS\tfile.md.",
            "Update docs/STATUS.md. Refresh docs/BENCHMARKS.md.",
            "Update docs/STATUS.md.;docs/BENCHMARKS.md.",
            "Update docs/STATUS?.md.",
            "Update docs/STATUS%.md.",
            "Update docs/STATUS:md.",
            "Update docs/STATUS\x00md.",
            "Update docs/STATUS\x1fmd.",
            "Update docs/STAT\u00a0US.md.",
            "Update docs/STAT\u00e9US.md.",
            "Observe docs/STATUS.md.",
            "Update docs/STATUS.md",
            " Update docs/STATUS.md.",
            "Update  docs/STATUS.md.",
            "Update\tdocs/STATUS.md.",
            "Update\u00a0docs/STATUS.md.",
            "Update only docs/STATUS.md.",
            "Update\r\ndocs/STATUS.md.",
            "Update docs/STATUS.md.\n",
            "Update docs/STATUS.md.\r",
            "Update docs/STATUS.md. ",
        )
        for requirement in invalid:
            with self.subTest(requirement=requirement):
                with self.assertRaises(ValueError):
                    doc_checkpoint.parse_requirement(requirement)

    def test_repository_path_alphabet_is_closed_and_exact(self) -> None:
        expected = frozenset(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789._-"
        )
        self.assertEqual(doc_checkpoint.REPOSITORY_PATH_CHARACTERS, expected)

    def test_path_alphabet_mutations_accepting_forbidden_classes_go_red(self) -> None:
        forbidden = (
            "%",
            ":",
            " ",
            "\t",
            "\r",
            "\n",
            "\x00",
            "\x1f",
            "\u00a0",
            "\u00e9",
        )
        for character in forbidden:
            with self.subTest(character=repr(character)):
                self.assertNotIn(character, doc_checkpoint.REPOSITORY_PATH_CHARACTERS)


class SemanticClassificationTests(unittest.TestCase):
    def test_runtime_code_is_a_feature_checkpoint(self) -> None:
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(["src/vt/cuda/matmul.cu"]),
        )

    def test_unlisted_runtime_code_remains_a_checkpoint(self) -> None:
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(["scripts/benchmark-grid.py"]),
        )

    def test_runtime_checkers_are_not_hidden_by_governance_names(self) -> None:
        for path in (
            "scripts/check-gemv-invocation-consistency.py",
            "tests/scripts/test_check_gemv_invocation_consistency.py",
        ):
            with self.subTest(path=path):
                classes = doc_checkpoint.classify_changed_paths([path])
                self.assertIn("feature_checkpoint", classes)
                self.assertNotIn("governance", classes)

    def test_exact_governance_checker_files_are_exempt(self) -> None:
        for path in (
            "scripts/check-commit-trailers.py",
            "scripts/check-policy.py",
            "scripts/check-doc-checkpoint.py",
            "scripts/check-prompt-contract.py",
            "scripts/check-protocol-consistency.py",
            ".agents/prompts/operator.md",
            "tests/scripts/test_doc_checkpoint.py",
            "tests/scripts/test_check_prompt_contract.py",
            "tests/scripts/test_check_protocol_consistency.py",
            "tests/scripts/test_check_commit_trailers.py",
            "tests/scripts/test_policy_waivers.py",
            "tests/scripts/test_check_pr_size.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    doc_checkpoint.classify_changed_paths([path]), {"governance"}
                )

    def test_similar_governance_names_remain_feature_checkpoints(self) -> None:
        for path in (
            "scripts/check-commit-trailers-extra.py",
            "tests/scripts/test_policy_waivers_extra.py",
            "tests/scripts/test_check_pr_size.md",
        ):
            with self.subTest(path=path):
                classes = doc_checkpoint.classify_changed_paths([path])
                self.assertIn("feature_checkpoint", classes)
                self.assertNotIn("governance", classes)

    def test_governance_only_task_one_files_are_not_a_checkpoint(self) -> None:
        paths = [
            ".agents/policy.csv",
            ".agents/waivers.csv",
            "scripts/policy_contract.py",
            "scripts/check-policy.py",
            "scripts/check-prompt-contract.py",
            "tests/scripts/test_policy_contract.py",
            "tests/scripts/test_check_prompt_contract.py",
        ]
        self.assertEqual(doc_checkpoint.classify_changed_paths(paths), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(set(paths)), [])

    def test_live_claim_checker_files_are_exact_governance(self) -> None:
        for path in (
            "scripts/claim-view.py",
            "scripts/ready-for-helper.py",
            "tests/scripts/test_claim_view.py",
            "tests/scripts/test_ready_for_helper.py",
            ".agents/completed/pre-cutover-claim-protocol.md",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    doc_checkpoint.classify_changed_paths([path]), {"governance"}
                )

    def test_exact_claim_cutover_migration_has_no_public_feature_obligation(self) -> None:
        paths = {
            "scripts/claim-view.py",
            "scripts/ready-for-helper.py",
            "tests/scripts/test_claim_view.py",
            "tests/scripts/test_ready_for_helper.py",
            ".agents/coordination.md",
            ".agents/completed/pre-cutover-claim-protocol.md",
            "scripts/check-doc-checkpoint.py",
            "tests/scripts/test_doc_checkpoint.py",
        }
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])

    def test_coordination_bypass_is_not_broadened_beyond_exact_cutover(self) -> None:
        self.assertEqual(
            doc_checkpoint.classify_changed_paths([".agents/coordination.md"]),
            {"feature_checkpoint", "live_state"},
        )
        classes = doc_checkpoint.classify_changed_paths(
            [
                ".agents/coordination.md",
                ".agents/completed/pre-cutover-claim-protocol.md",
                "src/vt/backend.cpp",
            ]
        )
        self.assertIn("feature_checkpoint", classes)
        self.assertIn("live_state", classes)
        pair_only = doc_checkpoint.classify_changed_paths(
            [
                ".agents/coordination.md",
                ".agents/completed/pre-cutover-claim-protocol.md",
            ]
        )
        self.assertIn("feature_checkpoint", pair_only)
        self.assertIn("live_state", pair_only)

    def test_exact_policy_consolidation_is_governance_only(self) -> None:
        paths = set(doc_checkpoint.POLICY_CONSOLIDATION_FILES)
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])

    def test_policy_consolidation_exemption_is_exact(self) -> None:
        paths = set(doc_checkpoint.POLICY_CONSOLIDATION_FILES)
        without_one = paths - {".agents/backend-matrix.md"}
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(sorted(without_one)),
        )
        with_product = paths | {"src/vt/backend.cpp"}
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(sorted(with_product)),
        )

    def test_exact_policy_cutover_is_governance_only(self) -> None:
        paths = set(doc_checkpoint.POLICY_CUTOVER_FILES)
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])

    def test_policy_cutover_exemption_is_exact(self) -> None:
        paths = set(doc_checkpoint.POLICY_CUTOVER_FILES)
        without_one = paths - {"scripts/agent-ready.py"}
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(sorted(without_one)),
        )
        with_product = paths | {"src/vt/backend.cpp"}
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(sorted(with_product)),
        )

    def test_exact_pr_size_bootstrap_is_governance_only(self) -> None:
        paths = set(doc_checkpoint.PR_SIZE_BOOTSTRAP_FILES)
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])

    def test_pr_size_bootstrap_exemption_is_exact(self) -> None:
        paths = set(doc_checkpoint.PR_SIZE_BOOTSTRAP_FILES)
        for removed in sorted(paths):
            with self.subTest(removed=removed):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths - {removed})),
                )
        with_product = paths | {"src/vt/backend.cpp"}
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(sorted(with_product)),
        )
        with_governance = paths | {"scripts/check-protocol-consistency.py"}
        self.assertIn(
            "feature_checkpoint",
            doc_checkpoint.classify_changed_paths(sorted(with_governance)),
        )

    def test_pending_pr_range_correction_is_exact(self) -> None:
        paths = set(doc_checkpoint.PENDING_PR_RANGE_FILES)
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])
        for removed in sorted(paths):
            with self.subTest(removed=removed):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths - {removed})),
                )
        for added in ("src/vt/backend.cpp", "scripts/check-protocol-consistency.py"):
            with self.subTest(added=added):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths | {added})),
                )

    def test_synthetic_merge_trailer_correction_is_exact(self) -> None:
        paths = set(doc_checkpoint.SYNTHETIC_MERGE_RANGE_FILES)
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])
        for removed in sorted(paths):
            with self.subTest(removed=removed):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths - {removed})),
                )
        for added in ("src/vt/backend.cpp", "scripts/check-protocol-consistency.py"):
            with self.subTest(added=added):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths | {added})),
                )

    def test_role_step_range_correction_is_exact(self) -> None:
        paths = set(doc_checkpoint.SYNTHETIC_MERGE_RANGE_FILES)
        self.assertEqual(doc_checkpoint.classify_changed_paths(sorted(paths)), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])
        for removed in sorted(paths):
            with self.subTest(removed=removed):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths - {removed})),
                )
        for added in ("src/vt/backend.cpp", "scripts/check-protocol-consistency.py"):
            with self.subTest(added=added):
                self.assertIn(
                    "feature_checkpoint",
                    doc_checkpoint.classify_changed_paths(sorted(paths | {added})),
                )

    def test_governance_design_is_not_misclassified_as_feature_work(self) -> None:
        path = "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md"
        self.assertEqual(doc_checkpoint.classify_changed_paths([path]), {"governance"})
        self.assertEqual(doc_checkpoint.checkpoint_errors({path}), [])

    def test_feature_support_paths_have_both_semantic_classes(self) -> None:
        classes = doc_checkpoint.classify_changed_paths(
            ["src/vllm/model_executor/models/qwen3.cpp"]
        )
        self.assertEqual(classes, {"feature_checkpoint", "feature_surface"})

    def test_user_interface_paths_are_usage_changes_and_checkpoints(self) -> None:
        classes = doc_checkpoint.classify_changed_paths(
            ["src/vllm/entrypoints/openai/api_server.cpp"]
        )
        self.assertEqual(classes, {"feature_checkpoint", "user_usage"})

    def test_configuration_and_exact_install_sources_require_usage(self) -> None:
        for path in (
            ".env.example",
            "CMakeLists.txt",
            "cmake/install.cmake",
        ):
            with self.subTest(path=path):
                classes = doc_checkpoint.classify_changed_paths([path])
                self.assertIn("feature_checkpoint", classes)
                self.assertIn("user_usage", classes)

    def test_unrelated_cmake_module_is_not_automatically_install_usage(self) -> None:
        classes = doc_checkpoint.classify_changed_paths(["cmake/FindNVTX.cmake"])
        self.assertIn("feature_checkpoint", classes)
        self.assertNotIn("user_usage", classes)

    def test_state_append_is_live_state_and_checkpoint(self) -> None:
        classes = doc_checkpoint.classify_changed_paths([".agents/state.md"])
        self.assertEqual(classes, {"feature_checkpoint", "live_state"})


class RequiredSurfaceTests(unittest.TestCase):
    def assertMissing(self, paths: set[str], surface: str) -> None:
        errors = doc_checkpoint.checkpoint_errors(paths)
        self.assertTrue(any(surface in error for error in errors), errors)

    def test_feature_checkpoint_requires_status_and_benchmarks(self) -> None:
        paths = {"src/vt/cuda/matmul.cu"}
        self.assertMissing(paths, "docs/STATUS.md")
        self.assertMissing(paths, "docs/BENCHMARKS.md")

    def test_each_checkpoint_surface_is_independently_required(self) -> None:
        self.assertMissing(
            {"src/vt/cuda/matmul.cu", "docs/STATUS.md"}, "docs/BENCHMARKS.md"
        )
        self.assertMissing(
            {"src/vt/cuda/matmul.cu", "docs/BENCHMARKS.md"}, "docs/STATUS.md"
        )

    def test_feature_surface_requires_features(self) -> None:
        self.assertMissing(
            {
                ".agents/model-matrix.md",
                "docs/STATUS.md",
                "docs/BENCHMARKS.md",
            },
            "docs/FEATURES.md",
        )

    def test_usage_change_requires_usage(self) -> None:
        self.assertMissing(
            {
                "examples/cli/main.cpp",
                "docs/STATUS.md",
                "docs/BENCHMARKS.md",
            },
            "docs/USAGE.md",
        )

    def test_state_append_requires_fresh_now(self) -> None:
        self.assertMissing(
            {
                ".agents/state.md",
                "docs/STATUS.md",
                "docs/BENCHMARKS.md",
            },
            ".agents/NOW.md",
        )

    def test_readme_churn_without_landing_trigger_is_rejected(self) -> None:
        errors = doc_checkpoint.checkpoint_errors({"README.md"})
        self.assertTrue(any("README.md" in error and "trigger" in error for error in errors))

    def test_readme_is_not_justified_by_coedited_public_projections(self) -> None:
        paths = {
            "README.md",
            "docs/USAGE.md",
            "docs/FEATURES.md",
            "docs/BENCHMARKS.md",
            "docs/STATUS.md",
        }
        errors = doc_checkpoint.checkpoint_errors(paths)
        self.assertTrue(any("README.md" in error for error in errors), errors)

    def test_readme_with_explicit_install_source_is_allowed(self) -> None:
        paths = {
            "README.md",
            "CMakeLists.txt",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
            "docs/USAGE.md",
        }
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])

    def test_explicit_landing_source_requires_readme_projection(self) -> None:
        paths = {
            "CMakeLists.txt",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
            "docs/USAGE.md",
        }
        self.assertMissing(paths, "README.md")

    def test_all_required_projections_satisfy_a_user_feature_change(self) -> None:
        paths = {
            "src/vllm/model_executor/models/qwen3.cpp",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
            "docs/FEATURES.md",
        }
        self.assertEqual(doc_checkpoint.checkpoint_errors(paths), [])


class PolicyMappingTests(unittest.TestCase):
    def test_required_surfaces_are_driven_by_public_policy_rules(self) -> None:
        self.assertEqual(
            doc_checkpoint.required_public_surfaces(
                {"feature_checkpoint", "feature_surface", "user_usage", "live_state"}
            ),
            {
                "docs/STATUS.md",
                "docs/BENCHMARKS.md",
                "docs/FEATURES.md",
                "docs/USAGE.md",
                ".agents/NOW.md",
            },
        )

    def test_every_public_rule_is_completely_parsed(self) -> None:
        for rule_id, rule in doc_checkpoint.public_document_rules().items():
            with self.subTest(rule_id=rule_id):
                binding = doc_checkpoint.parse_public_rule(rule)
                self.assertEqual(binding.rule_id, rule_id)
                self.assertEqual(binding.surface, rule.scope)
                self.assertEqual(binding.change_class, rule.trigger)


if __name__ == "__main__":
    unittest.main()
