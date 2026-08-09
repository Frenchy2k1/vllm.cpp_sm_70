#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-env-doc.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-env-doc.py"
SPEC = importlib.util.spec_from_file_location("check_env_doc", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_env_doc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_env_doc
SPEC.loader.exec_module(check_env_doc)

undocumented = check_env_doc.undocumented_env_vars


class UndocumentedEnvVarTests(unittest.TestCase):
    def test_documented_var_passes(self) -> None:
        self.assertEqual(
            undocumented({"VT_FOO"}, {"VT_FOO"}, set()), []
        )

    def test_allowlisted_var_passes(self) -> None:
        self.assertEqual(
            undocumented({"VT_FOO"}, set(), {"VT_FOO"}), []
        )

    def test_undocumented_var_fails(self) -> None:
        # A scanned var covered by neither surface is reported.
        self.assertEqual(
            undocumented({"VT_NEW_KNOB"}, set(), set()), ["VT_NEW_KNOB"]
        )

    def test_mixed_reports_only_the_uncovered(self) -> None:
        result = undocumented(
            {"VT_A", "VT_B", "VLLM_C"},
            documented={"VT_A"},
            allowlisted={"VT_B"},
        )
        self.assertEqual(result, ["VLLM_C"])

    def test_helpers_harvest_names(self) -> None:
        doc = "The `VT_CPU_REF` knob and `VLLM_CPP_CPU_THREADS`."
        self.assertEqual(
            check_env_doc.documented_names(doc),
            {"VT_CPU_REF", "VLLM_CPP_CPU_THREADS"},
        )
        allow = "# comment\nVT_GDN_TMA\nVT_MOE_DECODE   # trailing\n\n"
        self.assertEqual(
            check_env_doc.allowlisted_names(allow), {"VT_GDN_TMA", "VT_MOE_DECODE"}
        )

    def test_shipped_tree_is_fully_covered(self) -> None:
        # The real repo must pass: every scanned name is documented or allowlisted.
        scanned = check_env_doc.scan_env_names(ROOT)
        documented = check_env_doc.documented_names(
            (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")
        )
        allowlisted = check_env_doc.allowlisted_names(
            (ROOT / "scripts/env-doc-allowlist.txt").read_text(encoding="utf-8")
        )
        self.assertEqual(undocumented(scanned, documented, allowlisted), [])
        self.assertGreater(len(scanned), 100)  # the sweep actually found names

    def test_inherited_variables_have_exact_public_internal_split(self) -> None:
        public = {
            "VT_GEMMA4_EXPERT_VRAM_MB",
            "VT_SERVER_MAX_NEW_TOKENS",
            "VT_SERVER_MAX_PROMPT_CHARS",
        }
        internal = {
            "VT_GEMMA4_BATCH_EXPERTS",
            "VT_GEMMA4_CUSTOM_EXPERT",
            "VT_GEMMA4_FP8_NATIVE",
            "VT_GEMMA4_FUSED_EXPERTS",
            "VT_GEMMA4_HOST_AXPY",
            "VT_GEMMA4_PROFILE",
            "VT_ROCM_GEMM_COMPUTE",
            "VT_ROCM_GEMV",
            "VT_ROCM_HIPBLASLT",
        }
        inherited = public | internal
        scanned = check_env_doc.scan_env_names(ROOT)
        documented = check_env_doc.documented_names(
            (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")
        )
        allowlisted = check_env_doc.allowlisted_names(
            (ROOT / "scripts/env-doc-allowlist.txt").read_text(encoding="utf-8")
        )

        self.assertLessEqual(inherited, scanned)
        self.assertEqual(inherited & documented, public)
        self.assertEqual(inherited & allowlisted, internal)
        self.assertTrue((inherited & documented).isdisjoint(inherited & allowlisted))

    def test_a_fabricated_new_var_would_fail(self) -> None:
        # Mutation: pretend the code grew a new undocumented var; it must trip.
        scanned = check_env_doc.scan_env_names(ROOT)
        documented = check_env_doc.documented_names(
            (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")
        )
        allowlisted = check_env_doc.allowlisted_names(
            (ROOT / "scripts/env-doc-allowlist.txt").read_text(encoding="utf-8")
        )
        mutated = set(scanned) | {"VT_A_BRAND_NEW_UNDOCUMENTED_KNOB"}
        result = undocumented(mutated, documented, allowlisted)
        self.assertIn("VT_A_BRAND_NEW_UNDOCUMENTED_KNOB", result)


if __name__ == "__main__":
    unittest.main()
