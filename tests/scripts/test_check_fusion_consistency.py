#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-fusion-consistency.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-fusion-consistency.py"
SPEC = importlib.util.spec_from_file_location("check_fusion_consistency", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

drift_models = mod.drift_models


class DriftModelTests(unittest.TestCase):
    def test_adopted_model_passes(self) -> None:
        # residual sites present, but the file references FusedChain => adopted.
        self.assertEqual(drift_models({"qwen3": (3, True)}, set()), [])

    def test_unadopted_model_fails(self) -> None:
        # residual sites present, no FusedChain, not allowlisted => drift.
        self.assertEqual(drift_models({"gemma2": (3, False)}, set()), ["gemma2"])

    def test_allowlisted_unadopted_passes(self) -> None:
        self.assertEqual(drift_models({"gemma2": (3, False)}, {"gemma2"}), [])

    def test_no_residual_sites_never_trips(self) -> None:
        # A post-norm / LayerNorm model with zero add+RMSNorm sites is not scanned
        # into the map at all; even if it were, 0 sites never drifts.
        self.assertEqual(drift_models({"olmo2": (0, False)}, set()), [])

    def test_mixed_reports_only_uncovered(self) -> None:
        result = drift_models(
            {
                "qwen3": (3, True),      # adopted
                "gemma2": (3, False),    # drift
                "glm4": (3, False),      # allowlisted
                "opt": (0, False),       # no sites
            },
            allowlisted={"glm4"},
        )
        self.assertEqual(result, ["gemma2"])

    def test_regex_matches_residual_overload_only(self) -> None:
        adopt = "vt::FusedChain(d.q, dhn.t(), h, w, &res.t(), vt::kFusedAddRmsNormStd, eps);"
        hand = "vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, gemma, &res.t());"
        standalone = "vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, gemma);"
        self.assertEqual(mod.count_residual_rmsnorm(hand), 1)
        self.assertEqual(mod.count_residual_rmsnorm(standalone), 0)
        self.assertTrue(mod.uses_catalog(adopt))
        self.assertFalse(mod.uses_catalog(hand))

    def test_allowlist_parsing(self) -> None:
        text = "# comment\ngemma2  # trailing reason\nglm4\n\n"
        self.assertEqual(mod.allowlisted_names(text), {"gemma2", "glm4"})

    def test_shipped_tree_is_green(self) -> None:
        # The real repo must pass: every hand-fusing model is adopted or allowlisted.
        scanned = mod.scan_models(ROOT / "src/vllm/model_executor/models")
        allowlisted = mod.allowlisted_names(
            (ROOT / "scripts/fusion-consistency-allowlist.txt").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(drift_models(scanned, allowlisted), [])
        # the sweep actually found the add+RMSNorm sites
        self.assertGreater(len(scanned), 5)

    def test_a_new_unadopted_model_would_fail(self) -> None:
        # Mutation: pretend a new model landed with a bare hand-fusion; it must trip.
        scanned = dict(mod.scan_models(ROOT / "src/vllm/model_executor/models"))
        allowlisted = mod.allowlisted_names(
            (ROOT / "scripts/fusion-consistency-allowlist.txt").read_text(
                encoding="utf-8"
            )
        )
        scanned["brand_new_arch"] = (2, False)
        self.assertIn("brand_new_arch", drift_models(scanned, allowlisted))

    def test_removing_allowlist_entry_would_fail(self) -> None:
        # Mutation: if a known-drift model were dropped from the allowlist WITHOUT
        # being migrated, the checker must flag it (the enforcement teeth).
        #
        # The known-drift example is DERIVED from the allowlist, never hardcoded:
        # migrating a model retires its entry (the Tier-B2 fold did exactly that
        # for gemma/gemma2/gemma3), and that must close the gate, not break this
        # test.
        scanned = mod.scan_models(ROOT / "src/vllm/model_executor/models")
        allowlisted = mod.allowlisted_names(
            (ROOT / "scripts/fusion-consistency-allowlist.txt").read_text(
                encoding="utf-8"
            )
        )
        exposed = set(drift_models(scanned, set()))
        still_hand_fusing = {s for s, (n, c) in scanned.items() if n and not c}
        # Emptying the allowlist must expose every in-tree hand-fusing model,
        self.assertEqual(still_hand_fusing - exposed, set())
        # and the allowlist must be load-bearing: it suppresses at least one model
        # the checker really does see. That is what makes the green gate mean
        # something rather than being vacuous.
        self.assertTrue(
            exposed & allowlisted,
            "the allowlist suppresses nothing the checker detects; either the "
            "detector regressed or every allowlist entry is now stale",
        )


if __name__ == "__main__":
    unittest.main()
