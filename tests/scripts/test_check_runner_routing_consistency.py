#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-runner-routing-consistency.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-runner-routing-consistency.py"
ALLOWLIST = ROOT / "scripts/runner-routing-allowlist.txt"
SPEC = importlib.util.spec_from_file_location("check_runner_routing_consistency", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod  # register BEFORE exec so the frozen dataclass resolves
SPEC.loader.exec_module(mod)

ModelRoute = mod.ModelRoute
drift_models = mod.drift_models


def route(name: str, classification: str, private_loop: bool = False) -> ModelRoute:
    return ModelRoute(name, f"{name}_registry.cpp", f"Forward{name}", classification,
                      private_loop)


class DriftModelTests(unittest.TestCase):
    def test_device_model_passes(self) -> None:
        # device-resident logits on the runner => clean.
        self.assertEqual(drift_models({"qwen3_dense": route("qwen3_dense", "DEVICE")}, set()), [])

    def test_host_model_fails(self) -> None:
        # host logits off the sampler, not allowlisted => drift.
        self.assertEqual(drift_models({"laguna": route("laguna", "HOST")}, set()), ["laguna"])

    def test_allowlisted_host_passes(self) -> None:
        self.assertEqual(drift_models({"laguna": route("laguna", "HOST")}, {"laguna"}), [])

    def test_refuse_stub_never_trips(self) -> None:
        # A REFUSE-by-name stub (VT_CHECK(false)) decodes nothing => never drifts.
        self.assertEqual(drift_models({"kimi_k3": route("kimi_k3", "REFUSE")}, set()), [])

    def test_none_never_trips(self) -> None:
        # No recognizable logit producer is not treated as HOST drift.
        self.assertEqual(drift_models({"x": route("x", "NONE")}, set()), [])

    def test_mixed_reports_only_host_uncovered(self) -> None:
        result = drift_models(
            {
                "qwen3_dense": route("qwen3_dense", "DEVICE"),  # clean
                "laguna": route("laguna", "HOST"),              # drift
                "deepseek_v4": route("deepseek_v4", "HOST"),    # allowlisted
                "kimi_k3": route("kimi_k3", "REFUSE"),          # skip
            },
            allowlisted={"deepseek_v4"},
        )
        self.assertEqual(result, ["laguna"])

    def test_classify_body(self) -> None:
        device = "{ return WrapDeviceLogits(d, std::move(dl), n, v); }"
        view = "{ ForwardLogits fl; fl.device_tensor = t; return fl; }"
        host = "{ return HostLogits(std::move(logits), vocab); }"
        host_field = "{ ForwardLogits out; out.host = std::move(flat); return out; }"
        refuse = "{ VT_CHECK(false, kPending); return {}; }"
        self.assertEqual(mod.classify_body(device), "DEVICE")
        self.assertEqual(mod.classify_body(view), "DEVICE")
        self.assertEqual(mod.classify_body(host), "HOST")
        self.assertEqual(mod.classify_body(host_field), "HOST")
        self.assertEqual(mod.classify_body(refuse), "REFUSE")
        self.assertEqual(mod.classify_body(None), "NONE")

    def test_classify_body_device_wins_over_comment(self) -> None:
        # A comment mentioning HostLogits inside a device forward must not misclassify.
        body = "{ /* returns HostLogits on the opt-out */ return WrapDeviceLogits(x); }"
        self.assertEqual(mod.classify_body(body), "DEVICE")

    def test_extract_fn_body_matches_definition_not_call(self) -> None:
        text = (
            "ForwardLogits ForwardFoo(LoadedModel& m, const ModelForwardInput& in) {\n"
            "  if (in.gather_logits) return FooModel::ForwardDevice(in);\n"
            "  return HostLogits(FooModel::Forward(in), v);\n"
            "}\n"
            "// elsewhere a call: ForwardFoo(model, input);\n"
        )
        body = mod.extract_fn_body(text, "ForwardFoo")
        self.assertIsNotNone(body)
        self.assertIn("FooModel::ForwardDevice", body)
        self.assertIn("HostLogits", body)

    def test_resolve_alias(self) -> None:
        alias = {"LlamaModel": "Qwen3DenseModel", "MistralModel": "Qwen3DenseModel"}
        self.assertEqual(mod.resolve_alias("LlamaModel", alias), "Qwen3DenseModel")
        self.assertEqual(mod.resolve_alias("GemmaModel", alias), "GemmaModel")

    def test_allowlist_parsing(self) -> None:
        text = "# comment\nlaguna  # trailing reason\nqwen3_vl\n\n"
        self.assertEqual(mod.allowlisted_names(text), {"laguna", "qwen3_vl"})

    def test_shipped_tree_is_green(self) -> None:
        # The real repo must pass: every registered model is device-resident or
        # allowlisted (or a refuse stub).
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        allowlisted = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        self.assertEqual(drift_models(scanned, allowlisted), [])
        # the sweep actually found the registrations (>20 registered models)
        self.assertGreater(len(scanned), 20)
        # and it really classified device-resident models (not everything is HOST)
        self.assertGreater(
            sum(1 for r in scanned.values() if r.classification == "DEVICE"), 15
        )

    def test_known_off_framework_are_host(self) -> None:
        # The 3 off-framework models the audit found must classify HOST (so the
        # allowlist is load-bearing, not decorative).
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        for name in ("laguna", "deepseek_v4", "qwen3_vl"):
            self.assertIn(name, scanned)
            self.assertEqual(scanned[name].classification, "HOST", name)
        # Qwen3-VL additionally ships the private *GenerateCore host loop (inv. b).
        self.assertTrue(scanned["qwen3_vl"].private_generate_loop)
        # A clean model that keeps a *GenerateCore example helper (qwen3_5) is NOT
        # flagged — DEVICE classification means it never enters the (b) gate.
        self.assertEqual(scanned["qwen3_5_moe"].classification, "DEVICE")

    def test_refuse_stub_is_skipped_on_tree(self) -> None:
        # kimi_k3's ForwardDevice is VT_CHECK(false); it must be REFUSE, not HOST.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        self.assertEqual(scanned["kimi_k3"].classification, "REFUSE")
        self.assertNotIn("kimi_k3", drift_models(scanned, set()))

    def test_a_new_off_runner_model_would_fail(self) -> None:
        # Mutation: a NEW model landing with a HostLogits forward and no allowlist
        # entry must trip the gate.
        scanned = dict(mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR))
        allowlisted = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        scanned["brand_new_arch"] = route("brand_new_arch", "HOST")
        self.assertIn("brand_new_arch", drift_models(scanned, allowlisted))

    def test_removing_allowlist_entry_would_fail(self) -> None:
        # Mutation: dropping a known off-framework model from the allowlist WITHOUT
        # routing it through the runner must re-open the gate (the enforcement teeth).
        # The example is DERIVED from the tree, never hardcoded, so routing a model
        # (retiring its entry) closes the gate rather than breaking this test.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        allowlisted = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        exposed = set(drift_models(scanned, set()))
        host_models = {n for n, r in scanned.items() if r.classification == "HOST"}
        # Emptying the allowlist must expose every HOST model,
        self.assertEqual(host_models - exposed, set())
        # and the allowlist must be load-bearing: it suppresses at least one model
        # the checker really detects (else the green gate is vacuous).
        self.assertTrue(
            exposed & allowlisted,
            "the allowlist suppresses nothing the checker detects; either the "
            "detector regressed or every allowlist entry is now stale",
        )


if __name__ == "__main__":
    unittest.main()
