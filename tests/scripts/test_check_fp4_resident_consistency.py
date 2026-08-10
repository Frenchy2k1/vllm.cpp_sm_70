#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-fp4-resident-consistency.py.

The checker is the ONLY mechanism that bites when the `ResidentNvfp4` copy in
src/vllm/model_executor/models/qwen3_5.cpp silently drops part of the
ENG-LOAD-DIRECT-UPLOAD post-upload step (issue #150): that copy sits in an
anonymous namespace inside an 8.5k-line translation unit, so no test can call it.
These cases mutate a miniature of the real function to prove the checker actually
detects each dropped statement, and then run it against the LIVE tree so a rename
or a refactor that makes the invariant unreachable cannot pass silently.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-fp4-resident-consistency.py"
SPEC = importlib.util.spec_from_file_location("check_fp4_resident_consistency", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

function_bodies = mod.function_bodies
adopted_buffers = mod.adopted_buffers
body_violations = mod.body_violations
file_violations = mod.file_violations


# A miniature of the real ResidentNvfp4 carrying the full three-statement step for
# both buffers. Every mutation below removes exactly one thing from it.
GOOD = """
Nvfp4Dev ResidentNvfp4(Dev d, const Nvfp4Weight& w) {
  if (!w.d_packed) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    vllm::load_stats::AddDeviceUpload(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.d_packed = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    w.packed.d_dev = w.d_packed;
    AdoptDeviceBytesAsHost(d.b, w.packed);
  }
  if (!w.d_scale) {
    const size_t sb = w.scale.bytes.size();
    void* p = d.b.Alloc(sb);
    vllm::load_stats::AddDeviceUpload(sb);
    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);
    Backend* bk = &d.b;
    w.d_scale = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    w.scale.d_dev = w.d_scale;
    AdoptDeviceBytesAsHost(d.b, w.scale);
  }
  Nvfp4Dev r;
  return r;
}
"""


def body_of(text: str) -> str:
    bodies = function_bodies(text)
    assert len(bodies) == 1, f"expected one body, got {len(bodies)}"
    return bodies[0][1]


class BodyExtractionTests(unittest.TestCase):
    def test_finds_the_definition(self) -> None:
        bodies = function_bodies(GOOD)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][0], 2)
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.scale);", bodies[0][1])

    def test_call_sites_are_not_definitions(self) -> None:
        # `ResidentNvfp4(d, w)` appears dozens of times in qwen3_5.cpp. Matching a
        # call would give the checker a body it cannot reason about.
        self.assertEqual(function_bodies("Nvfp4Dev dw = ResidentNvfp4(d, w);"), [])

    def test_sibling_overloads_are_out_of_scope(self) -> None:
        # ResidentNvfp4Qkv / GateUp / Alpha / ScaleSwizzled build MERGED device
        # operands out of several borrowed tensors; an adoption is not representable
        # there and the row explicitly leaves them named-but-unclaimed. None of them
        # may be pulled in by the name prefix they share with ResidentNvfp4.
        for text in (
            "Nvfp4QkvDev ResidentNvfp4Qkv(Dev d, const FullAttnLayerWeights& w) { return r; }",
            "Nvfp4GateUpDev ResidentNvfp4GateUp(Dev d, const DenseMlpWeights& w) { return r; }",
            "Tensor ResidentNvfp4Alpha(Dev d, const Nvfp4Weight& w) { return t; }",
            "Tensor ResidentNvfp4ScaleSwizzled(Dev d, const Nvfp4Weight& w) { return t; }",
        ):
            self.assertEqual(function_bodies(text), [], text)

    def test_nested_braces_do_not_truncate_the_body(self) -> None:
        # The real body contains lambdas and if-blocks; brace matching must reach the
        # end, or the trailing `scale` statements would look absent.
        self.assertIn("w.scale.d_dev", body_of(GOOD))


class InvariantTests(unittest.TestCase):
    def test_the_real_shape_passes(self) -> None:
        self.assertEqual(body_violations(body_of(GOOD)), [])

    def test_adopted_buffers_are_resolved_by_name(self) -> None:
        self.assertEqual(sorted(adopted_buffers(body_of(GOOD))), ["packed", "scale"])

    def test_dropping_both_upload_counters_fails(self) -> None:
        # Mutation (1): the upload stops being accounted. These bytes were BORROWED
        # from the shard mmap, so no other counter can ever see them.
        text = GOOD.replace("    vllm::load_stats::AddDeviceUpload(pb);\n", "", 1)
        text = text.replace("    vllm::load_stats::AddDeviceUpload(sb);\n", "", 1)
        problems = body_violations(body_of(text))
        self.assertTrue(any("NOT counted" in p for p in problems), problems)

    def test_dropping_the_packed_publication_fails(self) -> None:
        # Mutation (2): `d_dev` is never published, so AdoptDeviceBytesAsHost returns
        # immediately and the adoption below it is a silent no-op.
        text = GOOD.replace("    w.packed.d_dev = w.d_packed;\n", "", 1)
        problems = body_violations(body_of(text))
        self.assertTrue(
            any("`packed` does not PUBLISH" in p for p in problems), problems
        )
        self.assertFalse(any("`scale` does not PUBLISH" in p for p in problems), problems)

    def test_dropping_the_scale_publication_fails(self) -> None:
        text = GOOD.replace("    w.scale.d_dev = w.d_scale;\n", "", 1)
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` does not PUBLISH" in p for p in problems), problems)

    def test_dropping_the_packed_adoption_fails(self) -> None:
        # Mutation (3): the post-upload residency step is skipped, so the consumed
        # source pages are never released and the model stays resident twice.
        text = GOOD.replace("    AdoptDeviceBytesAsHost(d.b, w.packed);\n", "", 1)
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` skips" in p for p in problems), problems)
        self.assertFalse(any("`scale` skips" in p for p in problems), problems)

    def test_dropping_the_scale_adoption_fails(self) -> None:
        text = GOOD.replace("    AdoptDeviceBytesAsHost(d.b, w.scale);\n", "", 1)
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` skips" in p for p in problems), problems)

    def test_dropping_all_six_statements_fails(self) -> None:
        # The exact revert a fresh reviewer applied to BOTH copies, which left every
        # existing suite green. Five distinct violations here.
        text = GOOD
        for line in (
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    w.packed.d_dev = w.d_packed;\n",
            "    w.scale.d_dev = w.d_scale;\n",
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, w.scale);\n",
        ):
            self.assertEqual(text.count(line), 1, line)
            text = text.replace(line, "", 1)
        self.assertEqual(len(body_violations(body_of(text))), 5)

    def test_publishing_after_the_adoption_fails(self) -> None:
        # Mutation (4): the ORDERING. Publishing d_dev after the adopt call leaves the
        # adoption looking at a null handle — present, and useless.
        text = GOOD.replace(
            "    w.packed.d_dev = w.d_packed;\n    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n    w.packed.d_dev = w.d_packed;\n",
            1,
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("publishes `d_dev` AFTER" in p for p in problems), problems)

    def test_a_missing_definition_is_a_violation_not_a_pass(self) -> None:
        # Renaming or deleting a copy must go RED, never silently vacuous.
        problems = file_violations("int main() { return 0; }", "some/file.cpp")
        self.assertEqual(len(problems), 1)
        self.assertIn("no `ResidentNvfp4", problems[0])


class LiveTreeTests(unittest.TestCase):
    def test_the_checker_passes_on_the_current_tree(self) -> None:
        r = subprocess.run(
            [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=ROOT
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_both_copies_are_actually_reached(self) -> None:
        # The gate is worthless if SOURCES drifts off the real files: it would print
        # OK over nothing. Assert both named files really contain a checked body.
        for rel in mod.SOURCES:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            self.assertEqual(len(function_bodies(text)), 1, str(rel))


if __name__ == "__main__":
    unittest.main()
