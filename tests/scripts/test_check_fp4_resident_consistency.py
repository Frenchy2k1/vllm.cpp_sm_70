#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-fp4-resident-consistency.py.

The checker is the ONLY mechanism that bites when the `ResidentNvfp4` copy in
src/vllm/model_executor/models/qwen3_5.cpp silently drops part of the
ENG-LOAD-DIRECT-UPLOAD post-upload step (issue #150): that copy sits in an
anonymous namespace inside an 8.5k-line translation unit, so no test can call it.
These cases mutate a miniature of the real function to prove the checker actually
detects each dropped statement, and then run it against the LIVE tree so a rename
or a refactor that makes the invariant unreachable cannot pass silently.

EVERY MUTATION DROPS EXACTLY ONE THING. The first version of this suite only ever
dropped BOTH `AddDeviceUpload` calls at once, which is why it could not see that
`_COUNT` was matched body-wide: one surviving call satisfied the clause for both
buffers, so dropping exactly one of them passed. The drop-exactly-one cases below
are the regression for that.
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
buffer_block = mod.buffer_block
body_violations = mod.body_violations
file_violations = mod.file_violations


# A miniature of the real ResidentNvfp4 carrying the full step for both buffers,
# each inside its own `if (!w.d_<buf>)` upload block. Every mutation below changes
# exactly one thing in it.
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
    return bodies[0][2]


def mutate(old: str, new: str = "", text: str = GOOD) -> str:
    """Replace `old` ONCE, asserting it was unique first — a near-duplicate anchor
    silently mutating the wrong statement is how a mutation suite lies."""
    assert text.count(old) == 1, f"anchor is not unique ({text.count(old)}): {old!r}"
    return text.replace(old, new, 1)


class BodyExtractionTests(unittest.TestCase):
    def test_finds_the_definition(self) -> None:
        bodies = function_bodies(GOOD)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][0], 2)
        self.assertEqual(bodies[0][1], "w")  # the captured weight parameter name
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.scale);", bodies[0][2])

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

    def test_a_renamed_weight_parameter_is_followed(self) -> None:
        # The clauses are bound to the parameter NAME, so a copy that calls it `nw`
        # must still pass on its own terms rather than going vacuously red.
        text = GOOD.replace("Nvfp4Weight& w)", "Nvfp4Weight& nw)").replace("w.", "nw.")
        bodies = function_bodies(text)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][1], "nw")
        self.assertEqual(body_violations(bodies[0][2], bodies[0][1]), [])


class BufferBlockTests(unittest.TestCase):
    """The scoping that makes the invariant PER BUFFER rather than body-wide."""

    def test_each_buffer_gets_its_own_block(self) -> None:
        body = body_of(GOOD)
        packed = buffer_block(body, "w", "packed")
        scale = buffer_block(body, "w", "scale")
        self.assertIsNotNone(packed)
        self.assertIsNotNone(scale)
        # Neither block may contain the other buffer's statements — that overlap is
        # precisely what let one AddDeviceUpload satisfy both clauses.
        assert packed is not None and scale is not None
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.packed);", packed)
        self.assertNotIn("w.scale", packed)
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.scale);", scale)
        self.assertNotIn("w.packed", scale)

    def test_a_missing_upload_block_is_a_violation(self) -> None:
        # A restructure the checker cannot scope must go RED and say so, never pass
        # by silently falling back to body-wide matching.
        text = mutate("if (!w.d_scale) {", "if (true) {")
        problems = body_violations(body_of(text))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("no `if (!w.d_scale)` upload block", problems[0])


class InvariantTests(unittest.TestCase):
    def test_the_real_shape_passes(self) -> None:
        self.assertEqual(body_violations(body_of(GOOD)), [])

    # --- (a) COUNTED, per buffer ------------------------------------------------
    def test_dropping_ONLY_the_packed_upload_counter_fails(self) -> None:
        # THE REGRESSION for the body-wide `_COUNT` bug: the surviving `scale`
        # counter must not satisfy `packed`.
        text = mutate("    vllm::load_stats::AddDeviceUpload(pb);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed`'s host->device upload is NOT counted" in p for p in problems), problems)
        self.assertFalse(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_dropping_ONLY_the_scale_upload_counter_fails(self) -> None:
        text = mutate("    vllm::load_stats::AddDeviceUpload(sb);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)
        self.assertFalse(any("`packed`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_dropping_both_upload_counters_fails(self) -> None:
        # Mutation (1): the upload stops being accounted. These bytes were BORROWED
        # from the shard mmap, so no other counter can ever see them.
        text = mutate("    vllm::load_stats::AddDeviceUpload(pb);\n")
        text = mutate("    vllm::load_stats::AddDeviceUpload(sb);\n", "", text)
        problems = body_violations(body_of(text))
        self.assertEqual(sum("is NOT counted" in p for p in problems), 2, problems)

    def test_counting_a_literal_zero_fails(self) -> None:
        # Present-but-meaningless: the call survives review by eye and accounts for
        # nothing. The argument has to be THIS buffer's byte count.
        text = mutate(
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    vllm::load_stats::AddDeviceUpload(0);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_counting_the_other_buffers_size_fails(self) -> None:
        # `pb` is not bound inside the `scale` block, so charging the scale upload
        # to the packed byte count is not a pass either.
        text = mutate(
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)

    # --- (b) COPIED, per buffer -------------------------------------------------
    def test_uploading_the_other_buffers_bytes_fails(self) -> None:
        text = mutate(
            "    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);\n",
            "    d.b.Copy(d.q, p, w.packed.bytes.data(), sb);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("does not upload `w.scale.bytes.data()`" in p for p in problems), problems)

    # --- (c) PUBLISHED, per buffer ----------------------------------------------
    def test_dropping_the_packed_publication_fails(self) -> None:
        # Mutation (2): `d_dev` is never published, so AdoptDeviceBytesAsHost returns
        # immediately and the adoption below it is a silent no-op.
        text = mutate("    w.packed.d_dev = w.d_packed;\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` does not PUBLISH" in p for p in problems), problems)
        self.assertFalse(any("`scale` does not PUBLISH" in p for p in problems), problems)

    def test_dropping_the_scale_publication_fails(self) -> None:
        text = mutate("    w.scale.d_dev = w.d_scale;\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` does not PUBLISH" in p for p in problems), problems)

    def test_publishing_a_null_handle_fails(self) -> None:
        # The statement is still there; it publishes nothing. AdoptDeviceBytesAsHost
        # keys on `d_dev` and returns on null, so this is the deletion in disguise.
        text = mutate("    w.packed.d_dev = w.d_packed;\n", "    w.packed.d_dev = nullptr;\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` does not PUBLISH" in p for p in problems), problems)

    def test_publishing_the_other_buffers_handle_fails(self) -> None:
        text = mutate("    w.scale.d_dev = w.d_scale;\n", "    w.scale.d_dev = w.d_packed;\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` does not PUBLISH" in p for p in problems), problems)

    # --- (d) ADOPTED, per buffer ------------------------------------------------
    def test_dropping_the_packed_adoption_fails(self) -> None:
        # Mutation (3): the post-upload residency step is skipped, so the consumed
        # source pages are never released and the model stays resident twice.
        text = mutate("    AdoptDeviceBytesAsHost(d.b, w.packed);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` skips" in p for p in problems), problems)
        self.assertFalse(any("`scale` skips" in p for p in problems), problems)

    def test_dropping_the_scale_adoption_fails(self) -> None:
        text = mutate("    AdoptDeviceBytesAsHost(d.b, w.scale);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` skips" in p for p in problems), problems)

    def test_adopting_another_objects_buffer_fails(self) -> None:
        # The call is present and adopts a `packed` — just not THIS weight's.
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, other.packed);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` skips" in p for p in problems), problems)

    # --- (e) ORDERED ------------------------------------------------------------
    def test_publishing_after_the_adoption_fails(self) -> None:
        # Mutation (4): the ORDERING. Publishing d_dev after the adopt call leaves the
        # adoption looking at a null handle — present, and useless.
        text = mutate(
            "    w.packed.d_dev = w.d_packed;\n    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n    w.packed.d_dev = w.d_packed;\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("publishes `d_dev` AFTER" in p for p in problems), problems)

    # --- the whole revert -------------------------------------------------------
    def test_dropping_all_six_statements_fails(self) -> None:
        # The exact revert a fresh reviewer applied to BOTH copies, which left every
        # existing suite green. Six distinct violations here, three per buffer.
        text = GOOD
        for line in (
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    w.packed.d_dev = w.d_packed;\n",
            "    w.scale.d_dev = w.d_scale;\n",
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, w.scale);\n",
        ):
            text = mutate(line, "", text)
        self.assertEqual(len(body_violations(body_of(text))), 6)

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
        # OK over nothing. Assert both named files really contain a checked body,
        # with both per-buffer upload blocks inside it.
        for rel in mod.SOURCES:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            bodies = function_bodies(text)
            self.assertEqual(len(bodies), 1, str(rel))
            _, recv, body = bodies[0]
            for buffer in mod.BUFFERS:
                self.assertIsNotNone(buffer_block(body, recv, buffer), f"{rel}:{buffer}")

    def test_dropping_ONE_live_upload_counter_goes_red(self) -> None:
        # The finding, against the REAL qwen3_5.cpp text rather than the miniature:
        # the previous body-wide checker returned 0 here.
        rel = Path("src/vllm/model_executor/models/qwen3_5.cpp")
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        for anchor in (
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
        ):
            self.assertEqual(text.count(anchor), 1, anchor)
            self.assertNotEqual(
                file_violations(text.replace(anchor, "", 1), str(rel)), [], anchor
            )


if __name__ == "__main__":
    unittest.main()
