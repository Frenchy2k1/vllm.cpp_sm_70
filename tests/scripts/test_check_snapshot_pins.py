"""GATE-PIN-UNPINNED-SNAPSHOTS (#471) — the checker's own gate.

A checker that cannot be shown failing has proven nothing, and a checker whose
pattern is quietly narrowed goes green while the defect walks back in. Each case
here synthesises a tree, so none of them depends on what the repo happens to
contain today.
"""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CHECKER = REPO_ROOT / "scripts" / "check-snapshot-pins.py"

_spec = importlib.util.spec_from_file_location("check_snapshot_pins", CHECKER)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class SnapshotPinChecker(unittest.TestCase):
    def _tree(self, tmp: pathlib.Path, body: str, rel: str) -> pathlib.Path:
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return target

    def test_an_unpinned_resolver_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, mod.UNPINNED_FIXTURE, "tests/parity/test_new_gate.cpp")
            problems = mod.check(tmp)
        self.assertTrue(
            any("UNPINNED checkpoint resolution" in p for p in problems),
            "a new directory_iterator over <repo>/snapshots/ must fail the gate",
        )

    def test_a_pinned_resolver_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, mod.PINNED_FIXTURE, "tests/parity/test_new_gate.cpp")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_shard_iteration_inside_a_resolved_dir_is_not_a_resolution(self) -> None:
        """The converted gates still iterate their OWN checkpoint for shards.

        If that tripped the checker, the only way to green would be to weaken the
        pattern, which is how a gate stops gating.
        """
        body = (
            '#include <filesystem>\n'
            'namespace fs = std::filesystem;\n'
            '// resolved via parity::HfSnapshot, then enumerate shards:\n'
            'void Shards(const std::string& dir) {\n'
            '  std::error_code ec;\n'
            '  for (const auto& e : fs::directory_iterator(dir, ec)) (void)e;\n'
            '}\n'
            '// the word snapshots appears in prose above\n'
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, body, "tests/parity/test_shards.cpp")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_a_commented_out_resolver_is_not_a_resolution(self) -> None:
        body = (
            '#include <filesystem>\n'
            'namespace fs = std::filesystem;\n'
            '// for (const auto& e : fs::directory_iterator(snaps, ec)) {}\n'
            '// snapshots\n'
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, body, "tests/parity/test_comment.cpp")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_a_stale_ledger_entry_fails(self) -> None:
        """A ledger that outlives its debt starts excusing unreviewed code."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            (tmp / "tests").mkdir()
            problems = mod.check(tmp)
        self.assertTrue(
            any("STALE ledger entry" in p for p in problems),
            "every ledger line must still name a real unpinned resolution",
        )

    def test_the_repo_itself_is_clean(self) -> None:
        self.assertEqual([], mod.check(REPO_ROOT))

    def test_the_ledger_holds_no_file_this_row_pinned(self) -> None:
        """The eight gates GATE-PIN-UNPINNED-SNAPSHOTS converted must not reappear.

        Re-adding one to the ledger would be the cheapest way to un-fix this row,
        so it is spelled out rather than left to the STALE check.
        """
        pinned_by_this_row = {
            "tests/parity/test_qwen3_dflash_draft_parity.cpp",
            "tests/parity/test_qwen3_dflash_kvprep_parity.cpp",
            "tests/parity/test_qwen27_dflash_spec_decode.cpp",
            "tests/parity/test_op_parity.cpp",
            "tests/parity/test_qwen36_paged_engine.cpp",
            "tests/parity/test_qwen36_async_serving.cpp",
            "tests/parity/test_qwen36_spec_decode.cpp",
            "tests/vllm/test_qwen36_weights.cpp",
        }
        self.assertEqual(set(), pinned_by_this_row & set(mod.LEDGER))

    def test_every_header_pin_has_an_accessor(self) -> None:
        """A constant with no accessor is a pin nothing resolves through."""
        import re

        header = (REPO_ROOT / "tests/parity/hf_snapshot.h").read_text(encoding="utf-8")
        constants = re.findall(r"inline constexpr const char\* (k\w*Revision)\s*=", header)
        self.assertGreaterEqual(len(constants), 3, constants)
        for name in constants:
            with self.subTest(constant=name):
                self.assertEqual(
                    1,
                    len(re.findall(rf"\b{name}\b,", header)),
                    f"{name} is never passed to HfSnapshot; it pins nothing",
                )


if __name__ == "__main__":
    unittest.main()
