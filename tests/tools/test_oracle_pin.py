"""The harness's oracle identity must equal the pin record, and must REFUSE the rollback.

Row `BENCH-ORACLE-PIN-RECONCILE`, issue #520, spec
`.agents/specs/bench-oracle-pin-reconcile.md`.

These tests exist because the previous constants were a second copy of
`.agents/upstream-sync.md` and the copy drifted for 17 days, during which the
gate did not merely prefer the old oracle -- it RAISED on the pinned one. Test 1
makes that drift impossible to reintroduce silently. Tests 2 and 3 are the #375
mutation: they fail if the commit term is weakened or deleted, which is the only
term that separates the pin from a rollback that runs, is deterministic, and
reports a perfectly clean version number.
"""

from __future__ import annotations

import pathlib
import re
import tempfile
import unittest

from tools.bench.serve_low_common import (
    FLASHINFER_VERSION,
    HarnessError,
    VLLM_COMMIT,
    VLLM_DISTRIBUTION_VERSION,
    VLLM_ORACLE_VERSION,
    assert_oracle_commit,
    read_parity_pin,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIN_RECORD = ROOT / ".agents" / "upstream-sync.md"

# The rollback's EXACT measured strings (dgx, ~/venvs/vllm-oracle-v0.25.0-stage,
# 2026-08-12). Hardcoded on purpose: this is the artifact the harness must
# refuse, so it may not be derived from whatever the constants happen to say.
ROLLBACK_RUNTIME_VERSION = "0.25.0"
ROLLBACK_FLASHINFER_VERSION = "0.6.13"
ROLLBACK_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"


class ParityPinRecordTests(unittest.TestCase):
    def test_constants_are_the_pin_record_not_a_copy(self) -> None:
        """Re-parsed independently of the loader, so a loader bug cannot agree with itself."""

        text = PIN_RECORD.read_text(encoding="utf-8")
        blocks = re.findall(r"^```parity-pin$\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)
        self.assertEqual(len(blocks), 1, "exactly one parity-pin block must exist")
        fields = dict(
            (part.strip() for part in line.split("=", 1))
            for line in blocks[0].splitlines()
            if line.strip()
        )
        self.assertEqual(VLLM_COMMIT, fields["vllm_commit"])
        self.assertEqual(VLLM_ORACLE_VERSION, fields["vllm_runtime_version"])
        self.assertEqual(VLLM_DISTRIBUTION_VERSION, fields["vllm_distribution_version"])
        self.assertEqual(FLASHINFER_VERSION, fields["flashinfer_version"])

    def test_the_rollback_is_not_the_pin(self) -> None:
        """Guards the reconcile itself: these are the values that were enforced."""

        self.assertNotEqual(VLLM_ORACLE_VERSION, ROLLBACK_RUNTIME_VERSION)
        self.assertNotEqual(FLASHINFER_VERSION, ROLLBACK_FLASHINFER_VERSION)
        self.assertNotEqual(VLLM_COMMIT, ROLLBACK_COMMIT)

    def test_metadata_and_runtime_strings_differ_on_the_pin(self) -> None:
        """The reason `metadata == runtime == CONST` had to be replaced, pinned as a fact."""

        self.assertNotEqual(VLLM_DISTRIBUTION_VERSION, VLLM_ORACLE_VERSION)
        self.assertTrue(VLLM_DISTRIBUTION_VERSION.startswith(VLLM_ORACLE_VERSION))


class ParityPinLoaderTests(unittest.TestCase):
    def _write(self, body: str) -> pathlib.Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = pathlib.Path(directory.name) / "upstream-sync.md"
        path.write_text(body, encoding="utf-8")
        return path

    def test_missing_record_is_fatal(self) -> None:
        with self.assertRaisesRegex(HarnessError, "unreadable"):
            read_parity_pin(pathlib.Path("/nonexistent/upstream-sync.md"))

    def test_absent_block_is_fatal_not_defaulted(self) -> None:
        path = self._write("# no pin here\n")
        with self.assertRaisesRegex(HarnessError, "exactly one"):
            read_parity_pin(path)

    def test_two_blocks_are_fatal(self) -> None:
        block = "```parity-pin\nvllm_commit = " + "a" * 40 + "\n```\n"
        with self.assertRaisesRegex(HarnessError, "exactly one"):
            read_parity_pin(self._write(block + block))

    def test_unknown_field_is_fatal(self) -> None:
        path = self._write("```parity-pin\nvllm_branch = main\n```\n")
        with self.assertRaisesRegex(HarnessError, "unknown parity-pin field"):
            read_parity_pin(path)

    def test_malformed_line_is_fatal(self) -> None:
        path = self._write("```parity-pin\nvllm_commit\n```\n")
        with self.assertRaisesRegex(HarnessError, "malformed"):
            read_parity_pin(path)

    def test_missing_field_is_fatal(self) -> None:
        path = self._write("```parity-pin\nvllm_commit = " + "a" * 40 + "\n```\n")
        with self.assertRaisesRegex(HarnessError, "omits"):
            read_parity_pin(path)

    def test_abbreviated_commit_is_fatal(self) -> None:
        """A short SHA would silently weaken the prefix check below."""

        path = self._write(
            "```parity-pin\n"
            "vllm_commit = 555967922\n"
            "vllm_runtime_version = 0.1\n"
            "vllm_distribution_version = 0.1\n"
            "flashinfer_version = 0.1\n"
            "```\n"
        )
        with self.assertRaisesRegex(HarnessError, "full 40-hex SHA"):
            read_parity_pin(path)

    def test_committed_record_parses(self) -> None:
        fields = read_parity_pin(PIN_RECORD)
        self.assertEqual(fields["vllm_commit"], VLLM_COMMIT)


class OracleCommitAssertionTests(unittest.TestCase):
    """THE #375 MUTATION. Delete the commit term and every test here fails."""

    def test_the_pin_is_accepted(self) -> None:
        assert_oracle_commit(VLLM_ORACLE_VERSION)
        # The distribution string carries the same segment plus a suffix.
        assert_oracle_commit(VLLM_DISTRIBUTION_VERSION)

    def test_the_rollback_is_refused(self) -> None:
        """`0.25.0` runs, is deterministic, and looks healthy. It is not the pin."""

        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit(ROLLBACK_RUNTIME_VERSION)

    def test_a_version_with_no_commit_segment_is_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit("0.26.0.dev0")

    def test_a_different_commit_is_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit("0.23.1rc1.dev1511+ge24d1b24")

    def test_none_is_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit(None)

    def test_a_prefix_shorter_than_seven_is_refused(self) -> None:
        """Otherwise `+g5` would match, and a one-character oracle check is no check."""

        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit(f"0.1+g{VLLM_COMMIT[:6]}")

    def test_a_longer_prefix_of_the_pin_is_accepted(self) -> None:
        assert_oracle_commit(f"0.1+g{VLLM_COMMIT[:12]}")
        assert_oracle_commit(f"0.1+g{VLLM_COMMIT}")


if __name__ == "__main__":
    unittest.main()
