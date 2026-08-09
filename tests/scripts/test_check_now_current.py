#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-now-current.py.

NOW.md only works if it is short and true. The mutations therefore cover both
failure directions: a digest that grew back into a status log, and a change that
moved what is live without refreshing the digest.
"""

from __future__ import annotations

import importlib.util
import csv
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


now = _load("now_current", "scripts/check-now-current.py")
state_record = _load("state_record_for_now", "scripts/state_record.py")


VALID = "\n".join(
    [
        "# NOW",
        "",
        "<!-- now-updated: 2026-08-04 -->",
        "",
        "## Live claims",
        "",
        "| Claim | State | Next |",
        "|---|---|---|",
        "| Thing | ACTIVE | Run the gate |",
        "",
        "## Current gate",
        "",
        "Token-exact against the pinned oracle, then every speed axis.",
        "",
        "## Next actions",
        "",
        "1. Do the next thing.",
    ]
)


class Baseline(unittest.TestCase):
    def test_valid_digest_passes(self) -> None:
        self.assertEqual(now.structure_errors(VALID), [])

    def test_headings_are_case_insensitive(self) -> None:
        self.assertEqual(now.structure_errors(VALID.replace("## Live claims", "## LIVE CLAIMS")), [])


class StructureMutations(unittest.TestCase):
    def test_missing_stamp_is_rejected(self) -> None:
        text = VALID.replace("<!-- now-updated: 2026-08-04 -->", "")
        self.assertTrue(any("freshness stamp" in e for e in now.structure_errors(text)))

    def test_malformed_stamp_is_rejected(self) -> None:
        text = VALID.replace("2026-08-04 -->", "yesterday -->")
        self.assertTrue(any("freshness stamp" in e for e in now.structure_errors(text)))

    def test_each_required_heading_is_enforced(self) -> None:
        for heading in now.REQUIRED_HEADINGS:
            text = VALID.replace(f"## {heading.capitalize()}", "## Something else")
            with self.subTest(heading=heading):
                self.assertTrue(
                    any(heading in e for e in now.structure_errors(text)),
                    f"dropping '{heading}' was not rejected",
                )

    def test_overlong_digest_is_rejected(self) -> None:
        text = VALID + "\n" + "\n".join(f"- line {i}" for i in range(now.MAX_LINES))
        self.assertTrue(any("line budget" in e for e in now.structure_errors(text)))

    def test_oversized_entry_is_rejected(self) -> None:
        text = VALID + "\n- " + "x" * (now.MAX_ENTRY_CHARS + 1)
        self.assertTrue(
            any("character budget" in e for e in now.structure_errors(text))
        )


class FreshnessMutations(unittest.TestCase):
    def event(
        self,
        *,
        kind: str = "checkpoint",
        outcome: str = "checkpoint",
        subjects: str = "ROW-ONE",
        next_action: str = "Run the next gate",
    ):
        return state_record.Event(
            "STATE-20260808T143000-001",
            "2026-08-08T14:30:00Z",
            kind,
            subjects,
            "verification",
            outcome,
            "",
            "",
            "",
            ".agents/state-events/2026-08/STATE-20260808T143000-001.md",
            "",
            "Checkpoint",
            next_action,
        )

    def test_live_event_without_digest_refresh_is_rejected(self) -> None:
        errors = now.freshness_errors(
            {".agents/state-index/2026-08-001.csv"},
            [self.event()],
            now_text=VALID,
        )
        self.assertTrue(any("did not" in e for e in errors))

    def test_live_event_with_digest_refresh_passes(self) -> None:
        self.assertEqual(
            now.freshness_errors(
                {".agents/state-index/2026-08-001.csv", ".agents/NOW.md"},
                [self.event()],
                now_text=VALID,
            ),
            [],
        )

    def test_migration_and_evidence_only_correction_do_not_require_refresh(self) -> None:
        for event in (
            self.event(kind="legacy_import", outcome="", subjects="", next_action=""),
            self.event(kind="correction", outcome="superseded", next_action="None"),
        ):
            with self.subTest(kind=event.kind):
                self.assertEqual(
                    now.freshness_errors(
                        {".agents/state-index/2026-08-001.csv"},
                        [event],
                        now_text=VALID,
                    ),
                    [],
                )

    def test_live_correction_without_digest_refresh_is_rejected(self) -> None:
        for outcome in ("checkpoint", "failed", "blocked"):
            with self.subTest(outcome=outcome):
                errors = now.freshness_errors(
                    {".agents/state-index/2026-08-001.csv"},
                    [self.event(kind="correction", outcome=outcome)],
                    now_text=VALID,
                )
                self.assertTrue(any("did not" in error for error in errors), errors)

    def test_terminal_correction_refreshes_when_subject_is_live(self) -> None:
        event = self.event(
            kind="correction", outcome="closed", subjects="Thing", next_action="None"
        )
        self.assertTrue(
            now.freshness_errors(
                {".agents/state-index/2026-08-001.csv"},
                [event],
                now_text=VALID,
            )
        )

    def test_terminal_event_refreshes_when_subject_is_live(self) -> None:
        event = self.event(outcome="landed", subjects="Thing", next_action="None")
        self.assertTrue(
            now.freshness_errors(
                {".agents/state-index/2026-08-001.csv"},
                [event],
                now_text=VALID,
            )
        )

    def test_terminal_unadvertised_event_with_no_next_action_is_forensic(self) -> None:
        event = self.event(outcome="closed", subjects="ROW-ABSENT", next_action="None")
        self.assertEqual(
            now.freshness_errors(
                {".agents/state-index/2026-08-001.csv"},
                [event],
                now_text=VALID,
            ),
            [],
        )

    def test_unrelated_change_is_not_forced_to_refresh(self) -> None:
        self.assertEqual(now.freshness_errors({"README.md"}), [])


class LiveTree(unittest.TestCase):
    def test_repository_digest_is_valid(self) -> None:
        self.assertTrue(now.NOW.exists(), f"{now.NOW_PATH} is missing")
        self.assertEqual(
            now.structure_errors(now.NOW.read_text(encoding="utf-8")), []
        )


class NonEventChangesAreNotAppends(unittest.TestCase):
    def test_index_rewrite_without_appended_events_is_exempt(self) -> None:
        self.assertEqual(
            now.freshness_errors(
                {".agents/state-index/2026-08-001.csv"}, [], now_text=VALID
            ),
            [],
        )


class CommittedRangeIntegration(unittest.TestCase):
    def test_committed_qualifying_event_without_now_refresh_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "NOW Test"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "now@example.invalid"], cwd=root, check=True)
            digest = root / ".agents/NOW.md"
            index = root / ".agents/state-index/2026-08-001.csv"
            index.parent.mkdir(parents=True)
            digest.write_text(VALID, encoding="utf-8")
            with index.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerow(state_record.EVENT_HEADER)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            base = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()

            event = FreshnessMutations().event()
            with index.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    [getattr(event, field) for field in state_record.EVENT_HEADER]
                )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "uncoupled event"], cwd=root, check=True)
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            stderr = io.StringIO()
            with (
                mock.patch.object(now, "ROOT", root),
                mock.patch.object(now, "NOW", digest),
                mock.patch.object(
                    sys,
                    "argv",
                    ["check-now-current.py", "--base", base, "--head", head],
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = now.main()
            self.assertEqual(code, 1)
            self.assertIn(event.event_id, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
