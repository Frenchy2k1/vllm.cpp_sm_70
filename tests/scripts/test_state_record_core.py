#!/usr/bin/env python3
"""Focused behavior tests for structured state-record parsing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import state_record
except ModuleNotFoundError:
    state_record = None


class ManifestParsingTests(unittest.TestCase):
    def write_manifest(self, root: Path, rows: list[str]) -> None:
        (root / ".agents").mkdir(exist_ok=True)
        (root / ".agents/state.csv").write_text(
            "schema_version,shard_id,index_path,event_prefix\n"
            + "".join(f"{row}\n" for row in rows),
            encoding="utf-8",
        )

    def test_invalid_manifest_header_is_rejected(self) -> None:
        self.assertIsNotNone(state_record, "state_record parser is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".agents").mkdir()
            (root / ".agents/state.csv").write_text(
                "wrong,header\n", encoding="utf-8"
            )

            shards, errors = state_record.parse_manifest(root)

            self.assertEqual(shards, [])
            self.assertTrue(any("header" in error for error in errors), errors)

    def test_valid_manifest_row_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_manifest(
                root,
                [
                    "1,2026-08-001,.agents/state-index/2026-08-001.csv,"
                    ".agents/state-events/2026-08/"
                ],
            )

            shards, errors = state_record.parse_manifest(root)

            self.assertEqual(errors, [])
            self.assertEqual(len(shards), 1)
            self.assertEqual(shards[0].shard_id, "2026-08-001")

    def test_manifest_rejects_unterminated_quoted_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".agents").mkdir()
            (root / ".agents/state.csv").write_text(
                "schema_version,shard_id,index_path,event_prefix\n"
                "1,2026-08-001,.agents/state-index/2026-08-001.csv,"
                '".agents/state-events/2026-08/',
                encoding="utf-8",
            )

            shards, errors = state_record.parse_manifest(root)

            self.assertEqual(shards, [])
            self.assertTrue(any("parse manifest" in error for error in errors), errors)

    def test_manifest_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".agents").mkdir()
            (root / ".agents/state.csv").write_bytes(
                b"schema_version,shard_id,index_path,event_prefix\n"
                b"1,2026-08-001,.agents/state-index/2026-08-001.csv,"
                b".agents/state-events/2026-08/\n\xff"
            )

            shards, errors = state_record.parse_manifest(root)

            self.assertEqual(shards, [])
            self.assertTrue(any("utf-8" in error.lower() for error in errors), errors)

    def test_manifest_rejects_invalid_scalar_and_path_contracts(self) -> None:
        cases = {
            "schema": (
                "2,2026-08-001,.agents/state-index/2026-08-001.csv,"
                ".agents/state-events/2026-08/"
            ),
            "shard id": (
                "1,august,.agents/state-index/august.csv,"
                ".agents/state-events/2026-08/"
            ),
            "index path": (
                "1,2026-08-001,.agents/state-index/wrong.csv,"
                ".agents/state-events/2026-08/"
            ),
            "event prefix": (
                "1,2026-08-001,.agents/state-index/2026-08-001.csv,"
                ".agents/state-events/2026-07/"
            ),
            "control character": (
                "1,2026-08-001,.agents/state-index/2026-08-001.csv,"
                ".agents/state-events/2026-08/\x00"
            ),
        }
        for expected, row in cases.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_manifest(root, [row])

                shards, errors = state_record.parse_manifest(root)

                self.assertEqual(shards, [])
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_manifest_requires_increasing_shard_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_manifest(
                root,
                [
                    "1,2026-08-002,.agents/state-index/2026-08-002.csv,"
                    ".agents/state-events/2026-08/",
                    "1,2026-08-001,.agents/state-index/2026-08-001.csv,"
                    ".agents/state-events/2026-08/",
                ],
            )

            shards, errors = state_record.parse_manifest(root)

            self.assertEqual(shards, [])
            self.assertTrue(any("order" in error for error in errors), errors)

    def test_manifest_rejects_files_over_64_kib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".agents").mkdir()
            (root / ".agents/state.csv").write_bytes(b"x" * (64 * 1024 + 1))

            shards, errors = state_record.parse_manifest(root)

            self.assertEqual(shards, [])
            self.assertTrue(any("64 KiB" in error for error in errors), errors)


class EventParsingTests(unittest.TestCase):
    shard = state_record.Shard(
        "1",
        "2026-08-001",
        ".agents/state-index/2026-08-001.csv",
        ".agents/state-events/2026-08/",
    )
    header = (
        "event_id,occurred_at,kind,subject_ids,phase,outcome,commit,pr,spec,"
        "evidence_path,supersedes,summary,next_action\n"
    )

    def make_root(self, directory: str) -> Path:
        root = Path(directory)
        (root / ".agents/state-index").mkdir(parents=True)
        (root / ".agents/state-events/2026-08").mkdir(parents=True)
        return root

    def parse_events(self, root: Path):
        self.assertTrue(hasattr(state_record, "parse_events"), "event parser is missing")
        return state_record.parse_events(root, [self.shard])

    def valid_fields(self, event_id: str = "STATE-20260808T143000-001") -> list[str]:
        return [
            event_id,
            "2026-08-08T14:30:00Z",
            "checkpoint",
            "POL-STATE-STRUCTURED;state-record-structure-1",
            "verification",
            "passed",
            "a" * 40,
            "pr:166",
            "docs/superpowers/specs/2026-08-08-structured-state-record-design.md",
            f".agents/state-events/2026-08/{event_id}.md",
            "",
            "Structured state parser passed its focused gate.",
            "Continue with relational validation.",
        ]

    def write_event(self, root: Path, fields: list[str], body: str | None = None) -> None:
        (root / self.shard.index_path).write_text(
            self.header + ",".join(fields) + "\n", encoding="utf-8"
        )
        evidence = root / fields[9]
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            body
            or (
                "# Parser checkpoint\n"
                f"<!-- state-event: {fields[0]} -->\n\n"
                "## Context\nInput fixture.\n\n"
                "## Outcome\nPassed.\n\n"
                "## Evidence\nFocused test.\n\n"
                "## Next action\nContinue.\n"
            ),
            encoding="utf-8",
        )

    def test_valid_post_cutover_event_is_parsed(self) -> None:
        self.assertTrue(hasattr(state_record, "parse_events"), "event parser is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            self.write_event(root, fields)

            events, errors = self.parse_events(root)

            self.assertEqual(errors, [])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_id, fields[0])
            self.assertEqual(events[0].outcome, "passed")

    def test_event_index_requires_exact_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            (root / self.shard.index_path).write_text(
                "wrong,event,header\n", encoding="utf-8"
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("event header" in error for error in errors), errors)

    def test_event_index_rejects_unterminated_quoted_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            self.write_event_file(root, fields)
            (root / self.shard.index_path).write_text(
                self.header + ",".join(fields[:-1]) + ',"unterminated',
                encoding="utf-8",
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("parse event index" in error for error in errors), errors)

    def test_event_index_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            self.write_event_file(root, fields)
            row = ",".join(fields).encode("utf-8").replace(
                fields[11].encode("utf-8"), b"invalid-\xff-summary"
            )
            (root / self.shard.index_path).write_bytes(
                self.header.encode("utf-8") + row + b"\n"
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("utf-8" in error.lower() for error in errors), errors)

    def test_event_index_rejects_files_over_256_kib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            (root / self.shard.index_path).write_bytes(
                self.header.encode("utf-8") + b"x" * (256 * 1024)
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("256 KiB" in error for error in errors), errors)

    def test_event_index_rejects_more_than_512_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            rows: list[str] = []
            for number in range(1, 514):
                event_id = f"STATE-LEGACY-{number:06d}"
                evidence_path = f".agents/state-events/2026-08/{event_id}.md"
                fields = [
                    event_id,
                    "",
                    "legacy_import",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    evidence_path,
                    "",
                    "",
                    "",
                ]
                rows.append(",".join(fields))
                (root / evidence_path).write_bytes(
                    f"<!-- state-event: {event_id} -->\n".encode("utf-8")
                    + b"<!-- legacy-payload:begin -->\n"
                    + f"payload {number}\n".encode("utf-8")
                    + b"<!-- legacy-payload:end -->\n"
                )
            (root / self.shard.index_path).write_text(
                self.header + "\n".join(rows) + "\n", encoding="utf-8"
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("512-event" in error for error in errors), errors)

    def test_event_rejects_invalid_scalar_contracts(self) -> None:
        cases: list[tuple[str, int, str]] = [
            ("event ID", 0, "STATE-BAD"),
            ("disagrees", 0, "STATE-20260808T143001-001"),
            ("timestamp", 1, "2026-08-08 14:30"),
            ("kind", 2, "note"),
            ("subject", 3, ""),
            ("subject", 3, "POL GOOD"),
            ("phase", 4, "coding"),
            ("outcome", 5, "ok"),
            ("commit", 6, "abc123"),
            ("PR", 7, "#166"),
            ("evidence path", 9, ".agents/other/event.md"),
            ("summary", 11, "s" * 201),
            ("next action", 12, "n" * 241),
            ("control character", 11, "bad\x00summary"),
        ]
        for expected, index, value in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = self.make_root(directory)
                fields = self.valid_fields()
                fields[index] = value
                self.write_event(root, fields)

                events, errors = self.parse_events(root)

                self.assertEqual(events, [])
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_event_requires_matching_marker_and_sections(self) -> None:
        bodies = {
            "event marker": (
                "# Wrong\n<!-- state-event: STATE-20260808T143000-999 -->\n\n"
                "## Context\nX\n\n## Outcome\nX\n\n## Evidence\nX\n\n"
                "## Next action\nX\n"
            ),
            "section": (
                "# Incomplete\n<!-- state-event: STATE-20260808T143000-001 -->\n\n"
                "## Context\nX\n"
            ),
        }
        for expected, body in bodies.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = self.make_root(directory)
                fields = self.valid_fields()
                self.write_event(root, fields, body)

                events, errors = self.parse_events(root)

                self.assertEqual(events, [])
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_event_rejects_duplicate_matching_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            marker = f"<!-- state-event: {fields[0]} -->"
            self.write_event(
                root,
                fields,
                "# Duplicate markers\n"
                f"{marker}\n{marker}\n\n"
                "## Context\nX\n\n## Outcome\nX\n\n## Evidence\nX\n\n"
                "## Next action\nX\n",
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("event marker" in error for error in errors), errors)

    def test_event_rejects_conflicting_second_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            self.write_event(
                root,
                fields,
                "# Conflicting markers\n"
                f"<!-- state-event: {fields[0]} -->\n"
                "<!-- state-event: STATE-20260808T143000-999 -->\n\n"
                "## Context\nX\n\n## Outcome\nX\n\n## Evidence\nX\n\n"
                "## Next action\nX\n",
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("event marker" in error for error in errors), errors)

    def test_event_evidence_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            self.write_event(root, fields)
            evidence = root / fields[9]
            evidence.write_bytes(evidence.read_bytes().replace(b"Input", b"invalid-\xff"))

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("UTF-8" in error for error in errors), errors)

    def test_event_rejects_oversized_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            fields = self.valid_fields()
            self.write_event(root, fields, "x" * (32 * 1024 + 1))

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("32 KiB" in error for error in errors), errors)

    def test_event_rows_must_be_in_increasing_id_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            later = self.valid_fields("STATE-20260808T143001-001")
            earlier = self.valid_fields("STATE-20260808T143000-001")
            index = root / self.shard.index_path
            index.write_text(
                self.header + ",".join(later) + "\n" + ",".join(earlier) + "\n",
                encoding="utf-8",
            )
            for fields in (later, earlier):
                self.write_event_file(root, fields)

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("order" in error for error in errors), errors)

    def write_event_file(self, root: Path, fields: list[str]) -> None:
        evidence = root / fields[9]
        evidence.write_text(
            "# Event\n"
            f"<!-- state-event: {fields[0]} -->\n\n"
            "## Context\nX\n\n## Outcome\nX\n\n## Evidence\nX\n\n"
            "## Next action\nX\n",
            encoding="utf-8",
        )

    def test_legacy_payload_is_extracted_byte_exactly(self) -> None:
        self.assertTrue(
            hasattr(state_record, "read_legacy_payload"),
            "legacy payload reader is missing",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.md"
            payload = b"original bytes\nwith trailing space \n"
            path.write_bytes(
                b"<!-- state-event: STATE-LEGACY-000001 -->\n"
                b"<!-- legacy-payload:begin -->\n"
                + payload
                + b"<!-- legacy-payload:end -->\n"
            )

            self.assertEqual(state_record.read_legacy_payload(path), payload)

    def test_timestamped_legacy_import_is_accepted_without_new_event_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            event_id = "STATE-20260808T143000-001"
            evidence_path = f".agents/state-events/2026-08/{event_id}.md"
            fields = [
                event_id,
                "2026-08-08T14:30:00Z",
                "legacy_import",
                "",
                "",
                "",
                "",
                "",
                "",
                evidence_path,
                "",
                "",
                "",
            ]
            (root / self.shard.index_path).write_text(
                self.header + ",".join(fields) + "\n", encoding="utf-8"
            )
            (root / evidence_path).write_bytes(
                f"<!-- state-event: {event_id} -->\n".encode("utf-8")
                + b"<!-- legacy-payload:begin -->\n"
                + b"historical payload\n"
                + b"<!-- legacy-payload:end -->\n"
            )

            events, errors = self.parse_events(root)

            self.assertEqual(errors, [])
            self.assertEqual([event.event_id for event in events], [event_id])

    def test_legacy_import_rejects_malformed_nonempty_metadata(self) -> None:
        cases = {
            "subject": (3, "POL GOOD"),
            "phase": (4, "coding"),
            "outcome": (5, "ok"),
        }
        for expected, (index, value) in cases.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = self.make_root(directory)
                event_id = "STATE-LEGACY-000001"
                evidence_path = f".agents/state-events/2026-08/{event_id}.md"
                fields = [
                    event_id,
                    "",
                    "legacy_import",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    evidence_path,
                    "",
                    "",
                    "",
                ]
                fields[index] = value
                (root / self.shard.index_path).write_text(
                    self.header + ",".join(fields) + "\n", encoding="utf-8"
                )
                (root / evidence_path).write_bytes(
                    f"<!-- state-event: {event_id} -->\n".encode("utf-8")
                    + b"<!-- legacy-payload:begin -->\n"
                    + b"historical payload\n"
                    + b"<!-- legacy-payload:end -->\n"
                )

                events, errors = self.parse_events(root)

                self.assertEqual(events, [])
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_legacy_import_rejects_calendar_invalid_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            event_id = "STATE-LEGACY-000001"
            evidence_path = f".agents/state-events/2026-08/{event_id}.md"
            fields = [
                event_id,
                "2026-02-30",
                "legacy_import",
                "",
                "",
                "",
                "",
                "",
                "",
                evidence_path,
                "",
                "",
                "",
            ]
            (root / self.shard.index_path).write_text(
                self.header + ",".join(fields) + "\n", encoding="utf-8"
            )
            (root / evidence_path).write_bytes(
                f"<!-- state-event: {event_id} -->\n".encode("utf-8")
                + b"<!-- legacy-payload:begin -->\n"
                + b"historical payload\n"
                + b"<!-- legacy-payload:end -->\n"
            )

            events, errors = self.parse_events(root)

            self.assertEqual(events, [])
            self.assertTrue(any("timestamp" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
