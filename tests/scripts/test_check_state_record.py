#!/usr/bin/env python3
"""Relational and Git-history tests for the structured state record."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import state_record


EVENT_HEADER = state_record.EVENT_HEADER
MANIFEST_HEADER = state_record.MANIFEST_HEADER
MIGRATION_HEADER = (
    "record_type",
    "source_commit",
    "source_blob",
    "source_bytes",
    "source_sha256",
    "event_id",
    "start_byte",
    "end_byte",
    "payload_bytes",
    "payload_sha256",
    "evidence_path",
)


class StateRepo:
    def __init__(self, directory: str) -> None:
        self.root = Path(directory)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "State Test"], cwd=self.root, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "state@example.invalid"],
            cwd=self.root,
            check=True,
        )
        (self.root / ".agents").mkdir()
        self.source_payload = b"# Legacy state\n\nHistorical bytes.\n"
        (self.root / ".agents/state.md").write_bytes(self.source_payload)
        self.commit("legacy source")
        self.source_revision = self.rev()
        self.install_structured_tree()

    def rev(self) -> str:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()

    def commit(self, message: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=self.root, check=True)
        return self.rev()

    def write_csv(self, path: str, header: tuple[str, ...], rows: list[list[str]]) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    def manifest_rows(self) -> list[list[str]]:
        with (self.root / ".agents/state.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))[1:]

    def event_rows(self, shard_id: str = "2026-08-001") -> list[list[str]]:
        path = self.root / f".agents/state-index/{shard_id}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))[1:]

    def write_event_rows(self, rows: list[list[str]], shard_id: str = "2026-08-001") -> None:
        self.write_csv(f".agents/state-index/{shard_id}.csv", EVENT_HEADER, rows)

    def install_structured_tree(self) -> None:
        event_id = "STATE-20260801T000000-001"
        evidence_path = f".agents/state-events/2026-08/{event_id}.md"
        self.write_csv(
            ".agents/state.csv",
            MANIFEST_HEADER,
            [[
                "1",
                "2026-08-001",
                ".agents/state-index/2026-08-001.csv",
                ".agents/state-events/2026-08/",
            ]],
        )
        self.write_event_rows(
            [[
                event_id,
                "2026-08-01T00:00:00Z",
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
            ]]
        )
        evidence = self.root / evidence_path
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_bytes(
            f"<!-- state-event: {event_id} -->\n".encode()
            + b"<!-- legacy-payload:begin -->\n"
            + self.source_payload
            + b"<!-- legacy-payload:end -->\n"
        )
        digest = hashlib.sha256(self.source_payload).hexdigest()
        source_blob = subprocess.check_output(
            ["git", "rev-parse", f"{self.source_revision}:.agents/state.md"],
            cwd=self.root,
            text=True,
        ).strip()
        self.migration_provenance = state_record.MigrationProvenance(
            commit=self.source_revision,
            blob=source_blob,
            byte_count=len(self.source_payload),
            sha256=digest,
        )
        self.write_csv(
            ".agents/completed/state-migration-manifest.csv",
            MIGRATION_HEADER,
            [
                [
                    "source",
                    self.source_revision,
                    source_blob,
                    str(len(self.source_payload)),
                    digest,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                [
                    "event",
                    "",
                    "",
                    "",
                    "",
                    event_id,
                    "0",
                    str(len(self.source_payload)),
                    str(len(self.source_payload)),
                    digest,
                    evidence_path,
                ],
            ],
        )
        (self.root / ".agents/state.md").write_text(
            "# Structured state record\n\n"
            "Current index: [.agents/state.csv](state.csv).\n\n"
            "Index shards: [.agents/state-index/](state-index/).\n\n"
            "Event evidence: [.agents/state-events/](state-events/).\n\n"
            "Migration coverage: "
            "[.agents/completed/state-migration-manifest.csv]"
            "(completed/state-migration-manifest.csv).\n\n"
            "Frozen legacy source: "
            f"https://github.com/example/project/blob/{self.source_revision}/.agents/state.md\n",
            encoding="utf-8",
        )
        (self.root / ".agents/specs").mkdir(exist_ok=True)
        (self.root / ".agents/specs/state-test.md").write_text("# Test spec\n")

    def validate(self, *, base: str | None = None) -> list[str]:
        return state_record.validate(
            self.root,
            base=base,
            migration_provenance=self.migration_provenance,
        )

    def add_event(
        self,
        *,
        event_id: str = "STATE-20260808T143000-001",
        kind: str = "checkpoint",
        subjects: str = "POL-STATE;state-record-structure-1",
        phase: str = "verification",
        outcome: str = "passed",
        spec: str = ".agents/specs/state-test.md",
        supersedes: str = "",
        shard_id: str = "2026-08-001",
    ) -> list[str]:
        period = shard_id[:7]
        evidence_path = f".agents/state-events/{period}/{event_id}.md"
        row = [
            event_id,
            "2026-08-08T14:30:00Z",
            kind,
            subjects,
            phase,
            outcome,
            "a" * 40,
            "pr:166",
            spec,
            evidence_path,
            supersedes,
            "Relational validator checkpoint.",
            "Continue migration work.",
        ]
        rows = self.event_rows(shard_id) if (self.root / f".agents/state-index/{shard_id}.csv").exists() else []
        rows.append(row)
        self.write_event_rows(rows, shard_id)
        evidence = self.root / evidence_path
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            "# Validator checkpoint\n"
            f"<!-- state-event: {event_id} -->\n\n"
            "## Context\nFixture.\n\n"
            "## Outcome\nPassed.\n\n"
            "## Evidence\nFocused test.\n\n"
            "## Next action\nContinue.\n",
            encoding="utf-8",
        )
        return row


class RelationalValidationTests(unittest.TestCase):
    def test_reconciled_imports_coexist_with_strict_post_cutover_checkpoint(self) -> None:
        """Catches treating independently valid structured events as migration output."""
        events, errors = state_record.load_events(ROOT)

        self.assertEqual(errors, [])
        by_id = {event.event_id: event for event in events}
        self.assertEqual(by_id["STATE-20260809T083000-001"].kind, "legacy_import")
        checkpoints = [
            event
            for event in events
            if event.kind == "checkpoint"
            and "state-record-structure-1" in event.subject_ids.split(";")
        ]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].pr, "pr:166")
        self.assertEqual(state_record.validate_repository(ROOT), [])

    def test_valid_structured_tree_passes(self) -> None:
        self.assertTrue(hasattr(state_record, "validate"), "validate() is missing")
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            repo.add_event()

            self.assertEqual(repo.validate(), [])

    def test_subjects_must_be_sorted_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            repo.add_event(subjects="z;a;a")

            errors = repo.validate()

            self.assertTrue(any("subject IDs" in error and "sorted" in error for error in errors), errors)

    def test_event_ids_are_globally_unique_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            event_id = "STATE-20260808T143000-001"
            repo.add_event(event_id=event_id)
            rows = repo.manifest_rows()
            rows.append([
                "1",
                "2026-09-001",
                ".agents/state-index/2026-09-001.csv",
                ".agents/state-events/2026-09/",
            ])
            repo.write_csv(".agents/state.csv", MANIFEST_HEADER, rows)
            duplicate = repo.add_event(event_id=event_id, shard_id="2026-09-001")
            duplicate[1] = "2026-08-08T14:30:00Z"
            repo.write_event_rows([duplicate], "2026-09-001")

            errors = repo.validate()

            self.assertTrue(any("globally unique" in error for error in errors), errors)

    def test_spec_must_be_allowed_and_resolve(self) -> None:
        cases = ("docs/other.md", ".agents/specs/missing.md")
        for spec in cases:
            with self.subTest(spec=spec), tempfile.TemporaryDirectory() as directory:
                repo = StateRepo(directory)
                repo.add_event(spec=spec)

                errors = repo.validate()

                self.assertTrue(any("spec" in error for error in errors), errors)

    def test_spec_symlink_cannot_escape_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            (repo.root / ".agents/specs/escape.md").symlink_to("/etc/passwd")
            repo.add_event(spec=".agents/specs/escape.md")

            errors = repo.validate()

            self.assertTrue(any("outside the repository" in error for error in errors), errors)

    def test_orphan_evidence_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            orphan = repo.root / ".agents/state-events/2026-08/STATE-ORPHAN.md"
            orphan.write_text("orphan\n", encoding="utf-8")

            errors = repo.validate()

            self.assertTrue(any("orphan evidence" in error for error in errors), errors)

    def test_supersession_is_correction_only_and_backward(self) -> None:
        cases = (
            ("checkpoint", "STATE-20260801T000000-001", "forbidden"),
            ("correction", "", "requires"),
            ("correction", "STATE-20260809T000000-001", "earlier"),
        )
        for kind, supersedes, expected in cases:
            with self.subTest(kind=kind, supersedes=supersedes), tempfile.TemporaryDirectory() as directory:
                repo = StateRepo(directory)
                repo.add_event(kind=kind, supersedes=supersedes)

                errors = repo.validate()

                self.assertTrue(any(expected in error for error in errors), errors)

    def test_valid_correction_supersedes_an_earlier_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            repo.add_event(
                kind="correction", supersedes="STATE-20260801T000000-001"
            )

            self.assertEqual(repo.validate(), [])

    def test_migration_manifest_and_stub_are_validated(self) -> None:
        mutations = (
            "stub",
            "hash",
            "range",
            "path",
            "provenance_missing",
            "provenance_commit",
            "provenance_blob",
            "provenance_size",
            "provenance_hash",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                repo = StateRepo(directory)
                self.assertEqual(repo.validate(), [])
                if mutation == "stub":
                    (repo.root / ".agents/state.md").write_text("# old state\n")
                else:
                    path = repo.root / ".agents/completed/state-migration-manifest.csv"
                    with path.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.reader(handle))
                    if mutation == "provenance_missing":
                        del rows[1]
                    elif mutation.startswith("provenance_"):
                        column = {
                            "provenance_commit": 1,
                            "provenance_blob": 2,
                            "provenance_size": 3,
                            "provenance_hash": 4,
                        }[mutation]
                        rows[1][column] = "0" * (64 if column == 4 else 40)
                    else:
                        column = {"range": 7, "hash": 9, "path": 10}[mutation]
                        rows[2][column] = {
                            "range": "999",
                            "hash": "0" * 64,
                            "path": ".agents/wrong.md",
                        }[mutation]
                    repo.write_csv(str(path.relative_to(repo.root)), MIGRATION_HEADER, rows[1:])

                errors = repo.validate()

                self.assertTrue(any("migration" in error or "compatibility stub" in error for error in errors), errors)

    def test_frozen_and_final_provenance_authority_mutations_are_rejected(self) -> None:
        """Catches coupling or replacing either independently pinned source role."""
        mutations = (
            (
                "frozen",
                "FROZEN_MIGRATION_PROVENANCE",
                replace(
                    state_record.FROZEN_MIGRATION_PROVENANCE,
                    commit=state_record.FINAL_MIGRATION_PROVENANCE.commit,
                ),
            ),
            (
                "final",
                "FINAL_MIGRATION_PROVENANCE",
                replace(state_record.FINAL_MIGRATION_PROVENANCE, sha256="0" * 64),
            ),
        )
        for label, attribute, value in mutations:
            with self.subTest(role=label), mock.patch.object(
                state_record, attribute, value
            ):
                errors = state_record.validate(ROOT)

            self.assertTrue(any("migration" in error for error in errors), errors)

    def test_compatibility_stub_requires_state_index_tree_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            stub = repo.root / ".agents/state.md"
            stub.write_text(
                stub.read_text(encoding="utf-8").replace(
                    "Index shards: [.agents/state-index/](state-index/).\n\n", ""
                ),
                encoding="utf-8",
            )

            errors = repo.validate()

            self.assertTrue(
                any(".agents/state-index/" in error for error in errors), errors
            )

    def test_compatibility_stub_requires_state_events_tree_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            stub = repo.root / ".agents/state.md"
            stub.write_text(
                stub.read_text(encoding="utf-8").replace(
                    "Event evidence: [.agents/state-events/](state-events/).\n\n", ""
                ),
                encoding="utf-8",
            )

            errors = repo.validate()

            self.assertTrue(
                any(".agents/state-events/" in error for error in errors), errors
            )


class GitImmutabilityTests(unittest.TestCase):
    def structured_base(self, repo: StateRepo) -> str:
        return repo.commit("structured base")

    def test_existing_evidence_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            base = self.structured_base(repo)
            path = repo.root / ".agents/state-events/2026-08/STATE-20260801T000000-001.md"
            path.write_bytes(path.read_bytes().replace(b"Historical", b"historical"))

            errors = repo.validate(base=base)

            self.assertTrue(any("immutable evidence" in error for error in errors), errors)

    def test_writable_shard_is_byte_preserving_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            base = self.structured_base(repo)
            rows = repo.event_rows()
            rows[0][11] = "rewritten legacy metadata"
            repo.write_event_rows(rows)

            errors = repo.validate(base=base)

            self.assertTrue(any("append-only" in error for error in errors), errors)

    def test_pre_cutover_base_uses_first_structured_commit_as_history_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            pre_cutover = repo.source_revision
            repo.commit("structured cutover")
            rows = repo.event_rows()
            rows[0][11] = "rewritten legacy metadata"
            repo.write_event_rows(rows)

            errors = repo.validate(base=pre_cutover)

            self.assertTrue(any("append-only" in error for error in errors), errors)

    def test_root_manifest_is_byte_preserving_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            base = self.structured_base(repo)
            rows = repo.manifest_rows()
            rows.insert(0, [
                "1",
                "2026-07-001",
                ".agents/state-index/2026-07-001.csv",
                ".agents/state-events/2026-07/",
            ])
            repo.write_csv(".agents/state.csv", MANIFEST_HEADER, rows)
            repo.write_event_rows([], "2026-07-001")

            errors = repo.validate(base=base)

            self.assertTrue(any("state manifest" in error and "append-only" in error for error in errors), errors)

    def test_appending_to_writable_shard_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            base = self.structured_base(repo)
            repo.add_event()

            self.assertEqual(repo.validate(base=base), [])

    def test_new_shard_seals_previous_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = StateRepo(directory)
            base = self.structured_base(repo)
            manifest = repo.manifest_rows()
            manifest.append([
                "1",
                "2026-09-001",
                ".agents/state-index/2026-09-001.csv",
                ".agents/state-events/2026-09/",
            ])
            repo.write_csv(".agents/state.csv", MANIFEST_HEADER, manifest)
            repo.write_event_rows([], "2026-09-001")
            old_rows = repo.event_rows()
            old_rows[0][11] = "mutated while sealing"
            repo.write_event_rows(old_rows)

            errors = repo.validate(base=base)

            self.assertTrue(any("sealed shard" in error for error in errors), errors)


class CliTests(unittest.TestCase):
    def test_cli_reports_valid_tree(self) -> None:
        checker = ROOT / "scripts/check-state-record.py"
        self.assertTrue(checker.exists(), "check-state-record.py is missing")
        result = subprocess.run(
            [sys.executable, str(checker), "--root", str(ROOT)],
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("structured state record", result.stdout)


if __name__ == "__main__":
    unittest.main()
