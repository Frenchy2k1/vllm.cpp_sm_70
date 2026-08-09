#!/usr/bin/env python3
"""Focused behavior tests for deterministic state-record migration."""

from __future__ import annotations

import csv
import contextlib
import hashlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MIGRATOR = ROOT / "scripts/migrate-state-record.py"
sys.path.insert(0, str(ROOT / "scripts"))

import state_record


MIGRATOR_SPEC = importlib.util.spec_from_file_location(
    "migrate_state_record", MIGRATOR
)
assert MIGRATOR_SPEC is not None and MIGRATOR_SPEC.loader is not None
migrate_state_record = importlib.util.module_from_spec(MIGRATOR_SPEC)
sys.modules[MIGRATOR_SPEC.name] = migrate_state_record
MIGRATOR_SPEC.loader.exec_module(migrate_state_record)


SOURCE = (
    "# State log\r\n\r\nPrelude café is deliberately unanchored.\r\n".encode()
    + b"## First event\r\n"
    + b"<!-- state: 2026-08-05T09:00 -->\r\n"
    + b"Alpha payload.\r\n"
    + b"## Duplicate timestamp\n"
    + b"<!-- state: 2026-08-05T09:00 -->\n"
    + b"Beta payload.\n\n## An unanchored subheading\nStill beta.\n"
    + b"## Date-only event\n"
    + b"<!-- state: 2026-08-06 -->\n"
    + b"Gamma payload without a final newline."
)


class MigrationRepo:
    def __init__(self, directory: str, source: bytes = SOURCE) -> None:
        self.root = Path(directory)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Migration Test"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "migration@example.invalid"],
            cwd=self.root,
            check=True,
        )
        (self.root / ".agents").mkdir()
        (self.root / ".agents/state.md").write_bytes(source)
        subprocess.run(["git", "add", ".agents/state.md"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "legacy source"], cwd=self.root, check=True
        )
        self.source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()
        source_blob = subprocess.check_output(
            ["git", "rev-parse", f"{self.source_revision}:.agents/state.md"],
            cwd=self.root,
            text=True,
        ).strip()
        self.provenance = state_record.MigrationProvenance(
            commit=self.source_revision,
            blob=source_blob,
            byte_count=len(source),
            sha256=hashlib.sha256(source).hexdigest(),
        )

    def run(self, mode: str, revision: str | None = None) -> subprocess.CompletedProcess[str]:
        arguments = [
            "--source-revision",
            revision or self.source_revision,
            "--output-root",
            str(self.root),
            mode,
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = migrate_state_record.main(arguments, self.provenance)
        return subprocess.CompletedProcess(
            arguments, returncode, stdout.getvalue(), stderr.getvalue()
        )

    def manifest_rows(self) -> list[list[str]]:
        path = self.root / ".agents/completed/state-migration-manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def event_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for path in sorted((self.root / ".agents/state-index").glob("*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(list(csv.reader(handle))[1:])
        return rows


class MigrationApplyTests(unittest.TestCase):
    def test_apply_and_verify_prelude_with_multi_month_history(self) -> None:
        """Catches legacy IDs sorting after timestamp IDs across dated shards."""
        source = (
            b"Unanchored prelude.\n"
            b"## July event\n"
            b"<!-- state: 2026-07-31T23:59:00Z -->\n"
            b"July payload.\n"
            b"## August event\n"
            b"<!-- state: 2026-08-01T00:01:00Z -->\n"
            b"August payload.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory, source)

            applied = repo.run("--apply")
            verified = repo.run("--verify")

            self.assertEqual(applied.returncode, 0, applied.stderr)
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_apply_partitions_indexes_and_evidence_by_event_month(self) -> None:
        """Catches collapsing a multi-month history into the final month."""
        source = (
            b"## July event\n"
            b"<!-- state: 2026-07-31T23:59:00Z -->\n"
            b"July payload.\n"
            b"## August event\n"
            b"<!-- state: 2026-08-01T00:01:00Z -->\n"
            b"August payload.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory, source)

            result = repo.run("--apply")

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_path = repo.root / ".agents/state.csv"
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.reader(handle))
            self.assertEqual(
                manifest[1:],
                [
                    [
                        "1",
                        "2026-07-001",
                        ".agents/state-index/2026-07-001.csv",
                        ".agents/state-events/2026-07/",
                    ],
                    [
                        "1",
                        "2026-08-001",
                        ".agents/state-index/2026-08-001.csv",
                        ".agents/state-events/2026-08/",
                    ],
                ],
            )
            rows = repo.event_rows()
            self.assertEqual(
                [row[9] for row in rows],
                [
                    ".agents/state-events/2026-07/STATE-20260731T235900-001.md",
                    ".agents/state-events/2026-08/STATE-20260801T000100-001.md",
                ],
            )

    def test_apply_segments_bytes_wraps_payloads_and_does_not_infer_lifecycle(self) -> None:
        """Catches normalized bytes, heuristic prelude splitting, and fabricated metadata."""
        self.assertTrue(MIGRATOR.exists(), "migrate-state-record.py is missing")
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)

            result = repo.run("--apply")

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = repo.manifest_rows()
            self.assertEqual(
                manifest[0],
                [
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
                ],
            )
            self.assertEqual(len(manifest), 6)
            source_blob = subprocess.check_output(
                ["git", "rev-parse", f"{repo.source_revision}:.agents/state.md"],
                cwd=repo.root,
                text=True,
            ).strip()
            self.assertEqual(
                manifest[1],
                [
                    "source",
                    repo.source_revision,
                    source_blob,
                    str(len(SOURCE)),
                    hashlib.sha256(SOURCE).hexdigest(),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            )
            first_heading = SOURCE.index(b"## First event")
            self.assertEqual(manifest[2][6:8], ["0", str(first_heading)])
            self.assertEqual(manifest[2][5], "STATE-LEGACY-000001")
            self.assertEqual(
                [row[5] for row in manifest[3:5]],
                ["STATE-20260805T090000-001", "STATE-20260805T090000-002"],
            )
            self.assertEqual(manifest[5][5], "STATE-LEGACY-000002")

            reconstructed = bytearray()
            expected_start = 0
            for row in manifest[2:]:
                self.assertEqual(row[:5], ["event", "", "", "", ""])
                event_id, start_s, end_s, count_s, digest, evidence_path = row[5:]
                start, end, count = int(start_s), int(end_s), int(count_s)
                self.assertEqual(start, expected_start)
                self.assertEqual(end - start, count)
                payload = state_record.read_legacy_payload(repo.root / evidence_path)
                self.assertEqual(payload, SOURCE[start:end])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                reconstructed.extend(payload)
                expected_start = end
                evidence = (repo.root / evidence_path).read_bytes()
                self.assertIn(f"<!-- state-event: {event_id} -->".encode(), evidence)
                self.assertIn(b"<!-- legacy-source:", evidence)
            self.assertEqual(expected_start, len(SOURCE))
            self.assertEqual(bytes(reconstructed), SOURCE)

            by_id = {row[0]: row for row in repo.event_rows()}
            self.assertEqual(by_id["STATE-20260805T090000-001"][1], "2026-08-05T09:00:00Z")
            self.assertEqual(by_id["STATE-LEGACY-000002"][1], "2026-08-06")
            for row in by_id.values():
                self.assertEqual(row[2], "legacy_import")
                self.assertEqual(row[3:9], ["", "", "", "", "", ""])
                self.assertEqual(row[10:], ["", "", ""])

    def test_reapplying_identical_generation_is_reproducible(self) -> None:
        """Catches environment-dependent output and refusal of byte-identical files."""
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            first = repo.run("--apply")
            self.assertEqual(first.returncode, 0, first.stderr)
            before = {
                path.relative_to(repo.root).as_posix(): path.read_bytes()
                for path in repo.root.glob(".agents/**/*")
                if path.is_file()
            }

            second = repo.run("--apply")

            self.assertEqual(second.returncode, 0, second.stderr)
            after = {
                path.relative_to(repo.root).as_posix(): path.read_bytes()
                for path in repo.root.glob(".agents/**/*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_apply_refuses_to_overwrite_different_generated_bytes(self) -> None:
        """Catches destructive regeneration over reviewed migration evidence."""
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            event_path = repo.root / repo.manifest_rows()[2][10]
            mutated = event_path.read_bytes().replace(b"Prelude", b"Tampered", 1)
            event_path.write_bytes(mutated)

            result = repo.run("--apply")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refus", result.stderr.lower())
            self.assertIn(event_path.relative_to(repo.root).as_posix(), result.stderr)
            self.assertEqual(event_path.read_bytes(), mutated)


class MigrationVerifyTests(unittest.TestCase):
    def final_scratch(self, directory: str) -> Path:
        scratch = Path(directory)
        shutil.copytree(ROOT / ".agents", scratch / ".agents")
        spec = Path(
            "docs/superpowers/specs/2026-08-09-state-migration-epochs-design.md"
        )
        (scratch / spec).parent.mkdir(parents=True)
        shutil.copy2(ROOT / spec, scratch / spec)
        subprocess.run(["git", "init", "-q"], cwd=scratch, check=True)
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True
        ).strip()
        common_objects = (ROOT / common_dir / "objects").resolve()
        alternates = scratch / ".git/objects/info/alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(f"{common_objects}\n", encoding="utf-8")
        return scratch

    def test_verify_allows_new_strict_shard_but_freezes_legacy_subset(self) -> None:
        """Catches treating the migration-only manifest as the whole live record."""
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            shutil.copytree(ROOT / ".agents", scratch / ".agents")
            spec = Path(
                "docs/superpowers/specs/2026-08-09-state-migration-epochs-design.md"
            )
            (scratch / spec).parent.mkdir(parents=True)
            shutil.copy2(ROOT / spec, scratch / spec)
            subprocess.run(["git", "init", "-q"], cwd=scratch, check=True)
            common_dir = subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True
            ).strip()
            common_objects = (ROOT / common_dir / "objects").resolve()
            alternates = scratch / ".git/objects/info/alternates"
            alternates.parent.mkdir(parents=True, exist_ok=True)
            alternates.write_text(f"{common_objects}\n", encoding="utf-8")

            event_id = "STATE-20260809T160000-001"
            evidence_path = f".agents/state-events/2026-08/{event_id}.md"
            manifest_path = scratch / ".agents/state.csv"
            with manifest_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    [
                        "1",
                        "2026-08-002",
                        ".agents/state-index/2026-08-002.csv",
                        ".agents/state-events/2026-08/",
                    ]
                )
            shard_path = scratch / ".agents/state-index/2026-08-002.csv"
            with shard_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerows(
                    [
                        state_record.EVENT_HEADER,
                        [
                            event_id,
                            "2026-08-09T16:00:00Z",
                            "checkpoint",
                            "state-record-structure-1",
                            "verification",
                            "passed",
                            "9e50a2f48b0c6162c0da155a2033429826f53f3e",
                            "pr:166",
                            spec.as_posix(),
                            evidence_path,
                            "",
                            "Independent strict checkpoint in a new shard.",
                            "Continue with the next structured-state gate.",
                        ],
                    ]
                )
            (scratch / evidence_path).write_text(
                "# Independent structured checkpoint\n"
                f"<!-- state-event: {event_id} -->\n\n"
                "## Context\nPost-cutover structured state remains independently appendable.\n\n"
                "## Outcome\nThe strict event occupies a new index shard.\n\n"
                "## Evidence\nThe structured checker validates the complete scratch tree.\n\n"
                "## Next action\nContinue the state-record gate.\n",
                encoding="utf-8",
            )

            migration = migrate_state_record.build_verification_migration(
                scratch, state_record.FINAL_MIGRATION_PROVENANCE.commit
            )
            self.assertEqual(state_record.validate_repository(scratch), [])
            migrate_state_record.verify_migration(
                scratch, migration, state_record.FROZEN_MIGRATION_PROVENANCE
            )

            legacy_path = scratch / ".agents/state-index/0000-00-001.csv"
            legacy_bytes = legacy_path.read_bytes()
            quoted_legacy_bytes = legacy_bytes.replace(
                b"STATE-LEGACY-000001,",
                b'"STATE-LEGACY-000001",',
                1,
            )
            manifest_bytes = manifest_path.read_bytes()
            quoted_manifest_bytes = manifest_bytes.replace(
                b"1,0000-00-001,",
                b'"1",0000-00-001,',
                1,
            )
            for path, original, differently_quoted in (
                (legacy_path, legacy_bytes, quoted_legacy_bytes),
                (manifest_path, manifest_bytes, quoted_manifest_bytes),
            ):
                with self.subTest(raw_row_path=path.relative_to(scratch)):
                    self.assertNotEqual(differently_quoted, original)
                    self.assertEqual(
                        list(csv.reader(differently_quoted.decode("utf-8").splitlines())),
                        list(csv.reader(original.decode("utf-8").splitlines())),
                    )
                    path.write_bytes(differently_quoted)
                    try:
                        validation_errors = state_record.validate_repository(scratch)
                        if path == legacy_path:
                            self.assertTrue(
                                any(
                                    "index row inventory" in error
                                    for error in validation_errors
                                ),
                                validation_errors,
                            )
                        else:
                            self.assertEqual(validation_errors, [])
                        with self.assertRaises(migrate_state_record.MigrationError):
                            migrate_state_record.verify_migration(
                                scratch,
                                migration,
                                state_record.FROZEN_MIGRATION_PROVENANCE,
                            )
                    finally:
                        path.write_bytes(original)

            frozen_rows = list(
                csv.reader(legacy_path.read_text(encoding="utf-8").splitlines())
            )
            for mutation in ("mutated", "removed"):
                with self.subTest(mutation=mutation):
                    rows = [row[:] for row in frozen_rows]
                    if mutation == "mutated":
                        rows[1][1] = "2026-08-01"
                    else:
                        del rows[1]
                    with legacy_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)
                    with self.assertRaises(migrate_state_record.MigrationError):
                        migrate_state_record.verify_migration(
                            scratch,
                            migration,
                            state_record.FROZEN_MIGRATION_PROVENANCE,
                        )
            with legacy_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerows(frozen_rows)

    def test_reconciliation_preserves_frozen_events_and_index_rows(self) -> None:
        """Catches regeneration of any reviewed frozen event or index-row byte."""
        frozen = migrate_state_record.build_migration(
            ROOT, state_record.MIGRATION_EPOCHS[0].commit
        )
        reconciled = migrate_state_record.build_verification_migration(
            ROOT, "f921062ba4fbf3341b02d2ac826bb3d84cc91a64"
        )
        frozen_events = {
            path: payload
            for path, payload in frozen.artifacts.items()
            if path.startswith(".agents/state-events/")
        }

        self.assertEqual(len(frozen_events), 146)
        self.assertEqual(
            {path: hashlib.sha256(payload).hexdigest() for path, payload in frozen_events.items()},
            {
                path: hashlib.sha256(reconciled.artifacts[path]).hexdigest()
                for path in frozen_events
            },
        )
        for path, frozen_bytes in frozen.artifacts.items():
            if path.startswith(".agents/state-index/"):
                self.assertTrue(reconciled.artifacts[path].startswith(frozen_bytes), path)

    def test_reviewed_append_epoch_builder_remains_available_for_archive(self) -> None:
        """Catches dropping the reviewed append-epoch artifact after reconciliation."""
        migration = migrate_state_record.build_verification_migration(
            ROOT, "f921062ba4fbf3341b02d2ac826bb3d84cc91a64"
        )

        self.assertEqual(
            migration.artifacts[migrate_state_record.MANIFEST_PATH],
            (ROOT / state_record.ARCHIVED_MIGRATION_MANIFEST).read_bytes(),
        )

    def test_reconciliation_accepts_the_pinned_concurrent_merge_snapshot(self) -> None:
        """Catches treating the reviewed final merge snapshot as an append epoch."""
        migration = migrate_state_record.build_verification_migration(
            ROOT, "dc2139b3ff95f6de9b6a8ec8cae4bd5d40262dc7"
        )

        self.assertEqual(
            migration.source_revision,
            "dc2139b3ff95f6de9b6a8ec8cae4bd5d40262dc7",
        )
        self.assertEqual(migration.source_blob, "e79acd955e7042f9d028e18d41569ff5b67e32c3")
        self.assertEqual(len(migration.source), 3_225_646)
        self.assertEqual(
            hashlib.sha256(migration.source).hexdigest(),
            "5c389a0cd84e6834263630af9e9d5a7a64f131ef06d479670dc6e3dc942ec202",
        )
        self.assertEqual(
            state_record.FINAL_BOUNDARY_OVERRIDES,
            {
                "STATE-20260809T001000-001": (3196620, 3197705),
                "STATE-20260809T083000-001": (3197705, 3200170),
                "STATE-20260809T110000-001": (3200170, 3202350),
                "STATE-20260809T130000-001": (3202350, 3205680),
            },
        )
        self.assertEqual(
            state_record.FINAL_NEW_EVENT_IDS,
            (
                "STATE-20260808T210000-003",
                "STATE-20260808T220000-002",
                "STATE-20260808T230000-002",
                "STATE-20260808T233000-002",
                "STATE-20260808T234500-002",
                "STATE-20260809T140000-001",
                "STATE-20260809T140000-002",
            ),
        )
        self.assertEqual(
            state_record.FINAL_NEW_INDEX_ROW_SHA256,
            {
                "STATE-20260808T210000-003":
                    "f6ef197bae96e5ff8d1b9e9f47c1cb3729c6f8eed3da0d28a1ac5f81b2acef3d",
                "STATE-20260808T220000-002":
                    "02e320a239a82ab00eeabbe3dc1d11d95e0467cfb93158460a1a87c02cd77b9b",
                "STATE-20260808T230000-002":
                    "6bf899ac1a80b9027ed46a9498ef738158f1bb22776673bc3258900d340a0ce9",
                "STATE-20260808T233000-002":
                    "60e2529de3820e8651979e12c2195c228be2b7ca5d75ca3071615a9114560b95",
                "STATE-20260808T234500-002":
                    "72e45c389dd8aa692eebc75949ac5f02a509b19ada919fa266760f6faa1958f8",
                "STATE-20260809T140000-001":
                    "4bbbe4a78b4b196a7553a07d9130ed059916c053892aaf81e72b8cb98def8b7a",
                "STATE-20260809T140000-002":
                    "05a03ffbc09399563a24d833d48c0d1b9df90b1c3ccaaa12b76bd87b07cd41dc",
            },
        )
        segment_ranges = {
            segment.event_id: (segment.start, segment.end)
            for segment in migration.segments
        }
        for event_id, expected_range in state_record.FINAL_BOUNDARY_OVERRIDES.items():
            self.assertEqual(segment_ranges[event_id], expected_range)
        reconstructed = b"".join(
            state_record.read_legacy_payload(ROOT / segment.evidence_path)
            for segment in migration.segments
        )
        self.assertEqual(len(reconstructed), 3_225_646)
        self.assertEqual(reconstructed, migration.source)

        artifact_hashes = {
            state_record.ARCHIVED_MIGRATION_MANIFEST:
                "2f8ea0da82038cb1b297b53b2b28f93a02d2963170b13572a6f676299b439113",
            migrate_state_record.MANIFEST_PATH:
                "22b963c5147bb4b673330a266d0c3e5a04f5e555aca5c971ced9de9bd71db8aa",
            ".agents/state-index/2026-08-001.csv":
                "b7607aa343dc0feaa4beb45a4e61299ec8e7afe9401b1d350254a9d3d42e0445",
            ".agents/state.csv":
                "354e55c87e7724cf777f3ca20e51de80f78c859698458df3805a54d4e8e2e86a",
            ".agents/state.md":
                "6d875619350e4ebbd6da8fc1efb4c219bfc7944ae1c69fe41815691a5d6e373c",
            ".agents/state-events/2026-08/STATE-20260809T130000-001.md":
                "d86f92366e9fb3b91ebcc520f30b70788af0f8070d604fa9a5cbdd663e4ae007",
        }
        for path, expected_hash in artifact_hashes.items():
            self.assertEqual(
                hashlib.sha256(migration.artifacts[path]).hexdigest(),
                expected_hash,
                path,
            )
        preserved_wrapper_hashes = {
            "STATE-20260809T001000-001":
                "629faa117ce380284ee5272a04a874795cb00632422392997bffc978e5512692",
            "STATE-20260809T083000-001":
                "3a0338cad087a6abe80442677d6e19a0fe910128d87ed1c37563431d0acd3f32",
            "STATE-20260809T110000-001":
                "098a616110e62bb37771e39fd28d7237aac2b310d9d8681030dbeec3c8da54c6",
        }
        for event_id, expected_hash in preserved_wrapper_hashes.items():
            path = f".agents/state-events/2026-08/{event_id}.md"
            self.assertEqual(
                hashlib.sha256(migration.artifacts[path]).hexdigest(), expected_hash
            )

    def test_post_final_append_preserves_the_reviewed_snapshot(self) -> None:
        """Catches rebuilding the 156 reviewed imports for a strict source append."""
        migration = migrate_state_record.build_verification_migration(
            ROOT, "776c56f1c8b78ab69ea01e14759187b243b24d9e"
        )

        self.assertEqual(
            state_record.PREVIOUS_FINAL_MIGRATION_PROVENANCE.commit,
            "dc2139b3ff95f6de9b6a8ec8cae4bd5d40262dc7",
        )
        self.assertEqual(
            state_record.FINAL_MIGRATION_PROVENANCE,
            state_record.MigrationProvenance(
                commit="776c56f1c8b78ab69ea01e14759187b243b24d9e",
                blob="e266a8892401bc744955ccc1cb3bc75f64e4f399",
                byte_count=3_231_342,
                sha256="913c76a2ab8303b1b5ad7dae3ec7876c5e29f0c319670b86eb646c4baa3119b6",
            ),
        )
        self.assertEqual(len(migration.segments), 157)
        self.assertEqual(
            (migration.segments[-2].event_id, migration.segments[-2].end),
            ("STATE-20260809T140000-002", 3_225_646),
        )
        self.assertEqual(
            (
                migration.segments[-1].event_id,
                migration.segments[-1].start,
                migration.segments[-1].end,
            ),
            ("STATE-20260809T150000-001", 3_225_646, 3_231_342),
        )
        previous_source = subprocess.check_output(
            [
                "git",
                "show",
                "dc2139b3ff95f6de9b6a8ec8cae4bd5d40262dc7:.agents/state.md",
            ],
            cwd=ROOT,
        )
        self.assertTrue(migration.source.startswith(previous_source))
        appended_wrapper = migration.artifacts[migration.segments[-1].evidence_path]
        appended = appended_wrapper.split(state_record.LEGACY_BEGIN, 1)[1].split(
            state_record.LEGACY_END, 1
        )[0]
        self.assertTrue(appended.startswith(b"\n"))
        self.assertEqual(appended, migration.source[3_225_646:3_231_342])

        previous_wrappers = {
            segment.evidence_path: (ROOT / segment.evidence_path).read_bytes()
            for segment in migration.segments[:-1]
        }
        self.assertEqual(len(previous_wrappers), 156)
        for path, expected in previous_wrappers.items():
            self.assertEqual(migration.artifacts[path], expected, path)

        reconstructed = b"".join(
            migration.artifacts[segment.evidence_path]
            .split(state_record.LEGACY_BEGIN, 1)[1]
            .split(state_record.LEGACY_END, 1)[0]
            for segment in migration.segments
        )
        self.assertEqual(reconstructed, migration.source)

    def test_verify_reports_final_snapshot_and_frozen_provenance_roles(self) -> None:
        """Catches presenting the live snapshot as the frozen historical source."""
        final_revision = "776c56f1c8b78ab69ea01e14759187b243b24d9e"
        frozen_revision = "994cd8d4122ecf44f72d51fabd61c45adaaea9d3"
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = migrate_state_record.main(
                [
                    "--source-revision",
                    final_revision,
                    "--output-root",
                    str(ROOT),
                    "--verify",
                ]
            )

        self.assertEqual(returncode, 0, stderr.getvalue())
        self.assertIn(
            f"final/current snapshot authority {final_revision}", stdout.getvalue()
        )
        self.assertIn(f"frozen provenance {frozen_revision}", stdout.getvalue())
        self.assertNotIn(f"frozen provenance {final_revision}", stdout.getvalue())

    def test_post_final_append_mutations_are_rejected(self) -> None:
        """Catches raw-row, boundary, wrapper, and coupled-authority drift."""
        event_id = state_record.FINAL_APPEND_EVENT_ID
        index_relative = Path(".agents/state-index/2026-08-001.csv")
        wrapper_relative = Path(
            f".agents/state-events/2026-08/{event_id}.md"
        )
        manifest_relative = Path(
            ".agents/completed/state-migration-manifest.csv"
        )

        for mutation in ("quoted_id", "timestamp", "boundary", "wrapper"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                scratch = self.final_scratch(directory)
                if mutation in {"quoted_id", "timestamp"}:
                    path = scratch / index_relative
                    original = path.read_bytes()
                    if mutation == "quoted_id":
                        mutated = original.replace(
                            f"{event_id},".encode("ascii"),
                            f'"{event_id}",'.encode("ascii"),
                            1,
                        )
                        self.assertEqual(
                            list(csv.reader(mutated.decode("utf-8").splitlines())),
                            list(csv.reader(original.decode("utf-8").splitlines())),
                        )
                    else:
                        mutated = original.replace(
                            b"2026-08-09T15:00:00Z,legacy_import",
                            b"2026-08-09T15:00:01Z,legacy_import",
                            1,
                        )
                    self.assertNotEqual(mutated, original)
                    path.write_bytes(mutated)
                elif mutation == "boundary":
                    path = scratch / manifest_relative
                    original = path.read_bytes()
                    mutated = original.replace(
                        f"{event_id},3225646,3231342,".encode("ascii"),
                        f"{event_id},3225647,3231342,".encode("ascii"),
                        1,
                    )
                    self.assertNotEqual(mutated, original)
                    path.write_bytes(mutated)
                else:
                    path = scratch / wrapper_relative
                    original = path.read_bytes()
                    mutated = original.replace(b"# Imported", b"! Imported", 1)
                    self.assertNotEqual(mutated, original)
                    path.write_bytes(mutated)

                errors = state_record.validate_repository(scratch)
                self.assertTrue(errors, mutation)
                try:
                    migration = migrate_state_record.build_verification_migration(
                        scratch, state_record.FINAL_MIGRATION_PROVENANCE.commit
                    )
                except migrate_state_record.MigrationError:
                    continue
                with self.assertRaises(migrate_state_record.MigrationError):
                    migrate_state_record.verify_migration(
                        scratch, migration, state_record.FROZEN_MIGRATION_PROVENANCE
                    )

        with tempfile.TemporaryDirectory() as directory:
            scratch = self.final_scratch(directory)
            path = scratch / index_relative
            original = path.read_bytes()
            mutated = original.replace(
                f"{event_id},".encode("ascii"),
                f'"{event_id}",'.encode("ascii"),
                1,
            )
            self.assertNotEqual(mutated, original)
            path.write_bytes(mutated)
            raw_row = next(
                raw
                for row, raw in migrate_state_record._raw_csv_rows(mutated)
                if row and row[0] == event_id
            )
            coupled_digest = hashlib.sha256(raw_row).hexdigest()
            with mock.patch.object(
                state_record,
                "FINAL_APPEND_INDEX_ROW_SHA256",
                coupled_digest,
            ):
                errors = state_record.validate_repository(scratch)
                self.assertTrue(
                    any("index row inventory" in error for error in errors),
                    errors,
                )

    def test_final_reconciliation_rejects_new_live_row_byte_mutations_before_build(self) -> None:
        """Catches deriving final raw-row authority from the mutable live index."""
        index_relative = Path(".agents/state-index/2026-08-001.csv")
        for event_id in state_record.FINAL_NEW_EVENT_IDS:
            with self.subTest(event_id=event_id), tempfile.TemporaryDirectory() as directory:
                scratch = self.final_scratch(directory)
                index_path = scratch / index_relative
                original = index_path.read_bytes()
                needle = f"{event_id},".encode("ascii")
                mutated = original.replace(
                    needle,
                    f'"{event_id}",'.encode("ascii"),
                    1,
                )
                self.assertNotEqual(mutated, original)
                self.assertEqual(
                    list(csv.reader(mutated.decode("utf-8").splitlines())),
                    list(csv.reader(original.decode("utf-8").splitlines())),
                )
                index_path.write_bytes(mutated)

                validation_errors = state_record.validate_repository(scratch)
                self.assertTrue(
                    any("final index row bytes" in error for error in validation_errors),
                    validation_errors,
                )
                with self.assertRaisesRegex(
                    migrate_state_record.MigrationError,
                    "final index row bytes",
                ):
                    migration = migrate_state_record.build_verification_migration(
                        scratch,
                        state_record.FINAL_MIGRATION_PROVENANCE.commit,
                    )
                    migrate_state_record.verify_migration(
                        scratch,
                        migration,
                        state_record.FROZEN_MIGRATION_PROVENANCE,
                    )

        with tempfile.TemporaryDirectory() as directory:
            scratch = self.final_scratch(directory)
            index_path = scratch / index_relative
            original = index_path.read_bytes()
            mutated = original.replace(
                b"STATE-20260808T210000-003,2026-08-08T21:00:00Z,",
                b"STATE-20260808T210000-003,2026-08-08T21:00:01Z,",
                1,
            )
            self.assertNotEqual(mutated, original)
            index_path.write_bytes(mutated)

            with self.assertRaisesRegex(
                migrate_state_record.MigrationError,
                "final index row bytes",
            ):
                migrate_state_record.build_verification_migration(
                    scratch,
                    state_record.FINAL_MIGRATION_PROVENANCE.commit,
                )

    def test_epoch_authority_and_range_mutations_are_rejected(self) -> None:
        """Catches mutable tuple fields, ancestry drift, prefix edits, and bad ranges."""
        self.assertTrue(hasattr(state_record, "MIGRATION_EPOCHS"))
        epochs = state_record.MIGRATION_EPOCHS
        self.assertGreaterEqual(len(epochs), 3)

        scalar_mutations = {
            "commit": replace(epochs[1], commit=epochs[0].commit),
            "blob": replace(epochs[1], blob="0" * 40),
            "byte_count": replace(epochs[1], byte_count=epochs[1].byte_count + 1),
            "sha256": replace(epochs[1], sha256="0" * 64),
        }
        for field, mutated_epoch in scalar_mutations.items():
            with self.subTest(field=field), mock.patch.object(
                state_record,
                "MIGRATION_EPOCHS",
                (epochs[0], mutated_epoch, *epochs[2:]),
            ):
                with self.assertRaises(migrate_state_record.MigrationError):
                    migrate_state_record.build_verification_migration(
                        ROOT, "f921062ba4fbf3341b02d2ac826bb3d84cc91a64"
                    )

    def test_final_authority_and_each_boundary_override_mutation_are_rejected(self) -> None:
        final = state_record.FINAL_MIGRATION_PROVENANCE
        authority_mutations = (
            replace(final, commit=state_record.FROZEN_MIGRATION_PROVENANCE.commit),
            replace(final, blob="0" * 40),
            replace(final, byte_count=final.byte_count + 1),
            replace(final, sha256="0" * 64),
        )
        for mutated in authority_mutations:
            with self.subTest(authority=mutated), mock.patch.object(
                state_record, "FINAL_MIGRATION_PROVENANCE", mutated
            ):
                with self.assertRaises(migrate_state_record.MigrationError):
                    migrate_state_record.build_verification_migration(
                        ROOT, final.commit
                    )

    def test_archived_manifest_and_preserved_wrapper_mutations_are_rejected(self) -> None:
        final = state_record.FINAL_MIGRATION_PROVENANCE
        epochs = state_record.MIGRATION_EPOCHS
        mutations = ("archive", "wrapper_metadata", "wrapper_payload")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                scratch = self.final_scratch(directory)
                if mutation == "archive":
                    path = scratch / state_record.ARCHIVED_MIGRATION_MANIFEST
                    path.write_bytes(path.read_bytes() + b"X")
                else:
                    path = scratch / (
                        ".agents/state-events/2026-08/"
                        "STATE-20260809T083000-001.md"
                    )
                    raw = path.read_bytes()
                    if mutation == "wrapper_metadata":
                        path.write_bytes(raw.replace(b"# Imported", b"! Imported", 1))
                    else:
                        path.write_bytes(raw.replace(b"Vulkan", b"vulkan", 1))

                with self.assertRaises(migrate_state_record.MigrationError):
                    migrate_state_record.build_verification_migration(
                        scratch, state_record.FINAL_MIGRATION_PROVENANCE.commit
                    )

        for event_id, boundary in state_record.FINAL_BOUNDARY_OVERRIDES.items():
            mutated_overrides = dict(state_record.FINAL_BOUNDARY_OVERRIDES)
            mutated_overrides[event_id] = (boundary[0], boundary[1] - 1)
            with self.subTest(boundary=event_id), mock.patch.object(
                state_record, "FINAL_BOUNDARY_OVERRIDES", mutated_overrides
            ):
                with self.assertRaises(migrate_state_record.MigrationError):
                    migrate_state_record.build_verification_migration(
                        ROOT, final.commit
                    )

        with mock.patch.object(
            state_record,
            "MIGRATION_EPOCHS",
            (epochs[0], epochs[2], epochs[1]),
        ):
            with self.assertRaisesRegex(
                migrate_state_record.MigrationError, "ordered ancestor"
            ):
                migrate_state_record.build_verification_migration(
                    ROOT, "f921062ba4fbf3341b02d2ac826bb3d84cc91a64"
                )

        original_read_source = migrate_state_record.read_source

        def prefix_mutating_read_source(root: Path, revision: str, source_path: str = ".agents/state.md"):
            resolved, blob, source = original_read_source(root, revision, source_path)
            if resolved == epochs[1].commit:
                source = b"X" + source[1:]
            return resolved, blob, source

        mutated_second = replace(
            epochs[1], sha256=hashlib.sha256(b"X" + subprocess.check_output(
                ["git", "show", f"{epochs[1].commit}:.agents/state.md"], cwd=ROOT
            )[1:]).hexdigest()
        )
        with mock.patch.object(
            state_record,
            "MIGRATION_EPOCHS",
            (epochs[0], mutated_second, *epochs[2:]),
        ), mock.patch.object(
            migrate_state_record, "read_source", side_effect=prefix_mutating_read_source
        ):
            with self.assertRaisesRegex(migrate_state_record.MigrationError, "prefix"):
                migrate_state_record.build_verification_migration(
                    ROOT, "f921062ba4fbf3341b02d2ac826bb3d84cc91a64"
                )

        tail_range = epochs[2].ranges[0]
        range_mutations = {
            "overlap": replace(tail_range, start=tail_range.start - 1),
            "gap": replace(tail_range, start=tail_range.start + 1),
            "epoch boundary": replace(tail_range, end=epochs[2].byte_count + 1),
        }
        for case, mutated_range in range_mutations.items():
            with self.subTest(case=case), mock.patch.object(
                state_record,
                "MIGRATION_EPOCHS",
                (*epochs[:2], replace(epochs[2], ranges=(mutated_range,))),
            ):
                with self.assertRaises(migrate_state_record.MigrationError):
                    migrate_state_record.build_verification_migration(
                        ROOT, "f921062ba4fbf3341b02d2ac826bb3d84cc91a64"
                    )

    def test_verify_accepts_exact_generated_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)

            result = repo.run("--verify")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("byte-exact", result.stdout)
            self.assertIn(
                f"using frozen provenance {repo.source_revision}", result.stdout
            )
            self.assertNotIn("final/current snapshot authority", result.stdout)

    def test_verify_accepts_newer_revision_with_identical_source_bytes(self) -> None:
        """Current-base verification must not rewrite immutable provenance."""
        frozen_commit = "994cd8d4122ecf44f72d51fabd61c45adaaea9d3"
        frozen_blob = "93a8d0da802a7ea7cbea4bee3bedffb4d90459f7"
        frozen_digest = "00c08e974724c19b5f79cce44df71c6fbfef4db32aa6acb545ef56546e3bb5e6"
        identical_source_commit = "6db9ec5095ea9c7ce56184abb86d1130ee7c04c4"
        current_blob = subprocess.check_output(
            ["git", "rev-parse", f"{identical_source_commit}:.agents/state.md"],
            cwd=ROOT,
            text=True,
        ).strip()
        current_source = subprocess.check_output(
            ["git", "show", f"{identical_source_commit}:.agents/state.md"], cwd=ROOT
        )
        self.assertEqual(current_blob, frozen_blob)
        self.assertEqual(len(current_source), 3191283)
        self.assertEqual(hashlib.sha256(current_source).hexdigest(), frozen_digest)
        self.assertEqual(
            subprocess.check_output(
                ["git", "rev-parse", f"{frozen_commit}^{{commit}}"],
                cwd=ROOT,
                text=True,
            ).strip(),
            frozen_commit,
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-qm", "new base, same source"],
                cwd=repo.root,
                check=True,
            )
            current_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo.root, text=True
            ).strip()

            result = repo.run("--verify", current_revision)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(current_revision, result.stdout)
            self.assertIn(repo.source_revision, result.stdout)

    def test_verify_rejects_newer_revision_with_different_source_bytes(self) -> None:
        """A matching commit shape cannot hide source content drift."""
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            state_path = repo.root / ".agents/state.md"
            compatibility_stub = state_path.read_bytes()
            state_path.write_bytes(SOURCE + b"\nnew source bytes\n")
            subprocess.run(
                ["git", "add", ".agents/state.md"], cwd=repo.root, check=True
            )
            subprocess.run(
                ["git", "commit", "-qm", "new base, changed source"],
                cwd=repo.root,
                check=True,
            )
            current_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo.root, text=True
            ).strip()
            state_path.write_bytes(compatibility_stub)

            result = repo.run("--verify", current_revision)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from final migration epoch", result.stderr)

    def test_verify_rejects_boundary_payload_header_and_missing_file_mutations(self) -> None:
        """Catches gaps/overlaps, payload edits, schema drift, and incomplete output."""
        mutations = (
            "gap",
            "overlap",
            "payload",
            "header",
            "missing",
            "provenance_missing",
            "provenance_commit",
            "provenance_blob",
            "provenance_size",
            "provenance_hash",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                repo = MigrationRepo(directory)
                applied = repo.run("--apply")
                self.assertEqual(applied.returncode, 0, applied.stderr)
                manifest_path = repo.root / ".agents/completed/state-migration-manifest.csv"
                rows = repo.manifest_rows()
                if mutation in {"gap", "overlap"}:
                    rows[3][6] = str(int(rows[3][6]) + (1 if mutation == "gap" else -1))
                    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)
                elif mutation == "payload":
                    event_path = repo.root / rows[3][10]
                    event_path.write_bytes(event_path.read_bytes().replace(b"Alpha", b"Omega", 1))
                elif mutation == "header":
                    rows[0][0] = "wrong_id"
                    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)
                elif mutation == "missing":
                    (repo.root / rows[3][10]).unlink()
                else:
                    if mutation == "provenance_missing":
                        del rows[1]
                    else:
                        column = {
                            "provenance_commit": 1,
                            "provenance_blob": 2,
                            "provenance_size": 3,
                            "provenance_hash": 4,
                        }[mutation]
                        rows[1][column] = "0" * (64 if column == 4 else 40)
                    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)

                result = repo.run("--verify")

                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.strip(), result)

    def test_missing_source_revision_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)

            result = repo.run("--verify", "not-a-revision")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source revision", result.stderr.lower())
            self.assertIn("not-a-revision", result.stderr)

    def test_invalid_anchor_timestamp_is_rejected_before_writing(self) -> None:
        """Catches generation of invalid structured metadata from a malformed anchor."""
        source = b"## Invalid date\n<!-- state: 2026-13-40 -->\npayload\n"
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory, source)

            result = repo.run("--apply")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid timestamp", result.stderr.lower())
            self.assertFalse((repo.root / ".agents/state.csv").exists())


if __name__ == "__main__":
    unittest.main()
