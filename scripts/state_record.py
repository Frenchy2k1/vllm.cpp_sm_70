#!/usr/bin/env python3
"""Parsing and validation primitives for the structured state record."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


MANIFEST_HEADER = ("schema_version", "shard_id", "index_path", "event_prefix")
EVENT_HEADER = (
    "event_id",
    "occurred_at",
    "kind",
    "subject_ids",
    "phase",
    "outcome",
    "commit",
    "pr",
    "spec",
    "evidence_path",
    "supersedes",
    "summary",
    "next_action",
)

MANIFEST_MAX_BYTES = 64 * 1024
SHARD_MAX_BYTES = 256 * 1024
SHARD_MAX_EVENTS = 512
EVENT_MAX_BYTES = 32 * 1024
SUMMARY_MAX_CHARS = 200
NEXT_ACTION_MAX_CHARS = 240
SHARD_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{3}$")
NEW_EVENT_ID_RE = re.compile(r"^STATE-(\d{8}T\d{6})-\d{3}$")
LEGACY_EVENT_ID_RE = re.compile(r"^STATE-LEGACY-\d{6}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PR_RE = re.compile(r"^pr:[1-9]\d*$")
SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
KINDS = frozenset({"legacy_import", "checkpoint", "decision", "handoff", "correction"})
PHASES = frozenset(
    {"spec", "implementation", "review", "verification", "integration", "handoff", "operations"}
)
OUTCOMES = frozenset(
    {"started", "checkpoint", "passed", "failed", "blocked", "landed", "closed", "superseded"}
)
REQUIRED_SECTIONS = ("Context", "Outcome", "Evidence", "Next action")
LEGACY_BEGIN = b"<!-- legacy-payload:begin -->\n"
LEGACY_END = b"<!-- legacy-payload:end -->\n"
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
STUB_MAX_BYTES = 4 * 1024
LEGACY_LINK_RE = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/blob/([0-9a-f]{40})/\.agents/state\.md"
)


@dataclass(frozen=True)
class MigrationProvenance:
    commit: str
    blob: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class LegacyRange:
    event_id: str
    occurred_at: str
    start: int
    end: int


@dataclass(frozen=True)
class MigrationEpoch:
    commit: str
    blob: str
    byte_count: int
    sha256: str
    ranges: tuple[LegacyRange, ...]


FROZEN_MIGRATION_PROVENANCE = MigrationProvenance(
    commit="994cd8d4122ecf44f72d51fabd61c45adaaea9d3",
    blob="93a8d0da802a7ea7cbea4bee3bedffb4d90459f7",
    byte_count=3191283,
    sha256="00c08e974724c19b5f79cce44df71c6fbfef4db32aa6acb545ef56546e3bb5e6",
)

PREVIOUS_FINAL_MIGRATION_PROVENANCE = MigrationProvenance(
    commit="dc2139b3ff95f6de9b6a8ec8cae4bd5d40262dc7",
    blob="e79acd955e7042f9d028e18d41569ff5b67e32c3",
    byte_count=3225646,
    sha256="5c389a0cd84e6834263630af9e9d5a7a64f131ef06d479670dc6e3dc942ec202",
)

FINAL_MIGRATION_PROVENANCE = MigrationProvenance(
    commit="776c56f1c8b78ab69ea01e14759187b243b24d9e",
    blob="e266a8892401bc744955ccc1cb3bc75f64e4f399",
    byte_count=3231342,
    sha256="913c76a2ab8303b1b5ad7dae3ec7876c5e29f0c319670b86eb646c4baa3119b6",
)

ARCHIVED_MIGRATION_MANIFEST = ".agents/completed/state-migration-manifest-f921.csv"
ARCHIVED_MIGRATION_MANIFEST_BYTES = 27431
ARCHIVED_MIGRATION_MANIFEST_SHA256 = (
    "2f8ea0da82038cb1b297b53b2b28f93a02d2963170b13572a6f676299b439113"
)
PRESERVED_WRAPPER_COUNT = 148
PRESERVED_WRAPPER_INVENTORY_SHA256 = (
    "87680b6a195552450a9f83f7711a2d352dc468ac6bcdf1650f90d711d5342b59"
)
PREVIOUS_FINAL_WRAPPER_COUNT = 156
PREVIOUS_FINAL_WRAPPER_INVENTORY_SHA256 = (
    "bcbd8bace3dbffaa23325b9a59f4b2f2e2754491b9abde7139a2c214f83cedab"
)
PREVIOUS_FINAL_INDEX_ROW_COUNT = 156
PREVIOUS_FINAL_INDEX_ROW_INVENTORY_SHA256 = (
    "00915524b22500a9e3ddb4a727f451c5057315b140241f80409eee1d825545ff"
)
FINAL_INDEX_ROW_COUNT = 157
FINAL_INDEX_ROW_INVENTORY_SHA256 = (
    "9aa5c11c5754e9412d0ea81a3f261a0e56388704eeef8778367f56ebaf7f3926"
)
FINAL_BOUNDARY_OVERRIDES = {
    "STATE-20260809T001000-001": (3196620, 3197705),
    "STATE-20260809T083000-001": (3197705, 3200170),
    "STATE-20260809T110000-001": (3200170, 3202350),
    "STATE-20260809T130000-001": (3202350, 3205680),
}
FINAL_APPEND_EVENT_ID = "STATE-20260809T150000-001"
FINAL_APPEND_BOUNDARY_OVERRIDES = {
    "STATE-20260809T140000-002": (3221201, 3225646),
    FINAL_APPEND_EVENT_ID: (3225646, 3231342),
}
FINAL_APPEND_INDEX_ROW_SHA256 = (
    "beb95851a72f6b15939312c2ef6e98a139155620f5578ffd8549c13901533e85"
)
FINAL_APPEND_WRAPPER_SHA256 = (
    "d344a78d79bcac347617bde1ab0d0f3d42fe4dc9e504a8c60ccdb05936ddb383"
)
FINAL_WRAPPER_COUNT = 157
FINAL_WRAPPER_INVENTORY_SHA256 = (
    "5d0cd0c557bbbd7832a97bb399b27d15900ac6efb2342658a2090e341ad06c45"
)
FINAL_NEW_EVENT_IDS = (
    "STATE-20260808T210000-003",
    "STATE-20260808T220000-002",
    "STATE-20260808T230000-002",
    "STATE-20260808T233000-002",
    "STATE-20260808T234500-002",
    "STATE-20260809T140000-001",
    "STATE-20260809T140000-002",
)
FINAL_NEW_INDEX_ROW_SHA256 = {
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
}

MIGRATION_EPOCHS = (
    MigrationEpoch(
        commit=FROZEN_MIGRATION_PROVENANCE.commit,
        blob=FROZEN_MIGRATION_PROVENANCE.blob,
        byte_count=FROZEN_MIGRATION_PROVENANCE.byte_count,
        sha256=FROZEN_MIGRATION_PROVENANCE.sha256,
        ranges=(),
    ),
    MigrationEpoch(
        commit="eee64f81812f97d87190d715c41dd004c1a89d9a",
        blob="b8c38281f5e3ef54d0be30f8c2c234a8ba754fd5",
        byte_count=3195928,
        sha256="ff9a66b618777f88bcb025a9636c04e87f9457d2c96606b6bd99eac151513a0f",
        ranges=(
            LegacyRange(
                "STATE-20260809T083000-001",
                "2026-08-09T08:30:00Z",
                3191283,
                3193748,
            ),
            LegacyRange(
                "STATE-20260809T110000-001",
                "2026-08-09T11:00:00Z",
                3193748,
                3195928,
            ),
        ),
    ),
    MigrationEpoch(
        commit="161bf64b71413db3fbdffe27e3d8765cc2138d2f",
        blob="48588f1f9154ea0bbc334b278d9d339217fa181f",
        byte_count=3199258,
        sha256="53e41e89e6d5de1b57db20654296392946ff2484a607ba0cbd142cca5bcb043f",
        ranges=(
            LegacyRange(
                "STATE-20260809T130000-001",
                "2026-08-09T13:00:00Z",
                3195928,
                3199258,
            ),
        ),
    ),
)


def event_order_key(event_id: str) -> tuple[int, str]:
    """Order unsortable legacy imports before timestamped event IDs."""
    return (0 if LEGACY_EVENT_ID_RE.fullmatch(event_id) else 1, event_id)


@dataclass(frozen=True)
class Shard:
    schema_version: str
    shard_id: str
    index_path: str
    event_prefix: str


@dataclass(frozen=True)
class Event:
    event_id: str
    occurred_at: str
    kind: str
    subject_ids: str
    phase: str
    outcome: str
    commit: str
    pr: str
    spec: str
    evidence_path: str
    supersedes: str
    summary: str
    next_action: str


def parse_manifest(root: Path) -> tuple[list[Shard], list[str]]:
    path = root / ".agents/state.csv"
    errors: list[str] = []
    try:
        if path.stat().st_size > MANIFEST_MAX_BYTES:
            return [], [f"{path}: manifest exceeds the 64 KiB limit"]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, strict=True)
            header = tuple(next(reader, ()))
            if header != MANIFEST_HEADER:
                return [], [f"{path}: expected manifest header {MANIFEST_HEADER!r}"]
            shards: list[Shard] = []
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                location = f"{path}:{line_number}"
                if len(row) != len(MANIFEST_HEADER):
                    errors.append(
                        f"{location}: expected {len(MANIFEST_HEADER)} columns, "
                        f"found {len(row)}"
                    )
                    continue
                shard = Shard(*row)
                if any(any(control in field for control in ("\r", "\n", "\0")) for field in row):
                    errors.append(f"{location}: control character in manifest scalar")
                if shard.schema_version != "1":
                    errors.append(f"{location}: schema version must be 1")
                if SHARD_ID_RE.fullmatch(shard.shard_id) is None:
                    errors.append(f"{location}: invalid shard id {shard.shard_id!r}")
                expected_index = f".agents/state-index/{shard.shard_id}.csv"
                if shard.index_path != expected_index:
                    errors.append(
                        f"{location}: index path must be {expected_index!r}"
                    )
                period = shard.shard_id[:7]
                expected_prefix = f".agents/state-events/{period}/"
                if shard.event_prefix != expected_prefix:
                    errors.append(
                        f"{location}: event prefix must be {expected_prefix!r}"
                    )
                shards.append(shard)
    except (OSError, UnicodeError, csv.Error, TypeError) as exc:
        errors.append(f"{path}: cannot parse manifest: {exc}")
        return [], errors
    shard_ids = [shard.shard_id for shard in shards]
    if shard_ids != sorted(set(shard_ids)):
        errors.append(f"{path}: shard IDs must be unique and in increasing order")
    if errors:
        return [], errors
    return shards, errors


def _has_control(field: str) -> bool:
    return any(control in field for control in ("\r", "\n", "\0"))


def _valid_utc_timestamp(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _valid_calendar_date(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _validate_evidence(root: Path, event: Event, *, legacy: bool) -> list[str]:
    path = root / event.evidence_path
    errors: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: cannot read evidence: {exc}"]
    if not legacy and len(raw) > EVENT_MAX_BYTES:
        errors.append(f"{path}: post-cutover event exceeds the 32 KiB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        return [*errors, f"{path}: evidence is not UTF-8: {exc}"]
    marker = f"<!-- state-event: {event.event_id} -->"
    marker_lines = [line for line in text.splitlines() if "<!-- state-event:" in line]
    if marker_lines != [marker]:
        errors.append(
            f"{path}: event must contain exactly one event marker matching {event.event_id}"
        )
    if legacy:
        try:
            read_legacy_payload(path)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
        return errors
    for section in REQUIRED_SECTIONS:
        match = re.search(
            rf"(?ms)^## {re.escape(section)}\s*\n(.+?)(?=^## |\Z)", text
        )
        if match is None or not match.group(1).strip():
            errors.append(f"{path}: required section {section!r} is missing or empty")
    return errors


def _event_scalar_errors(event: Event, shard: Shard, location: str) -> list[str]:
    errors: list[str] = []
    fields = tuple(getattr(event, name) for name in EVENT_HEADER)
    if any(_has_control(field) for field in fields):
        errors.append(f"{location}: control character in event scalar")
    legacy = event.kind == "legacy_import"
    new_match = NEW_EVENT_ID_RE.fullmatch(event.event_id)
    if not legacy and new_match is None:
        errors.append(f"{location}: invalid post-cutover event ID {event.event_id!r}")
    if (
        legacy
        and LEGACY_EVENT_ID_RE.fullmatch(event.event_id) is None
        and new_match is None
    ):
        errors.append(f"{location}: invalid legacy event ID {event.event_id!r}")
    if event.kind not in KINDS:
        errors.append(f"{location}: invalid kind {event.kind!r}")
    if legacy:
        if new_match is not None:
            if not _valid_utc_timestamp(event.occurred_at):
                errors.append(
                    f"{location}: timestamped legacy event requires an RFC 3339 UTC timestamp"
                )
            else:
                compact = datetime.strptime(
                    event.occurred_at, "%Y-%m-%dT%H:%M:%SZ"
                ).strftime("%Y%m%dT%H%M%S")
                if new_match.group(1) != compact:
                    errors.append(
                        f"{location}: event ID timestamp disagrees with occurred_at"
                    )
        elif (
            event.occurred_at
            and not _valid_calendar_date(event.occurred_at)
            and not _valid_utc_timestamp(event.occurred_at)
        ):
            errors.append(f"{location}: invalid legacy timestamp {event.occurred_at!r}")
        if event.subject_ids and any(
            SUBJECT_RE.fullmatch(subject) is None
            for subject in event.subject_ids.split(";")
        ):
            errors.append(f"{location}: invalid subject ID syntax")
        if event.phase and event.phase not in PHASES:
            errors.append(f"{location}: invalid phase {event.phase!r}")
        if event.outcome and event.outcome not in OUTCOMES:
            errors.append(f"{location}: invalid outcome {event.outcome!r}")
    else:
        if not _valid_utc_timestamp(event.occurred_at):
            errors.append(f"{location}: invalid timestamp {event.occurred_at!r}")
        elif new_match is not None:
            compact = datetime.strptime(
                event.occurred_at, "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y%m%dT%H%M%S")
            if new_match.group(1) != compact:
                errors.append(f"{location}: event ID timestamp disagrees with occurred_at")
        if not event.subject_ids:
            errors.append(f"{location}: subject IDs are required")
        elif any(
            SUBJECT_RE.fullmatch(subject) is None
            for subject in event.subject_ids.split(";")
        ):
            errors.append(f"{location}: invalid subject ID syntax")
        if event.phase not in PHASES:
            errors.append(f"{location}: invalid phase {event.phase!r}")
        if event.outcome not in OUTCOMES:
            errors.append(f"{location}: invalid outcome {event.outcome!r}")
        if not event.summary:
            errors.append(f"{location}: summary is required")
        if not event.next_action:
            errors.append(f"{location}: next action is required")
    if event.commit and COMMIT_RE.fullmatch(event.commit) is None:
        errors.append(f"{location}: commit must be a lowercase 40-character SHA")
    if event.pr and PR_RE.fullmatch(event.pr) is None:
        errors.append(f"{location}: PR must use pr:<number>")
    expected_evidence = f"{shard.event_prefix}{event.event_id}.md"
    if event.evidence_path != expected_evidence:
        errors.append(f"{location}: evidence path must be {expected_evidence!r}")
    if len(event.summary) > SUMMARY_MAX_CHARS:
        errors.append(f"{location}: summary exceeds {SUMMARY_MAX_CHARS} characters")
    if len(event.next_action) > NEXT_ACTION_MAX_CHARS:
        errors.append(
            f"{location}: next action exceeds {NEXT_ACTION_MAX_CHARS} characters"
        )
    return errors


def parse_events(root: Path, shards: list[Shard]) -> tuple[list[Event], list[str]]:
    events: list[Event] = []
    errors: list[str] = []
    for shard in shards:
        path = root / shard.index_path
        try:
            if path.stat().st_size > SHARD_MAX_BYTES:
                errors.append(f"{path}: index shard exceeds the 256 KiB limit")
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle, strict=True)
                header = tuple(next(reader, ()))
                if header != EVENT_HEADER:
                    errors.append(f"{path}: expected event header {EVENT_HEADER!r}")
                    continue
                shard_events: list[Event] = []
                for line_number, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    location = f"{path}:{line_number}"
                    if len(row) != len(EVENT_HEADER):
                        errors.append(
                            f"{location}: expected {len(EVENT_HEADER)} columns, "
                            f"found {len(row)}"
                        )
                        continue
                    event = Event(*row)
                    errors.extend(_event_scalar_errors(event, shard, location))
                    errors.extend(
                        _validate_evidence(
                            root, event, legacy=event.kind == "legacy_import"
                        )
                    )
                    shard_events.append(event)
        except (OSError, UnicodeError, csv.Error, TypeError) as exc:
            errors.append(f"{path}: cannot parse event index: {exc}")
            continue
        if len(shard_events) > SHARD_MAX_EVENTS:
            errors.append(f"{path}: index shard exceeds the 512-event limit")
        event_ids = [event.event_id for event in shard_events]
        if event_ids != sorted(set(event_ids), key=event_order_key):
            errors.append(f"{path}: event IDs must be unique and in increasing order")
        events.extend(shard_events)
    if errors:
        return [], errors
    return events, []


def load_events(root: Path) -> tuple[list[Event], list[str]]:
    """Load the complete ordered event set from the repository indexes."""
    shards, errors = parse_manifest(root)
    if errors:
        return [], errors
    return parse_events(root, shards)


def read_legacy_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw.count(LEGACY_BEGIN) != 1 or raw.count(LEGACY_END) != 1:
        raise ValueError("legacy payload must contain one begin and one end marker")
    start = raw.index(LEGACY_BEGIN) + len(LEGACY_BEGIN)
    end = raw.index(LEGACY_END, start)
    return raw[start:end]


def _git_bytes(root: Path, revision: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _git_blob_oid(root: Path, revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_revision_exists(root: Path, revision: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _structured_history_floor(root: Path, base: str) -> str | None:
    """Return the first structured-state commit after a pre-cutover base."""
    result = subprocess.run(
        ["git", "rev-list", "--reverse", f"{base}..HEAD", "--", ".agents/state.csv"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return next((line for line in result.stdout.splitlines() if line), None)


def _csv_rows(raw: bytes, header: tuple[str, ...], label: str) -> tuple[list[list[str]], list[str]]:
    try:
        text = raw.decode("utf-8")
        reader = csv.reader(text.splitlines(keepends=True))
        actual = tuple(next(reader, ()))
        if actual != header:
            return [], [f"{label}: expected header {header!r}"]
        return [row for row in reader if row], []
    except (UnicodeError, csv.Error) as exc:
        return [], [f"{label}: cannot parse CSV: {exc}"]


def _spec_errors(root: Path, event: Event) -> list[str]:
    if not event.spec:
        return []
    path = PurePosixPath(event.spec)
    allowed = (
        event.spec.startswith(".agents/specs/"),
        event.spec.startswith("docs/superpowers/specs/"),
    )
    if path.is_absolute() or ".." in path.parts or not any(allowed):
        return [f"{event.event_id}: spec must be repository-relative under an allowed specs directory"]
    target = root / event.spec
    try:
        resolved = target.resolve(strict=True)
    except OSError:
        return [f"{event.event_id}: spec does not resolve: {event.spec}"]
    if not resolved.is_relative_to(root):
        return [f"{event.event_id}: spec resolves outside the repository"]
    if not resolved.is_file():
        return [f"{event.event_id}: spec does not resolve: {event.spec}"]
    return []


def _relation_errors(root: Path, events: list[Event]) -> list[str]:
    errors: list[str] = []
    event_ids = [event.event_id for event in events]
    if event_ids != sorted(event_ids, key=event_order_key):
        errors.append("event IDs must be globally ordered across index shards")
    if len(event_ids) != len(set(event_ids)):
        errors.append("event IDs must be globally unique across index shards")

    evidence_paths = [event.evidence_path for event in events]
    if len(evidence_paths) != len(set(evidence_paths)):
        errors.append("each evidence path must be named by exactly one index row")
    indexed = set(evidence_paths)
    evidence_root = root / ".agents/state-events"
    existing = {
        path.relative_to(root).as_posix()
        for path in evidence_root.rglob("*.md")
        if path.is_file()
    } if evidence_root.exists() else set()
    for orphan in sorted(existing - indexed):
        errors.append(f"orphan evidence file has no index row: {orphan}")
    for missing in sorted(indexed - existing):
        errors.append(f"indexed evidence file is missing: {missing}")

    positions = {event.event_id: index for index, event in enumerate(events)}
    for index, event in enumerate(events):
        if event.subject_ids:
            subjects = event.subject_ids.split(";")
            if subjects != sorted(set(subjects)):
                errors.append(
                    f"{event.event_id}: subject IDs must be sorted and duplicate-free"
                )
        errors.extend(_spec_errors(root, event))
        if event.kind == "correction":
            if not event.supersedes:
                errors.append(f"{event.event_id}: correction requires supersedes")
            elif event.supersedes not in positions or positions[event.supersedes] >= index:
                errors.append(
                    f"{event.event_id}: supersedes must name an earlier event"
                )
        elif event.supersedes:
            errors.append(
                f"{event.event_id}: supersedes is forbidden outside correction events"
            )
    return errors


def _stub_and_migration_errors(
    root: Path,
    events: list[Event],
    epochs: tuple[MigrationEpoch, ...],
) -> list[str]:
    errors: list[str] = []
    if not epochs:
        return ["migration epoch authority tuple is empty"]
    stub = root / ".agents/state.md"
    try:
        raw_stub = stub.read_bytes()
    except OSError as exc:
        return [f"compatibility stub cannot be read: {exc}"]
    if len(raw_stub) > STUB_MAX_BYTES:
        errors.append("compatibility stub exceeds the 4 KiB limit")
    try:
        stub_text = raw_stub.decode("utf-8")
    except UnicodeError as exc:
        return [*errors, f"compatibility stub is not UTF-8: {exc}"]
    for required in (
        ".agents/state.csv",
        ".agents/state-index/",
        ".agents/state-events/",
        ".agents/completed/state-migration-manifest.csv",
    ):
        if required not in stub_text:
            errors.append(f"compatibility stub must link {required}")
    links = LEGACY_LINK_RE.findall(stub_text)
    expected_links = [epochs[0].commit]
    if len(epochs) > 1:
        expected_links.append(epochs[-1].commit)
    if links != expected_links:
        errors.append("compatibility stub disagrees with migration epoch provenance")

    sources: list[bytes | None] = []
    for index, epoch in enumerate(epochs):
        source = _git_bytes(root, epoch.commit, ".agents/state.md")
        sources.append(source)
        if source is None:
            errors.append(f"migration epoch {index} Git source does not resolve")
            continue
        if _git_blob_oid(root, epoch.commit, ".agents/state.md") != epoch.blob:
            errors.append(f"migration epoch {index} provenance blob does not match Git")
        if len(source) != epoch.byte_count:
            errors.append(f"migration epoch {index} provenance byte count does not match Git")
        if hashlib.sha256(source).hexdigest() != epoch.sha256:
            errors.append(f"migration epoch {index} provenance SHA-256 does not match Git")
        if index and sources[index - 1] is not None:
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", epochs[index - 1].commit, epoch.commit],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if ancestor.returncode != 0:
                errors.append("migration epochs must be in ordered ancestor sequence")
            if not source.startswith(sources[index - 1]):
                errors.append(f"migration epoch {index} changes the cumulative source prefix")

    manifest = root / ".agents/completed/state-migration-manifest.csv"
    try:
        raw_manifest = manifest.read_bytes()
    except OSError as exc:
        return [*errors, f"migration manifest cannot be read: {exc}"]
    rows, csv_errors = _csv_rows(raw_manifest, MIGRATION_HEADER, str(manifest))
    errors.extend(csv_errors)
    if csv_errors:
        return errors

    legacy_events = {event.event_id: event for event in events if event.kind == "legacy_import"}
    seen: set[str] = set()
    expected_start = 0
    reconstructed = bytearray()
    source_index = -1
    for line_number, row in enumerate(rows, start=2):
        location = f"migration manifest:{line_number}"
        if len(row) != len(MIGRATION_HEADER):
            errors.append(f"{location}: expected {len(MIGRATION_HEADER)} columns")
            continue
        record_type, *fields = row
        if record_type == "source":
            source_index += 1
            if source_index >= len(epochs):
                errors.append(f"{location}: unexpected migration source epoch")
                continue
            epoch = epochs[source_index]
            expected = [
                "source", epoch.commit, epoch.blob, str(epoch.byte_count), epoch.sha256,
                "", "", "", "", "", "",
            ]
            if row != expected:
                if source_index == 0:
                    errors.append("migration manifest disagrees with frozen migration provenance")
                else:
                    errors.append(f"{location}: migration manifest disagrees with epoch {source_index}")
            if source_index and expected_start != epochs[source_index - 1].byte_count:
                errors.append(f"{location}: source epoch boundary disagrees with reconstructed range")
            previous = sources[source_index - 1] if source_index else b""
            if source_index and previous is not None and bytes(reconstructed) != previous:
                errors.append(f"{location}: payloads do not reconstruct the preceding epoch")
            continue
        if record_type != "event":
            errors.append(f"{location}: record type must be source or event")
            continue
        source_fields = fields[:4]
        if any(source_fields):
            errors.append(f"{location}: migration event row cannot contain source provenance")
        event_id, start_s, end_s, count_s, digest, evidence_path = fields[4:]
        if event_id in seen:
            errors.append(f"{location}: duplicate migration event ID {event_id}")
        seen.add(event_id)
        event = legacy_events.get(event_id)
        if event is None:
            errors.append(f"{location}: migration row must name one legacy_import event")
            continue
        if evidence_path != event.evidence_path:
            errors.append(f"{location}: migration evidence path disagrees with the index")
        try:
            start, end, count = int(start_s), int(end_s), int(count_s)
        except ValueError:
            errors.append(f"{location}: migration byte ranges must be integers")
            continue
        try:
            payload = read_legacy_payload(root / event.evidence_path)
        except (OSError, ValueError) as exc:
            errors.append(f"{location}: migration payload cannot be read: {exc}")
            continue
        if start != expected_start or end < start:
            errors.append(f"{location}: migration ranges must be contiguous and increasing")
        if source_index < 0:
            errors.append(f"{location}: migration event precedes its source epoch")
        elif end > epochs[source_index].byte_count:
            errors.append(f"{location}: migration range crosses its source epoch boundary")
        if end - start != count or count != len(payload):
            errors.append(f"{location}: migration payload byte count disagrees with its range")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or hashlib.sha256(payload).hexdigest() != digest:
            errors.append(f"{location}: migration payload hash mismatch")
        expected_start = end
        reconstructed.extend(payload)
    if seen != set(legacy_events):
        errors.append("migration manifest must cover every legacy_import event exactly once")
    if source_index + 1 != len(epochs):
        errors.append("migration manifest must contain every configured source epoch exactly once")
    final_source = sources[-1]
    if final_source is not None and bytes(reconstructed) != final_source:
        errors.append("migration manifest payloads do not reconstruct the final source")
    return errors


def _authority_source(
    root: Path, provenance: MigrationProvenance, label: str
) -> tuple[bytes | None, list[str]]:
    errors: list[str] = []
    source = _git_bytes(root, provenance.commit, ".agents/state.md")
    if source is None:
        return None, [f"{label} Git source does not resolve"]
    if _git_blob_oid(root, provenance.commit, ".agents/state.md") != provenance.blob:
        errors.append(f"{label} provenance blob does not match Git")
    if len(source) != provenance.byte_count:
        errors.append(f"{label} provenance byte count does not match Git")
    if hashlib.sha256(source).hexdigest() != provenance.sha256:
        errors.append(f"{label} provenance SHA-256 does not match Git")
    return source, errors


def _historical_epoch_errors(root: Path) -> list[str]:
    errors: list[str] = []
    previous: bytes | None = None
    for index, epoch in enumerate(MIGRATION_EPOCHS):
        provenance = MigrationProvenance(
            epoch.commit, epoch.blob, epoch.byte_count, epoch.sha256
        )
        source, source_errors = _authority_source(
            root, provenance, f"historical migration epoch {index}"
        )
        errors.extend(source_errors)
        if index:
            ancestor = subprocess.run(
                [
                    "git", "merge-base", "--is-ancestor",
                    MIGRATION_EPOCHS[index - 1].commit, epoch.commit,
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if ancestor.returncode != 0:
                errors.append("historical migration epochs must be in ordered ancestor sequence")
            if source is not None and previous is not None and not source.startswith(previous):
                errors.append(f"historical migration epoch {index} changes the cumulative source prefix")
        previous = source
    archive = root / ARCHIVED_MIGRATION_MANIFEST
    try:
        raw_archive = archive.read_bytes()
    except OSError as exc:
        errors.append(f"archived migration manifest cannot be read: {exc}")
    else:
        if (
            len(raw_archive) != ARCHIVED_MIGRATION_MANIFEST_BYTES
            or hashlib.sha256(raw_archive).hexdigest()
            != ARCHIVED_MIGRATION_MANIFEST_SHA256
        ):
            errors.append("archived migration manifest bytes disagree with authority")
    return errors


def _final_index_row_errors(root: Path, events: list[Event]) -> list[str]:
    if set(FINAL_NEW_INDEX_ROW_SHA256) != set(FINAL_NEW_EVENT_IDS):
        return ["final index row byte authority disagrees with event IDs"]
    raw_by_id: dict[str, bytes] = {}
    for path in sorted((root / ".agents/state-index").glob("*.csv")):
        try:
            raw = path.read_bytes()
            records = _raw_csv_record_bytes(raw)
            rows, csv_errors = _csv_rows(raw, EVENT_HEADER, str(path))
        except OSError as exc:
            return [f"final index row bytes cannot be read: {exc}"]
        if records is None or not records or csv_errors:
            return ["final index row bytes are not valid CSV records"]
        for row, record in zip(rows, records[1:]):
            if len(row) == len(EVENT_HEADER) and row[2] == "legacy_import":
                raw_by_id[row[0]] = record

    digests = [hashlib.sha256(record).hexdigest() for record in raw_by_id.values()]
    errors: list[str] = []
    for event_id, expected_digest in FINAL_NEW_INDEX_ROW_SHA256.items():
        if digests.count(expected_digest) != 1:
            errors.append(
                f"final index row bytes disagree with authority: {event_id}"
            )
    appended = raw_by_id.get(FINAL_APPEND_EVENT_ID)
    if (
        appended is None
        or hashlib.sha256(appended).hexdigest() != FINAL_APPEND_INDEX_ROW_SHA256
    ):
        errors.append(
            f"final index row bytes disagree with authority: {FINAL_APPEND_EVENT_ID}"
        )

    legacy_ids = [event.event_id for event in events if event.kind == "legacy_import"]
    inventory: list[tuple[str, bytes]] = []
    for event_id in legacy_ids:
        raw = raw_by_id.get(event_id)
        if raw is None:
            continue
        inventory.append((
            event_id,
            f"{event_id},{hashlib.sha256(raw).hexdigest()}\n".encode("ascii"),
        ))
    inventory_bytes = [row for _, row in inventory]
    if (
        len(inventory_bytes) != FINAL_INDEX_ROW_COUNT
        or hashlib.sha256(b"".join(inventory_bytes)).hexdigest()
        != FINAL_INDEX_ROW_INVENTORY_SHA256
    ):
        errors.append("final index row inventory disagrees with authority")
    previous_inventory = [
        row for event_id, row in inventory if event_id != FINAL_APPEND_EVENT_ID
    ]
    if (
        len(previous_inventory) != PREVIOUS_FINAL_INDEX_ROW_COUNT
        or hashlib.sha256(b"".join(previous_inventory)).hexdigest()
        != PREVIOUS_FINAL_INDEX_ROW_INVENTORY_SHA256
    ):
        errors.append("previous final index row inventory disagrees with authority")
    return errors


def _final_stub_and_migration_errors(root: Path, events: list[Event]) -> list[str]:
    errors: list[str] = []
    stub = root / ".agents/state.md"
    try:
        raw_stub = stub.read_bytes()
    except OSError as exc:
        return [f"compatibility stub cannot be read: {exc}"]
    if len(raw_stub) > STUB_MAX_BYTES:
        errors.append("compatibility stub exceeds the 4 KiB limit")
    try:
        stub_text = raw_stub.decode("utf-8")
    except UnicodeError as exc:
        return [*errors, f"compatibility stub is not UTF-8: {exc}"]
    for required in (
        ".agents/state.csv",
        ".agents/state-index/",
        ".agents/state-events/",
        ".agents/completed/state-migration-manifest.csv",
        ARCHIVED_MIGRATION_MANIFEST,
    ):
        if required not in stub_text:
            errors.append(f"compatibility stub must link {required}")
    expected_links = [
        FROZEN_MIGRATION_PROVENANCE.commit,
        FINAL_MIGRATION_PROVENANCE.commit,
    ]
    if LEGACY_LINK_RE.findall(stub_text) != expected_links:
        errors.append("compatibility stub disagrees with final migration provenance")

    errors.extend(_historical_epoch_errors(root))
    errors.extend(_final_index_row_errors(root, events))
    final_source, source_errors = _authority_source(
        root, FINAL_MIGRATION_PROVENANCE, "final migration snapshot"
    )
    errors.extend(source_errors)
    previous_source, previous_source_errors = _authority_source(
        root, PREVIOUS_FINAL_MIGRATION_PROVENANCE, "previous final migration snapshot"
    )
    errors.extend(previous_source_errors)
    if (
        final_source is not None
        and previous_source is not None
        and not final_source.startswith(previous_source)
    ):
        errors.append("final migration snapshot changes the previous final prefix")

    manifest = root / ".agents/completed/state-migration-manifest.csv"
    try:
        raw_manifest = manifest.read_bytes()
    except OSError as exc:
        return [*errors, f"migration manifest cannot be read: {exc}"]
    rows, csv_errors = _csv_rows(raw_manifest, MIGRATION_HEADER, str(manifest))
    errors.extend(csv_errors)
    if csv_errors:
        return errors

    expected_source_row = [
        "source",
        FINAL_MIGRATION_PROVENANCE.commit,
        FINAL_MIGRATION_PROVENANCE.blob,
        str(FINAL_MIGRATION_PROVENANCE.byte_count),
        FINAL_MIGRATION_PROVENANCE.sha256,
        "", "", "", "", "", "",
    ]
    if not rows or rows[0] != expected_source_row:
        errors.append("migration manifest disagrees with final snapshot provenance")

    legacy_events = {event.event_id: event for event in events if event.kind == "legacy_import"}
    seen: set[str] = set()
    expected_start = 0
    reconstructed = bytearray()
    wrapper_inventory: list[bytes] = []
    previous_wrapper_inventory: list[bytes] = []
    final_wrapper_inventory: list[bytes] = []
    historical_regenerated = {*FINAL_NEW_EVENT_IDS, "STATE-20260809T130000-001"}
    overrides = dict(FINAL_BOUNDARY_OVERRIDES)
    overrides.update(FINAL_APPEND_BOUNDARY_OVERRIDES)
    for line_number, row in enumerate(rows[1:], start=3):
        location = f"migration manifest:{line_number}"
        if len(row) != len(MIGRATION_HEADER):
            errors.append(f"{location}: expected {len(MIGRATION_HEADER)} columns")
            continue
        if row[0] != "event":
            errors.append(f"{location}: final manifest permits only one source row")
            continue
        if any(row[1:5]):
            errors.append(f"{location}: migration event row cannot contain source provenance")
        event_id, start_s, end_s, count_s, digest, evidence_path = row[5:]
        if event_id in seen:
            errors.append(f"{location}: duplicate migration event ID {event_id}")
        seen.add(event_id)
        event = legacy_events.get(event_id)
        if event is None:
            errors.append(f"{location}: migration row must name one legacy_import event")
            continue
        if evidence_path != event.evidence_path:
            errors.append(f"{location}: migration evidence path disagrees with the index")
        try:
            start, end, count = int(start_s), int(end_s), int(count_s)
        except ValueError:
            errors.append(f"{location}: migration byte ranges must be integers")
            continue
        if start != expected_start or end <= start:
            errors.append(f"{location}: migration ranges must be contiguous and increasing")
        override = overrides.get(event_id)
        if override is not None and (start, end) != override:
            errors.append(f"{location}: migration boundary override disagrees with authority")
        try:
            wrapper = (root / evidence_path).read_bytes()
            payload = read_legacy_payload(root / evidence_path)
        except (OSError, ValueError) as exc:
            errors.append(f"{location}: migration payload cannot be read: {exc}")
            expected_start = end
            continue
        if end - start != count or count != len(payload):
            errors.append(f"{location}: migration payload byte count disagrees with its range")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or hashlib.sha256(payload).hexdigest() != digest:
            errors.append(f"{location}: migration payload hash mismatch")
        if final_source is not None and payload != final_source[start:end]:
            errors.append(f"{location}: extracted payload differs from configured final segment")
        wrapper_row = (
            f"{evidence_path},{hashlib.sha256(wrapper).hexdigest()}\n".encode("ascii")
        )
        final_wrapper_inventory.append(wrapper_row)
        if event_id != FINAL_APPEND_EVENT_ID:
            previous_wrapper_inventory.append(wrapper_row)
        if event_id not in historical_regenerated and event_id != FINAL_APPEND_EVENT_ID:
            wrapper_inventory.append(wrapper_row)
        if (
            event_id == FINAL_APPEND_EVENT_ID
            and hashlib.sha256(wrapper).hexdigest() != FINAL_APPEND_WRAPPER_SHA256
        ):
            errors.append("final appended wrapper bytes disagree with authority")
        reconstructed.extend(payload)
        expected_start = end

    if seen != set(legacy_events):
        errors.append("migration manifest must cover every legacy_import event exactly once")
    if not set(FINAL_NEW_EVENT_IDS).issubset(seen):
        errors.append("migration manifest is missing configured concurrent events")
    if FINAL_APPEND_EVENT_ID not in seen:
        errors.append("migration manifest is missing the configured final append event")
    if (
        len(wrapper_inventory) != PRESERVED_WRAPPER_COUNT
        or hashlib.sha256(b"".join(wrapper_inventory)).hexdigest()
        != PRESERVED_WRAPPER_INVENTORY_SHA256
    ):
        errors.append("preserved wrapper hash inventory disagrees with authority")
    if (
        len(previous_wrapper_inventory) != PREVIOUS_FINAL_WRAPPER_COUNT
        or hashlib.sha256(b"".join(previous_wrapper_inventory)).hexdigest()
        != PREVIOUS_FINAL_WRAPPER_INVENTORY_SHA256
    ):
        errors.append("previous final wrapper hash inventory disagrees with authority")
    if (
        len(final_wrapper_inventory) != FINAL_WRAPPER_COUNT
        or hashlib.sha256(b"".join(final_wrapper_inventory)).hexdigest()
        != FINAL_WRAPPER_INVENTORY_SHA256
    ):
        errors.append("final wrapper hash inventory disagrees with authority")
    if final_source is not None and bytes(reconstructed) != final_source:
        errors.append("migration manifest payloads do not reconstruct the final source")
    return errors


def _base_event_paths(root: Path, revision: str, manifest_rows: list[list[str]]) -> list[str]:
    paths: list[str] = []
    for row in manifest_rows:
        if len(row) != len(MANIFEST_HEADER):
            continue
        shard_raw = _git_bytes(root, revision, row[2])
        if shard_raw is None:
            continue
        event_rows, row_errors = _csv_rows(shard_raw, EVENT_HEADER, f"{revision}:{row[2]}")
        if row_errors:
            continue
        paths.extend(row[9] for row in event_rows if len(row) == len(EVENT_HEADER))
    return paths


def _raw_csv_record_bytes(raw: bytes) -> list[bytes] | None:
    lines = raw.splitlines(keepends=True)
    try:
        reader = csv.reader((line.decode("utf-8") for line in lines), strict=True)
        records: list[bytes] = []
        start = 0
        for _ in reader:
            end = reader.line_num
            records.append(b"".join(lines[start:end]))
            start = end
        return records
    except (UnicodeError, csv.Error):
        return None


def _preserves_raw_rows_in_order(base: bytes, current: bytes) -> bool:
    base_records = _raw_csv_record_bytes(base)
    current_records = _raw_csv_record_bytes(current)
    if base_records is None or current_records is None or not base_records:
        return False
    if not current_records or current_records[0] != base_records[0]:
        return False
    cursor = iter(current_records[1:])
    return all(any(candidate == expected for candidate in cursor) for expected in base_records[1:])


def _git_history_errors(
    root: Path,
    base: str,
    shards: list[Shard],
    *,
    allow_final_snapshot_insertions: bool = False,
) -> list[str]:
    if not _git_revision_exists(root, base):
        return [f"Git base revision does not resolve: {base}"]
    base_manifest = _git_bytes(root, base, ".agents/state.csv")
    if base_manifest is None:
        floor = _structured_history_floor(root, base)
        if floor is None:
            return []
        base = floor
        base_manifest = _git_bytes(root, base, ".agents/state.csv")
        if base_manifest is None:
            return [f"structured history floor cannot be read: {base}"]
    errors: list[str] = []
    current_manifest_path = root / ".agents/state.csv"
    try:
        current_manifest = current_manifest_path.read_bytes()
    except OSError as exc:
        return [f"state manifest cannot be read for Git comparison: {exc}"]
    if not current_manifest.startswith(base_manifest):
        errors.append("state manifest must be byte-preserving append-only against base")
    base_rows, base_csv_errors = _csv_rows(
        base_manifest, MANIFEST_HEADER, f"{base}:.agents/state.csv"
    )
    errors.extend(base_csv_errors)
    latest = shards[-1].index_path if shards else ""
    for row in base_rows:
        if len(row) != len(MANIFEST_HEADER):
            continue
        path = row[2]
        base_bytes = _git_bytes(root, base, path)
        try:
            current_bytes = (root / path).read_bytes()
        except OSError as exc:
            errors.append(f"{path}: cannot compare index shard to base: {exc}")
            continue
        if base_bytes is None:
            errors.append(f"{path}: base index shard cannot be read")
        elif path == latest:
            raw_rows_preserved = (
                allow_final_snapshot_insertions
                and _preserves_raw_rows_in_order(base_bytes, current_bytes)
            )
            if not current_bytes.startswith(base_bytes) and not raw_rows_preserved:
                errors.append(f"{path}: writable shard must be byte-preserving append-only")
        elif current_bytes != base_bytes:
            errors.append(f"{path}: sealed shard is immutable")
    for path in _base_event_paths(root, base, base_rows):
        base_bytes = _git_bytes(root, base, path)
        try:
            current_bytes = (root / path).read_bytes()
        except OSError:
            current_bytes = None
        if base_bytes is None or current_bytes != base_bytes:
            errors.append(f"{path}: immutable evidence changed or disappeared")
    return errors


def validate(
    root: Path,
    *,
    base: str | None = None,
    migration_provenance: MigrationProvenance = FROZEN_MIGRATION_PROVENANCE,
    migration_epochs: tuple[MigrationEpoch, ...] | None = None,
) -> list[str]:
    """Return all structured-record validation failures rooted at *root*."""
    root = root.resolve()
    shards, errors = parse_manifest(root)
    if errors:
        return errors
    events, event_errors = parse_events(root, shards)
    errors.extend(event_errors)
    if event_errors:
        return errors
    errors.extend(_relation_errors(root, events))
    final_snapshot = (
        migration_epochs is None
        and migration_provenance == FROZEN_MIGRATION_PROVENANCE
    )
    if final_snapshot:
        errors.extend(_final_stub_and_migration_errors(root, events))
    else:
        if migration_epochs is None:
            migration_epochs = (
                MigrationEpoch(
                    migration_provenance.commit,
                    migration_provenance.blob,
                    migration_provenance.byte_count,
                    migration_provenance.sha256,
                    (),
                ),
            )
        errors.extend(_stub_and_migration_errors(root, events, migration_epochs))
    if base is not None:
        errors.extend(
            _git_history_errors(
                root,
                base,
                shards,
                allow_final_snapshot_insertions=final_snapshot,
            )
        )
    return errors


def validate_repository(root: Path, base_ref: str | None = None) -> list[str]:
    """Validate the repository using the configured final snapshot authority."""
    return validate(root, base=base_ref)
