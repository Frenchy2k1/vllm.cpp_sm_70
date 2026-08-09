# Structured state record design

**Task:** `state-record-structure-1`

**Base audited:** `031410e8f01be61ab57aa228550156a365292351`

**Scope:** migrate only `.agents/state.md`; other ledgers and coordination
records retain their existing formats.

## Problem

`.agents/state.md` is both an append-only checkpoint log and a container for
long-form evidence. At the audited base it is roughly 3.2 MB and 43,000 lines.
That makes cold inspection expensive, gives concurrent sessions one large merge
surface, and leaves lifecycle metadata embedded in prose that checkers cannot
validate reliably.

The existing `NOW.md` snapshot fixes cold-resume cost, but it does not fix the
record itself. The state record needs bounded machine-readable indexes and
small immutable evidence units without discarding or rewriting historical
evidence.

## Goals

1. Remove the multi-megabyte state file from the repository's current tree.
2. Preserve every byte of its historical payload and retain a permanent path
   back to the final pre-migration Git blob.
3. Make checkpoint metadata structured, agent-agnostic, and mechanically
   enforceable after a declared cutover.
4. Keep full reasoning, commands, measurements, failures, and handoff detail in
   human-readable Markdown.
5. Make history immutable: corrections append a superseding event rather than
   editing prior evidence.
6. Preserve the `spec -> implementation -> review -> verification ->
   integration` execution method as structured phase data.
7. Keep every active file bounded as the project continues to grow.

## Non-goals

- This change does not restructure `parity-ledger.md`,
  `benchmark-record.md`, `coordination.md`, roadmaps, or area matrices.
- It does not make the state record the authority for a matrix row's current
  lifecycle value. The owning roadmap and area matrix remain the current-state
  authorities; the state indexes are authoritative for checkpoint-event
  metadata and history.
- It does not infer missing historical owners, phases, outcomes, commits, or row
  identifiers from prose.
- It does not rewrite Git history to remove the old blob.
- It does not generate `NOW.md`; it retains human ownership while enforcing
  freshness coupling.

## Repository layout

```text
.agents/state.md                         # small compatibility stub
.agents/state.csv                        # bounded shard manifest
.agents/state-index/
  2026-07-001.csv                        # canonical event metadata
  2026-08-001.csv
.agents/state-events/
  2026-07/
    STATE-LEGACY-000001.md               # immutable narrative/evidence
  2026-08/
    STATE-20260808T143000-001.md
.agents/completed/
  state-migration-manifest.csv           # source byte ranges and hashes
```

`.agents/state.csv` is a small manifest, not an ever-growing event table. It
contains one row per time-and-size shard:

```csv
schema_version,shard_id,index_path,event_prefix
1,2026-07-001,.agents/state-index/2026-07-001.csv,.agents/state-events/2026-07/
```

The lexicographically greatest shard is writable. Adding a newer shard seals
every older shard. A shard rolls when its next row would exceed 256 KiB or 512
events, so a busy month cannot recreate a large index. The root manifest is
capped at 64 KiB. No wall-clock rule is involved, so CI results remain
deterministic and historical corrections are recorded at the current event
time with `supersedes` rather than backdated into a sealed shard.

## Event index schema

Each event-index shard uses this exact header:

```csv
event_id,occurred_at,kind,subject_ids,phase,outcome,commit,pr,spec,evidence_path,supersedes,summary,next_action
```

### Field rules

- `event_id` is globally unique and sortable. New events use
  `STATE-YYYYMMDDTHHMMSS-NNN`; imported entries without a trustworthy timestamp
  use `STATE-LEGACY-NNNNNN`.
- `occurred_at` is an RFC 3339 UTC timestamp for new events. A legacy import may
  retain a date-only value or be empty when the source did not state a reliable
  time.
- `kind` is one of `legacy_import`, `checkpoint`, `decision`, `handoff`, or
  `correction`.
- `subject_ids` is a semicolon-delimited, sorted, duplicate-free list of exact
  matrix row, claim, governance task, policy, or PR identifiers. It is required
  after cutover and may be empty only for `legacy_import`.
- `phase` is one of `spec`, `implementation`, `review`, `verification`,
  `integration`, `handoff`, or `operations`. It may be empty only for
  `legacy_import`.
- `outcome` is one of `started`, `checkpoint`, `passed`, `failed`, `blocked`,
  `landed`, `closed`, or `superseded`. It may be empty only for
  `legacy_import`.
- `commit` is empty or one lowercase 40-character Git object ID. Abbreviated or
  synthetic SHAs are rejected.
- `pr` is empty or an exact repository PR reference such as `pr:161`.
- `spec` is empty only when the event kind does not require a spec; when present
  it is one repository-relative path under `.agents/specs/` or
  `docs/superpowers/specs/` and must resolve.
- `evidence_path` is required, repository-relative, and names exactly one
  Markdown event file beneath the `event_prefix` declared for its shard.
- `supersedes` is empty or names one earlier event. It is required for
  `correction`, forbidden for other kinds, cannot form a cycle, and never
  authorizes mutation of the superseded event.
- `summary` and `next_action` are single-line UTF-8 scalar fields with bounded
  lengths. `next_action` uses an explicit terminal value when no further action
  is owed.
- CSV fields may not contain CR, LF, NUL, or unbounded evidence prose. Rich
  content belongs in the Markdown event.

The checker owns the conditional required-field matrix by `kind` and `outcome`.
All columns are present on every row even when a permitted value is empty.

## Markdown event contract

New event files contain only one repeated metadata value, the event ID:

```markdown
# Short human title
<!-- state-event: STATE-20260808T143000-001 -->

## Context

## Outcome

## Evidence

## Next action
```

This avoids two competing metadata authorities. The CSV row owns timestamp,
subjects, phase, outcome, commit, PR, spec, supersession, summary, and next
action. Markdown supplies the full narrative and must not redefine those
fields.

Every post-cutover Markdown event is at most 32 KiB. Larger raw output belongs
in the appropriate benchmark/evidence artifact and is linked from the event.
The required sections must be present and non-empty; a terminal event states
explicitly that no next action remains.

Imported event files prepend an event-ID and legacy-source marker to an
otherwise byte-identical payload. The migration verifier extracts only the
payload region when proving preservation. Legacy imports are exempt from the
new section and size contracts but remain immutable after migration.

## Authority and correction semantics

- Event-index CSV shards are the sole authority for event metadata.
- Markdown evidence is the sole home of the event's full narrative.
- Roadmap and area-matrix rows remain the authority for current lifecycle
  state.
- `NOW.md` remains the bounded live snapshot and cannot override either source.
- A correction appends a new `kind=correction` event whose `supersedes` field
  identifies the earlier event. The new event explains the correction and
  supplies replacement facts. Consumers follow the acyclic supersession chain
  to the newest event.
- Deletion, rename, content mutation, or metadata mutation of an indexed event
  is rejected after it has landed.

## Historical migration

The migration is mechanical and lossless, not a semantic rewrite.

1. Freeze the source at an exact base commit and record the blob object ID and
   SHA-256 in the migration manifest.
2. Discover candidate entry boundaries. The enforced tail uses each `##`
   heading plus its immediately following `<!-- state: ... -->` anchor. The
   pre-anchor legacy section uses an explicit reviewed byte-range map; heuristics
   may propose ranges but cannot become authority without complete coverage.
3. Assign deterministic IDs by source order. Duplicate timestamps receive a
   stable sequence suffix. Missing timestamps use the legacy ordinal form.
4. Copy each source range into one event payload without normalization of line
   endings, whitespace, links, spelling, or claims.
5. Index historical entries as `kind=legacy_import`. Populate only fields
   explicitly present and mechanically unambiguous; leave missing structured
   fields empty.
6. Emit `state-migration-manifest.csv` with one typed `source` row carrying the
   exact source commit, Git blob object ID, byte count, and SHA-256, followed by
   typed `event` rows carrying event ID, source byte range, payload byte count,
   payload SHA-256, and evidence path.
7. Verify that ranges are contiguous, non-overlapping, begin at byte zero, end
   at the source size, and that concatenating extracted payloads reproduces the
   original blob byte-for-byte and hash-for-hash.
8. Replace `.agents/state.md` with a small compatibility stub linking to the
   manifest, indexes, event tree, and a permanent Git permalink to the final
   pre-migration blob. Old line-number citations remain resolvable through that
   immutable Git object.

No inferred metadata is presented as historical truth. The first event created
under the strict schema declares the cutover and references this design.

## Validation

A new `scripts/check-state-record.py` and mutation suite enforce:

1. Exact root and event-index CSV headers, dialect, encoding, and scalar limits.
2. Manifest shard IDs, paths, directory placement, size limits, and one writable latest
   shard.
3. Globally unique, correctly formatted, ordered event IDs.
4. Controlled `kind`, `phase`, and `outcome` values plus their conditional
   required fields.
5. Exact subject syntax, sorted uniqueness, full SHAs, PR syntax, and resolving
   specs.
6. One existing evidence file per row and one index row per evidence file.
7. Matching event-ID marker and a valid Markdown section/size contract for new
   events.
8. Acyclic, backward-only supersession references with correction-only use.
9. Immutability of sealed shards and all landed evidence files.
10. Append-only behavior in the writable shard: existing rows retain identical
    bytes and new rows appear only at the end.
11. Complete migration byte-range coverage and payload hashes.
12. A small compatibility stub with the permanent legacy-blob link.

Mutation tests delete, orphan, duplicate, reorder, edit, rename, oversize, and
cross-link representative records. They also alter each migration boundary and
payload byte, remove each required post-cutover section, introduce supersession
cycles, edit sealed shards, and insert multiline CSV prose. Every mutation must
turn the checker red for the intended reason.

## NOW coupling and policy cutover

`check-now-current.py` changes its freshness trigger from a `state.md` append to
an appended row in a state-index shard. It distinguishes events that change
live state or next action from forensic-only corrections:

- `checkpoint`, `handoff`, `started`, `failed`, and `blocked` events require a
  same-change `NOW.md` refresh.
- A terminal `landed` or `closed` event requires refresh when its subject is
  represented in `NOW.md` or it changes the advertised next action.
- A historical migration or evidence-only correction does not manufacture a
  `NOW.md` change.

The policy registry and workflow move together in the atomic cutover:

- `POL-NOW-COUPLING` names appended structured events instead of `state.md`.
- `POL-STATE-ORDER` is replaced by structured ordering, append-only, and
  immutability rules routed to the new checker procedure.
- Path classification recognizes the manifest, index shards, event Markdown,
  migration manifest, checker, tests, and migration tool exactly.
- Preflight and CI run both the checker and its mutation suite.
- `check-state-order.py` and `sort-state-tail.py` are archived or removed only
  in the same commit that installs their complete structured replacement.

Active instructions, checkers, and documentation references move to event IDs
and the new paths. Verbatim legacy payloads may retain historical `state.md`
references; the compatibility stub directs those readers to the frozen blob.

## Delivery sequence

The work is delivered in a new draft PR on `row/state-record-structure-1` with
exact expiring waiver `WAIVER-PR-SIZE-002` for the mechanical file fan-out.

1. **Design commit:** this approved specification only.
2. **Dormant tooling commit:** migration generator, structured checker, and
   focused tests that do not yet replace the live gate.
3. **Atomic cutover commit:** generated event files and indexes, compatibility
   stub, policy/workflow/checker routing, NOW coupling, and retirement of the
   old ordering tools.
4. **Verification commit if needed:** link cleanup and evidence corrections
   found by the full mutation and clean-checkout gates; it may not change the
   approved schema or weaken validation.

The atomic cutover commit must be internally valid. No commit may claim that
the structured state record is active while the old checker remains the gate,
or vice versa.

## Verification and acceptance

The PR is complete only when:

- the source blob and every migrated payload pass the byte-coverage verifier;
- no indexed event is missing, orphaned, duplicated, mutable, or malformed;
- post-cutover events pass the strict schema and Markdown contracts;
- old active references have migrated and the compatibility link resolves;
- `NOW.md` freshness mutations fail and valid coupled changes pass;
- policy, procedure, path classification, preflight, CI, and mutation tests are
  synchronized;
- a clean checkout has no multi-megabyte active state file;
- full local preflight passes on the PR head; an inherited base failure blocks
  publication or integration until the base is repaired and the branch rebased;
- a fresh reviewer performs static inspection and targeted mutation review of
  the immutable head; and
- the operator independently runs the declared gate before integration.

The migration is recoverable without history rewriting: the compatibility stub
identifies the exact pre-migration Git blob, while deterministic tooling can
regenerate and compare the structured tree from that source.

## Risks and controls

- **Silent evidence loss:** prevented by byte-range coverage, per-payload
  hashes, and whole-source reconstruction.
- **Fabricated legacy structure:** prevented by the explicit `legacy_import`
  kind and nullable historical fields.
- **A new large index:** prevented by 256-KiB/512-event shards and a capped
  one-row-per-shard root manifest.
- **Parallel append drift:** reduced to the small writable CSV shard and caught
  by byte-preserving append-only and ordering checks. Reconciliation takes the
  target shard wholesale and reapplies the new row.
- **Metadata duplication:** prevented by allowing Markdown to repeat only the
  event ID.
- **Broken historical citations:** controlled by the small compatibility stub
  and permanent pre-migration blob link.
- **Checker cutover gaps:** prevented by an atomic policy/procedure/checker/data
  commit and mutation-bound CI wiring.
- **Scope expansion:** prevented by leaving every non-state record format
  unchanged in this PR.
