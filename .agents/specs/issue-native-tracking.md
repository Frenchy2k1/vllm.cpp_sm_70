# Issue-native tracking protocol

User-directed 2026-08-06. Status: **accepted design, not yet enforced.** This
document is the contract; the migration phases and CI guards named below are the
work it implies. `AGENTS.md`, `.agents/workflow.md` and the checkers are
deliberately untouched until this is reviewed.

## Scope

Move the project's **control plane** — which work exists, what state it is in,
and who owns it — from hand-merged Markdown into GitHub issues, so that tracking
is deterministic and concurrent sessions cannot corrupt it.

Out of scope, and explicitly staying in files: the **doctrine** (`AGENTS.md`,
`.agents/directives.md`, `.agents/discipline.md`, `.agents/gates.md`, the
benchmark and parity-lever protocols) and the **evidence** (`.agents/state.csv`
plus immutable state events,
`.agents/benchmark-record.md`, `.agents/parity-ledger.md`, every
`.agents/specs/` card, `docs/STATUS.md`, `docs/BENCHMARKS.md`,
`docs/FEATURES.md`, goldens). Also out of scope: what work to do (the roadmap's
ordering) and how to do it.

## Our baseline — why this exists

The record is 30 files and ~52k lines under `.agents/`, plus **714 ID'd rows**
across seven area matrices, all of which carry a recognised lifecycle state (the
count comes from `parse_claim_rows` in `check-agent-record.py`, with zero parse
errors; an earlier ad-hoc regex estimate of 393/367 was wrong and is
superseded). Two properties of it are in tension:

- **Evidence** is append-only, git-provenanced, greppable and shipped with the
  code. This works. `benchmark-record.md` exists precisely so a lever is not
  re-run after being closed, and that only works because it is local and
  greppable.
- **Control state** — `State`, `Owner`, and the `coordination.md` claim table —
  is *mutable state concurrently written by parallel worktrees*. This does not
  work, and the failures are recorded:

  - 2026-08-04: a three-way merge silently produced a **variant** of another
    session's binding numbers. No conflict, no marker.
  - union-merging parallel appends interleaved the former monolithic state
    tail, so "newest last" was false and cold resume returned a jumble. The
    structured state cutover removed that writable format and its repair tools.
  - union-resolving the keyed tables **duplicates rows** rather than merging
    them.
  - `coordination.md` is a mutex implemented as a text file merged across
    worktrees. On 2026-08-04 two sessions pushed to `main` within minutes and
    neither claimed anything in it.

Every one of those is the same bug: **one fact stored in two writable places,
reconciled by a three-way merge.**

A second, quieter symptom: **54 rows are simultaneously marked `ACTIVE`.** That
cannot be true. A stale `ACTIVE` cell inside a several-hundred-row table is
invisible rot; an issue with no assignee and no linked PR is visibly stale.

Meanwhile GitHub already carries half the workflow — 44 PRs, branches already
named `row/<ROW-ID>` — while the issue tracker holds exactly one issue. The
mapping already exists by accident.

### Row census (2026-08-06)

| Bucket | Count | Gets an issue? |
|---|---|---|
| `ACTIVE` 54, `SPIKE` 43, `GATING` 10, `BLOCKED` 7, `READY` 6 | **120** | yes — live work |
| `PARTIAL` | **68** | yes — a known open gap is roadmap content |
| `ANCHOR-BACKFILL` | 57 | not at backfill; on transition |
| `INVENTORIED` 449, `DONE` 20 | 469 | no — inventory and history |

**~188 issues at backfill** (user-directed 2026-08-06: include `PARTIAL`). A
`PARTIAL` row is a capability with working modes and explicitly missing ones —
on a public roadmap that is exactly the content outsiders need, and leaving the
68 of them invisible would undersell what is genuinely open. `ANCHOR-BACKFILL`
is evidence debt rather than a capability gap, so it stays file-side until
someone picks it up.

A wholesale conversion would create 714 issues, 449 of them dead inventory. The
tracker is the **live window**; the matrices remain the **permanent inventory**.

**Coverage caveat carried into P0.** `check-agent-record.py`'s `MATRIX_PATHS`
covers only 5 of the 7 matrices; `feature-matrix.md` and `sglang-matrix.md` hold
11 live rows it never sees. The audit and the backfill must cover all seven, or
those rows become unaudited public issues.

## Design

### The principle

**Every field has exactly one writable home.** This is the whole design; the
rest is consequence.

| Field | Home | Why |
|---|---|---|
| `ID`, `Item`, `Upstream`, `Our code`, `Tests/evidence`, `Spike/spec` | matrix row (file) | durable evidence anchors; CI already verifies path class and line range |
| `State` | issue label | volatile, concurrently mutated |
| `Owner` / claim | issue assignee | a real server-side lock, not a text-file mutex |
| Dependencies | issue links | |
| Per-attempt narrative | issue comments, summarising the files | |

Consequence: **the `State` and `Owner` columns are removed from the matrices**
and replaced by one `Issue` column holding `#N`. The `coordination.md`
active-claim table is retired.

### Lifecycle

An issue is the live window of a row. The membership rule is exact and
bidirectional, because that is what makes the tracker deterministic:

**An open issue exists for a row if and only if its state is `SPIKE`, `READY`,
`ACTIVE`, `GATING`, `PARTIAL` or `BLOCKED`.**

- **opened** when a row enters that set;
- **closed** by the landing commit at `DONE`, at which point the durable anchors
  are written back into the matrix row.

`INVENTORIED`, `DONE` and `OUT-OF-SCOPE` never carry an open issue.
`ANCHOR-BACKFILL` does not either — it is evidence debt on already-landed code,
and it gains an issue only when someone claims the backfill, at which point the
row moves into the live set.

### Keys, labels, milestones

**Row ID stays the primary handle.** Branches remain `row/<ROW-ID>`, matrices
key on Row ID, issue titles are prefixed `[ENG-SCHED-CORE] …`, and `#N` is a
pointer. Nothing existing is renamed, no historical reference in `state.md` or
the ledger breaks, and the ID survives if the tracker is ever swapped out. CI
asserts the ID↔issue mapping is bijective.

Labels mirror the existing tabular lifecycle rather than inventing one:

- `state:spike|ready|active|gating|partial|blocked` — a closed issue is `DONE`;
- `area:engine|model|quant|kernel|backend|feature|sglang` — names the owning
  matrix;
- `tier:T0|T1|T2|T3`;
- `blocked:hardware|external|upstream`;
- `roadmap` — the public headline tracks.

**Milestones are the roadmap's ordered blocks** (order-0 perf closure,
`ROAD-V1-MM`, backend expansion, …). Milestone order *is* the portfolio order,
which produces the public ordered roadmap for free.

Issues are written to be read by outsiders, not only by agents: a prose summary
of what the row is and why it matters, above the machine fields.

### The claim becomes an assignment

`scripts/agent-role.py claim helper --row ENG-FOO` today writes a
`coordination.md` row. It will instead, in one server-side write:

1. refuse if the issue is already assigned;
2. assign the issue and set `state:active`;
3. create the `row/ENG-FOO` worktree and branch;
4. open the draft PR.

Assignment is atomic and unmergeable, so the claim race disappears. The
operator/helper roles, the coordinator record (an exclusive lock when this was
written; a record of who is coordinating where since issue #285), and the
"helper works in a worktree and opens a draft PR at the start" rule are
unchanged — only the *medium* of the claim changes.

### Read cache for offline work

`.agents/rows.generated.md` is an **untracked, machine-written** projection of
the open issues (`ID | state | assignee | issue | milestone`), refreshed from the
API by `scripts/sync-rows.py`, which `agent-preflight.sh` runs at session start.
It is never hand-edited and never committed, so it can never conflict and never
churns `main`. A fresh clone with no network simply has no cache until its first
online run.

`.agents/NOW.md` becomes generated by the same script: the open
`state:active`/`state:gating` issues plus the gate being chased. "Generated,
therefore always current" replaces `check-now-current.py`'s hand-maintained
freshness coupling.

### What `coordination.md` keeps

The active-claim table and the completed-claim archive go (the archive to
`.agents/completed/`). The **row contract**, the **spike gate**, the canonical
hierarchy, and the dependency and GPU-lock rules stay — those are doctrine and
belong in files. The file shrinks to a few hundred lines of contract.

## Enforcement

The split follows what is provable from the tree.

**Offline, local, in `agent-preflight.sh`** — unchanged in spirit. Everything
provable from the tree stays provable from the tree: `check-doc-checkpoint.py`,
`check-state-record.py`, `check-protocol-consistency.py`, and every anchor
path-class and line-range check in `check-agent-record.py`.

**CI-only, new `scripts/check-issue-record.py`** (Actions already provides a
token):

1. every matrix `Issue #N` resolves, and that issue's Row ID matches the row;
2. no orphan open issues — every one has a matrix row;
3. **every row in a live state has an open issue, and every open issue's row is
   in a live state** — the membership rule above, enforced in both directions.
   This is what makes the tracker exhaustive rather than merely consistent;
4. no two open issues share a Row ID;
5. a **closed** issue implies its matrix row carries `DONE`-grade anchors — code
   + tests/evidence + spec + exact parity-ledger line + closing commit present
   in Git history. This is today's contract, preserved verbatim;
6. `state:ready` or later implies a resolving `.agents/specs/<slug>.md` exists
   in the tree. **This preserves the T0 spike gate**, which is the one gate most
   at risk from the migration.

**Degradation rule.** When `gh` is unavailable or unauthenticated, the
issue-dependent checks *skip loudly* and never block local work; they are
**required in CI**, so nothing merges unreconciled. Local stays offline-capable,
the merge gate stays strict. Never weaken a checker to make a transition pass —
repair the record.

`check-agent-record.py` loses its `State`/`Owner` validation and gains `Issue`
column validation; its mutation suite moves in the same change.
`claim-view.py` and `ready-for-helper.py` read the cache instead of
`coordination.md`.

### The closing keyword must live in the commit message

Because `gh pr merge` squashes as `localai-bot`, this project lands squashes
locally via `commit-tree` and a direct push to `main`. GitHub auto-closes an
issue from a **PR body** keyword only when the PR is merged through GitHub —
which we do not do. A `Closes #N` in the PR body would therefore never fire, and
every issue would stay open forever while the tracker looked authoritative.

**The closing keyword goes in the squash commit message**, next to
`FOLLOWING_AGENTS_PROTOCOL` (GitHub does close issues from commit messages
pushed to the default branch). Preflight checks that a commit on a `row/<ID>`
branch carries it.

## Migration

Six independently landable phases. The tracker keeps working throughout; a wrong
phase is reverted alone. **P3 is the irreversible one and lands only after the
P2 backfill is verified.**

| Phase | Work |
|---|---|
| P0 | **The live-state audit** — reconcile the 54 `ACTIVE` rows against branches and commits → in-flight / landed / abandoned; confirm each of the 68 `PARTIAL` rows names its missing modes; correct the matrices, across all 7 matrices. Pure file-side, no issue machinery |
| P1 | Label and milestone schema, `.github/ISSUE_TEMPLATE/row.yml`, `scripts/sync-rows.py`, `scripts/check-issue-record.py` (skip-when-offline, required-in-CI) |
| P2 | Idempotent backfill of the ~188 live rows into issues; dry-run first |
| P3 | Strip `State`/`Owner` from the seven matrices, add `Issue`; update `check-agent-record.py` **and its mutation suite** |
| P4 | `agent-role.py claim` assigns instead of writing a row; `coordination.md` claim table retired to `completed/`; `NOW.md` generated; rewire `claim-view.py`, `ready-for-helper.py`, `check-now-current.py` |
| P5 | Prose: `AGENTS.md` T0, `.agents/workflow.md`, `.agents/directives.md` |

**P0 lands first and stands alone** (user-directed 2026-08-06). It is an audit,
not bookkeeping: it is the first time the 54 `ACTIVE` claims are tested against
reality. Its value does not depend on the migration — a truthful matrix is worth
having even if every later phase stalls — and running it first means the backfill
mints ~188 issues from a corrected record rather than publishing the rot. Because
`PARTIAL` rows now become public issues, P0 also checks that each states its
missing modes, which the row contract already requires; a vague `PARTIAL` row
would otherwise become a vague public issue.

**P5 constraint.** `check-protocol-consistency.py` asserts that the obligation
blocks appear *verbatim* in `.agents/workflow.md` and equal the checker's
constants. Prose and gate must move in the same change — that checker exists
because an obligation was once migrated in `AGENTS.md` and the checker but not
in the manual, which went on instructing agents to do the thing the migration
had removed. Prose is what agents actually read.

## Risks and decisions

**Accepted cost: a lifecycle transition is no longer verifiable from the tree
alone.** CI must query the API, which is flakier than a local file read and
cannot be verified offline. This is paid only for the surfaces where concurrency
is the actual problem — claims and row status — and not for the ~48k lines of
evidence. The degradation rule bounds the blast radius: offline work is never
blocked, and the merge gate is never relaxed.

**Rejected: full conversion.** Moving `state.md` (36,277 lines) and
`benchmark-record.md` (12,145) into issues would trade append-only git
provenance, local grep, offline cold-start and most CI gating for nothing the
control-plane migration does not already deliver. Issue comments are mutable,
deletable, and live on someone else's server; T0 says evidence is moved, never
deleted, and issues cannot promise that.

**Rejected: spike cards as issue bodies.** It would make issues
self-describing, but "no row enters `READY`/`ACTIVE` without a *committed*
`.agents/specs/<slug>.md`" would stop being checkable from a commit, and spikes
would stop being greppable and diffable. Issues link to the spec; the spec stays
a file.

**Rejected: `State` in both places with a CI reconciler.** It preserves today's
offline workflow but re-creates the two-writers problem the migration exists to
kill; the reconciler becomes the thing that breaks.

**Binding-number rule.** A number that exists only in an issue comment is **not
binding**. Binding numbers live in `docs/BENCHMARKS.md`, the parity ledger and
the benchmark record, in git.

**Open risk: issue-tracker lock-in.** Row IDs staying primary is the mitigation
— the matrices remain a complete inventory keyed independently of GitHub, so the
control plane can be rehomed without touching the evidence.
