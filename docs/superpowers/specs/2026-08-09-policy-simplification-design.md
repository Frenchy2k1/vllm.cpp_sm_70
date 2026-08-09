# Policy simplification — the code is the state, git is the history

Status: accepted design, user-directed 2026-08-09. Supersedes the registry layer
of [PR #128's design](2026-08-07-internal-policy-optimization-design.md) and
consumes its explicitly deferred "structured-record PR"
(that document, lines 572-586).

## The principle

**Anything that must be kept in sync is the defect.**

A checker is code and cannot drift from itself. A registry that *describes* the
checker can. Git holds the history exactly; a state log that *narrates* the
history can drift from it. The only thing neither the code nor git records is
*why* the code is the way it is — that is what a spec is for, and it is the one
derived artifact worth maintaining.

Applied:

| Concern | Single source | Retired duplicate |
|---|---|---|
| What is enforced | the checker in `scripts/` | `.agents/policy.csv` |
| Rules an agent must follow | `.agents/*.md`, indexed by `AGENTS.md` | the generated `Compact T0` block |
| What happened, when, by whom | `git log` | `.agents/state*` |
| Why the code is this way | `.agents/specs/<row>.md` | narrative state events |
| Current project state | matrices, roadmap, ledger | — |

## Measured problem

On `main` at `81291a89`:

- **~31k lines of governance code** — 15,847 in `scripts/check-*.py` +
  `agent-*.py`, 15,222 in `tests/scripts/`.
- **24 of 60 policy rules have no checker.** Their `enforcement` column names
  only `scripts/check-policy.py`, which validates the CSV's own shape. PR #128's
  own contract (line 147) requires `enforcement` to name "real checker
  entrypoints"; these do not.
- **All 60 rules already exist as prose** in `workflow.md`, `verification.md`,
  and `porting.md`, inside generated `<!-- policy-procedure -->` blocks. The CSV
  is a strict duplicate, and `check-protocol-consistency.py` (384 lines) plus
  its test (963 lines) exist only to keep the duplicate in sync.
- **~85% of red CI is bookkeeping**, not code. Over the last 20 failed runs:
  `check-doc-checkpoint.py` 16, `check-public-doc-tables.py` 11,
  `check-device-leakage.py` 5, `check-protocol-consistency.py` 4,
  `check-pr-size.py` 2, `check-role-discipline.py` 2, `check-agent-record.py` 1.
- **`.agents/` is ~14 MB live** — `state-events/` 3.5 MB across 158 events,
  `parity-ledger.md` 2.0 MB, `benchmark-record.md` 1.5 MB,
  `coordination.md` 524 KB. Nothing prunes any of it.

### Why the doc gate fails most

`classify_changed_paths()` classifies by *which directory was touched*. Any edit
under `src/`, `include/`, or `tests/` becomes `feature_checkpoint`, which demands
`docs/STATUS.md` + `docs/BENCHMARKS.md` + `.agents/NOW.md` in the same commit. A
one-line compile fix owes three public-doc edits.

Because the trigger is wrong, it has accreted six hardcoded exact-path-set
escape hatches — `POLICY_CONSOLIDATION_FILES`, `POLICY_CUTOVER_FILES`,
`PR_SIZE_BOOTSTRAP_FILES`, `PENDING_PR_RANGE_FILES`,
`SYNTHETIC_MERGE_RANGE_FILES`, `CLAIM_CUTOVER_FILES` — each a fossil of one
legitimate change the gate blocked, plus an inline 15-line "considered and
REJECTED" essay. Every new escape needs a checker edit, which itself trips
`POL-CHECKER-CHANGE`. The system fights its own repairs.

## Design

### 1. `AGENTS.md` holds all the policy

`AGENTS.md` is the only file every agent harness loads automatically. A rule
kept anywhere else is a rule an agent may never read — linked files are not part
of the instruction chain, as PR #128 itself observed (its lines 64-71). So the
rules go where they are guaranteed to be seen.

`AGENTS.md` becomes complete and self-contained: every obligation, in plain
prose, ordered by the phase of work it governs — session and claim, spec,
implement, verify, review, land, record. No rule IDs, no generated block, no
`Compact T0` to render or `--check`.

Budget: 16 KiB, raised from PR #128's 12 KiB because the file's job changed —
it was an index pointing at the rules, and it now *is* the rules. It lands at
about 13 KiB. No checker enforced the old budget (the prose in PR #128 asserted
it; `check-protocol-consistency.py`, which is deleted here, is where it would
have lived), so this is a documented target rather than a relaxed gate.

### 2. `.agents/*` files are activity guidance, not rules

They stop being rule registries and become the how-to an agent reads when it is
doing that specific activity. `AGENTS.md` indexes them, one line each:

- `porting.md` — porting from vLLM: the pin, upstream sync, test porting,
  shared seams.
- `verification.md` — gating and benchmarking: what a gate proves, how to
  measure honestly, the oracle protocol.
- `bugfixing.md` — reproduce, red test first, minimum fix, mutation-verify.
- `environment.md` — hosts, hardware, contention protocol, prohibitions.

None of them can create an obligation; if a sentence there is binding, it
belongs in `AGENTS.md` instead. That is what makes the two layers unable to
contradict each other: only one layer is normative.

**Nothing verifies that the prose and the checker agree** — that check is
precisely the cost being removed. Prose states the rule; the checker is the gate.

### 3. History is git

`.agents/state.md`, `state.csv`, `state-index/` and the state machinery are
deleted. The 158 existing events move verbatim to
`.agents/completed/state-events/` — moved, never deleted, satisfying the
evidence-preservation rule and PR #128's requirement to preserve every evidence
item before removing a legacy source.

An agent needing history uses git, and `AGENTS.md` states this as the method
with the exact commands:

| Question | Command |
|---|---|
| Did this row already land? | `git log --oneline --grep '<ROW-ID>'` |
| When did this symbol appear or change? | `git log -S'<symbol>' --oneline -- <path>` |
| What happened to this file, across renames? | `git log --follow --oneline -- <path>` |
| What is on main that I do not have? | `git log --oneline HEAD..origin/main` |
| Why is this line the way it is? | `git log -L '<start>,<end>:<path>'` |
| What did this commit actually change? | `git show --stat <sha>` |

Roadmap checking is the same move: the roadmap row states the *current*
lifecycle, and `git log --oneline --grep '<ROW-ID>'` plus the row's committed
spec state how it got there. Neither requires a narrative log, and unlike a
narrative log neither can disagree with the tree.

The facts a state event used to carry route to homes that already exist:
measurements to `benchmark-record.md` / `parity-ledger.md`; host prohibitions to
`environment.md`; the ephemeral handoff to `NOW.md` and the `coordination.md`
claim, both overwritten rather than appended.

### 4. Specs carry the why

A row reaching `DONE` must have `.agents/specs/<slug>.md` with an `## Outcome`
section: what was measured, what was rejected and why, and why a default is set
the way it is. `check-agent-record.py` asserts its presence. This is the only
obligation this change adds, and it exists because it is the one class of
knowledge neither the code nor git records.

### 4b. Roadmap rows link GitHub issues

The repository has real issues (`#201` ROCm hipblasGemmEx, `#199` macOS MLX
`-Werror`, `#193` A100 sm_80 GDN, `#192` C-linkage, `#170` GHCR images) that no
roadmap or matrix row references. Work is discoverable in two disconnected
places.

**The rule: no work without an open issue.** Before claiming a row or starting
implementation, the agent confirms an open GitHub issue tracks the work. If none
exists, it opens one. The issue number is then linked from three places that
must agree: the roadmap issue table, the row's spec, and the PR body.

`roadmap_v1.md` gains an **issue table** — a keyed, live map of issue → row →
state, maintained in place like every other keyed record:

```markdown
| Issue | Row | Title | State |
|---|---|---|---|
| #201 | `BACKEND-ROCM` | hipblasGemmEx overload mismatch | OPEN |
| #193 | `BACKEND-CUDA-SM80` | A100 crashes + wrong GDN output | OPEN |
```

Each roadmap and area-matrix row also gains an `Issue` cell, so the linkage
reads in both directions.

`check-agent-record.py` validates the form (`#<n>` or a full URL), that every
row in an `ACTIVE` or `DONE` state names an issue, and that the roadmap issue
table and the row cells agree. It does **not** query GitHub — the check stays
network-free, so it can never fail for connectivity, which is exactly the class
of flake this whole change is removing.

This makes the issue tracker the intake surface and the roadmap the ordering
surface, rather than two disconnected inventories.

### 5. The doc gate triggers on lifecycle, not on paths

`check-doc-checkpoint.py` asks "did a row change lifecycle state or gain an
accepted measurement", not "which directory did you touch".

- Editing `src/`, `include/`, `tests/` owes nothing.
- A row moving `READY` → `ACTIVE` → `DONE`, or gaining an accepted measurement,
  owes `STATUS` + `BENCHMARKS` + `NOW`.

All six escape-hatch path sets delete, because they were only needed to
compensate for the wrong trigger.

### 6. Records roll over on age

`scripts/roll-records.py` moves entries older than 30 days, or belonging to a
`DONE` row, into `.agents/completed/`, leaving an index line behind. It runs on
a schedule and is not a blocking gate — a cleanup task that reddens CI would
reproduce the problem this change exists to fix.

## What is deleted

| Path | Lines |
|---|---|
| `.agents/policy.csv` | 60 rules |
| `.agents/state.md`, `state.csv`, `state-index/` | — |
| `scripts/policy_contract.py` | 565 |
| `scripts/check-policy.py` | 35 |
| `scripts/check-protocol-consistency.py` | 384 |
| `scripts/state_record.py` | 1,239 |
| `scripts/check-state-record.py` | 29 |
| `scripts/migrate-state-record.py` | 1,136 |
| `tests/scripts/test_policy_contract.py` | 473 |
| `tests/scripts/test_policy_waivers.py` | 81 |
| `tests/scripts/test_check_protocol_consistency.py` | 963 |
| `tests/scripts/test_migrate_state_record.py` | 1,138 |
| `tests/scripts/test_state_record_core.py` | 594 |
| `tests/scripts/test_check_state_record.py` | 558 |
| `tests/scripts/test_state_record_cutover.py` | 266 |
| **Total** | **7,461** |

Dependents lose their `policy_contract` / `state_record` imports and inline the
constant they need: `check-doc-checkpoint.py`, `check-pr-size.py`,
`check-commit-trailers.py`, `check-prompt-contract.py`, `check-now-current.py`,
`agent-preflight.sh`, `.github/workflows/ci.yml`.

The 36 real checkers are untouched. Nothing that mechanically protects
correctness, parity, or the ABI surface is weakened.

## The rules do not change — only how they are posed

This is a change of form, not of obligation. Every rule survives; it stops being
a CSV row with an ID and becomes a readable sentence in the file that owns its
domain. In particular these are preserved exactly, because they are the method:

- **Spec first.** No row enters `READY` or `ACTIVE` without a committed
  `.agents/specs/<slug>.md`. The spec is committed before implementation, not
  written up afterwards.
- **Roadmap rows.** `roadmap_v1.md` and the area matrices keep their stable row
  IDs, lifecycle states, and links to the owning spec and evidence.
- **Sub-agents, not self-service.** The operator delegates implementation to a
  fresh implementer, dispatches a *different* fresh reviewer that both inspects
  statically and mutates the claimed guarantees, returns findings to a fresh
  implementer without repairing them in the coordinating session, and repeats
  until PASS.
- **Operator verifies.** The operator reruns the row's gate itself. An
  implementer or reviewer report is an input, never a gate result.
- **PR discipline.** Feature code reaches `main` through a reviewed `row/*` PR.
- **One surface.** A capability is `DONE` only when `include/vllm.h` exposes it
  and examples are thin clients of it.
- **Mirror vLLM.** Upstream defines behavior; port its tests in the same change.
- **Correctness before performance**, on the pinned oracle, on every axis, with
  a reproducible recipe, and no declared ceilings.

Three mechanisms change form because their carrier is gone:

- **Checker changes** still require red-before / green-after mutation evidence
  and a spec. They no longer require naming affected rule IDs, because there are
  no IDs. The substance — you may not turn a red gate green by deleting the
  assertion — is unchanged.
- **Waivers** stay in `.agents/waivers.csv`, keyed to the *checker* rather than
  to a rule ID. Still exactly scoped, owned, evidenced, and expiring.
- **Prose and checkers may drift.** Accepted deliberately: only the checker
  blocks, so drift costs clarity, never correctness. Eliminating that drift is
  what cost 1,347 lines of sync checking.

## Risk accepted

PR #128 documented six concrete contradictions in the pre-CSV prose era
(its lines 43-62: boot order, onboarding, state-entry obligation, claim tables,
README triggers, the token-identity gate). Removing the registry re-admits the
possibility of contradictory prose.

Mitigated by structure rather than by a sync checker: one domain per file with
no overlap, no second copy of any rule, and `AGENTS.md` reduced to an index so
it cannot become a competing authority. The 2026-08-07 contradictions were all
between *two copies* of the same rule. This design keeps one copy.

## Landing

Direct to `main`, user-directed. `check-role-discipline.py` classifies
`.agents/`, `scripts/`, and `AGENTS.md` as integration paths that are
deliberately pushable without a PR; `pr-size` is `pull_request`-only. Every
commit carries the `FOLLOWING_AGENTS_PROTOCOL` trailer.

Order, each green before the next:

1. This spec.
2. History is git — retire the state record.
3. The doc gate triggers on lifecycle.
4. `policy.csv` retires into the `.agents/*` topic files; `AGENTS.md` becomes an
   index.
5. Records roll over on age.
