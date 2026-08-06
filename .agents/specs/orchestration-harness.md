# The orchestration harness — how an operator runs a row

User-directed 2026-08-06. Status: **accepted design, not yet implemented.**
This document is the contract; the prose and gates named in § Enforcement are
the work it implies.

Subsystem **B** of two. Subsystem A —
[session onboarding](session-onboarding.md) — landed the entry point: a session
declares operator, helper or read-only, and declares interactive or headless.
This spec is what an **operator** does next.

## Scope

The loop an operator follows to take a row from `READY` to a merged PR:
decompose it, dispatch implementer sub-agents, run the row's gates, have an
**independent reviewer sub-agent** attack the result, and iterate. Plus the two
disciplines that make the loop trustworthy: a gate command that can actually
fail, and a review that mutates rather than reads.

Out of scope: what work to do (the roadmap), the correctness and performance
directives (`AGENTS.md`, `.agents/directives.md`), and the session interview
(subsystem A).

## Our baseline — we already have almost all of this, under other names

`~/_git/skills/spec-driven-development` is a working headless pipeline: spec →
plan → serial sub-agent implementation under a TDD gate → reviewer sub-agent →
pushed branch. Reading it against this repo, the striking thing is how little is
missing. Nearly every concept already exists here with a different name:

| spec-driven-development | vllm.cpp equivalent | Status |
|---|---|---|
| spec → `docs/spec/…` | **spike spec** `.agents/specs/<slug>.md`, 9 required sections | exists, CI-gated by the spike gate |
| plan task carrying `{id, deps}` | **row** with a stable ID and dependencies, in an area matrix | exists |
| **Verify** command, must exit nonzero | the row's **Gates** field — "exact commands" | exists, **unenforced** |
| worktree under `~/.cache/sdd/…` | helper worktree on `row/<ROW-ID>` | exists; `agent-role.py` materializes it |
| `.sdd-state.json` | `.agents/state.md` plus the claim | exists |
| `pending / done / blocked` | `INVENTORIED / SPIKE / READY / ACTIVE / GATING / PARTIAL / DONE / BLOCKED` | exists |
| PR at the end, never merge | the draft PR **is** the claim; the operator merges | exists |
| implementer sub-agent | — | **missing** |
| **reviewer sub-agent** | — | **missing** |

So B is not a new system. It is three additions to one that is already here.

## The evidence that decides the design

This spec is not written from taste. Two branches were executed through exactly
this loop — the P0 live-state audit and subsystem A — and they produced a
measurement.

**Ten times, a test passed with the thing it named deleted.** Not ten sloppy
tests; ten tests that looked correct to every reader, including the ones that
wrote them. A sample:

- `assertIn("REQUIRE_ROLE=1", text)` — satisfied by an unrelated line elsewhere
  in the same file, so flipping the default the gate existed to set left the
  whole suite green.
- Five tests calling `cmd_env_set()` directly and never `main()`, so deleting
  the CLI flag left them green while the only command an agent types did
  nothing.
- `assertIn("merged", reason)` — `"unmerged"` contains `"merged"`, so it passed
  for both verdicts.
- `assertNotIn("PARTIAL", CHECK_FAILS_ON)` — passes for an empty set, i.e. for a
  gate that fails on nothing at all.
- A test asserting a filename appears in a path list, which passed when that
  file contributed zero rows.
- A row id asserted with `assertIn`, satisfied by the fixture's own queue.

**Not one of these was found by reading the diff.** Every one was found by a
reviewer that mutated the code and re-ran the suite. And roughly half originated
in the *plan text* — written by the same agent that reviewed its own work and
called it good.

That is the whole argument for a separate reviewer, and it also says what kind
of reviewer: **the value is mutation, not diff-reading.** A reviewer that reads
a diff and comments on style would have caught none of the ten.

A second measurement, same two branches: **every single Important finding was
produced by an independent sub-agent, and none by the implementer's own
self-review.** Self-review reliably caught typos and never caught a defect the
author had reasoned themselves into.

## Design

### The operator's loop

The operator does not write the feature. It decomposes, dispatches, verifies,
and integrates.

```
row (READY, spike merged)
  └─ decompose into tasks, each with a Gate command that exits nonzero on failure
      └─ for each task, serially:
          ├─ dispatch implementer sub-agent (fresh, TDD, commits in the worktree)
          ├─ RUN THE GATE YOURSELF — never accept the implementer's word
          ├─ dispatch reviewer sub-agent (fresh, never the one that wrote it)
          │    └─ MUTATE, don't read: for each test, delete what it names and re-run
          ├─ findings? → fix round (bounded), then a SCOPED re-review
          └─ clean? → next task
      └─ draft PR (already open — it IS the claim) → operator merges
```

**Serial, one task at a time.** Two implementers in one worktree means
concurrent edits to one checkout. Parallelism comes from multiple helpers in
multiple worktrees, which the role model already provides.

**The operator runs the gate itself.** This is the one failure mode nothing else
catches: if "done" is the implementer's opinion of its own work, the loop has no
floor.

### The reviewer sub-agent, and what makes it different

A reviewer is dispatched **fresh** for every task, is **never** the agent that
wrote the code, and is told to attack rather than assess.

Its binding instruction is **mutation over reading**:

> For each test in the change, delete or invert the line it names and re-run the
> suite. A test that stays green is a finding, regardless of how it reads.

Two supporting rules, both learned the same way:

- **Do not trust the report.** A stated rationale — "kept it simple
  deliberately", "left it per YAGNI" — is the implementer grading its own work
  and never downgrades a finding's severity. On these two branches, three
  implementer reports contained a claim that was false and disclosed as true;
  each was caught by a reviewer reproducing it rather than accepting it.
- **A finding the plan mandated is still a finding.** Roughly half of all
  Important findings were defects in the plan text. A reviewer that treats the
  plan as authority cannot find them; it must report them, labelled, and the
  human decides.

### Gate-command discipline

A row's `Gates` field already promises "exact commands". Nothing checks that a
gate command can **fail**. A gate that is `true`, `echo ok`, or a command whose
exit status is masked by a pipe collapses "done" into an opinion.

Three rules, each of which this project has already been bitten by:

1. **A gate command must exit nonzero on failure.** A task you cannot write one
   for is a task that cannot be run through this loop. Split it, restate it, or
   narrow its deliverable until a real command can judge it.
2. **Never pipe a gate.** `cmd | tail` reports the exit status of `tail`.
   Redirect to a file and check `$?`.
3. **Verify the committed form, not the staged one.** `check-doc-checkpoint.py`
   runs `--staged` in preflight, which passes vacuously once work is committed.
   Eleven commits on the P0 branch were red while every preflight was green.

### Headless mode

Subsystem A made mode a declaration: interactive by default, headless only when
stated. This loop honours it.

| | interactive | headless |
|---|---|---|
| ambiguity | ask | decide, record in `.agents/state.md`, continue |
| a task that will not go green | ask | park it, skip its dependents, carry on |
| landing | operator merges | never merge, never delete the worktree; push and report |

Headless never asks — a question to an absent human is a hang, not a pause — and
therefore every decision it makes must appear in the final report. Interactive
is the default precisely because the judgment calls this loop surfaces
(a state transition on the canonical record, a benchmark that moves a binding
number) are the human's to make.

### What the loop must never do

- **Never let the reviewer fix what it found.** Findings go back to a fresh
  implementer. A reviewer that edits has reviewed its own work.
- **Never fix findings in the operator session.** It pollutes the context that
  exists to coordinate, and controller fixes skip review entirely.
- **Never mark a task done without having seen its gate exit 0** with your own
  eyes.
- **Never weaken a gate to make a transition pass.** Repair the record.

## Enforcement

**Prose.** The loop lives in `.agents/workflow.md`, next to subsystem A's
interview, because that is what an agent reads. `AGENTS.md`'s operator bullet
points at it.

**`scripts/check-gate-commands.py`** (new, CI-gated, with a mutation suite):
every row at `READY` or later carries at least one gate command; no gate command
is `true`, `:`, `echo …`, or piped into another command. This is checkable from
the tree and needs no network.

**The reviewer prompt is a tracked artifact**, not folklore:
`.agents/prompts/reviewer.md`, carrying the mutation instruction verbatim. A
prompt that lives only in an operator's head is not a protocol.

**`scripts/check-protocol-consistency.py`** extends to assert the loop appears
in `.agents/workflow.md`, exactly as it now asserts the role interview. Prose
and gate move in the same change — that checker exists because an obligation was
once migrated in `AGENTS.md` and the checker but not in the manual.

## Work breakdown

| # | Work |
|---|---|
| 1 | `.agents/prompts/reviewer.md` and `.agents/prompts/implementer.md` as tracked artifacts |
| 2 | `scripts/check-gate-commands.py` + mutation suite; wire into preflight and CI |
| 3 | The loop written into `.agents/workflow.md`, with `check-protocol-consistency.py` extended in the same change |
| 4 | `AGENTS.md` operator bullet points at the loop; `operator-helper-protocol.md` records that the operator drives work through sub-agents |
| 5 | Backfill gate commands for the rows that lack one, or record honestly that they cannot be gated yet |

Item 5 will find rows that cannot state a failing gate command. That is a
finding, not an obstacle: those rows cannot be run through this loop until they
are narrowed, and saying so is more useful than a `true` that lets them pass.

## Risks and decisions

**Accepted: this makes every task slower.** Two branches of evidence say the
review loop roughly doubles the cost of a task and catches defects that reading
does not. The alternative is not a faster loop; it is the same loop with the
findings still in the tree.

**Accepted: the reviewer is another agent, with the same blind spots.** It is
not smarter than the implementer — it is *differently positioned*, and it
mutates. The mutation instruction is what makes independence pay; without it a
reviewer converges on the implementer's reasoning, which is exactly how the ten
tests survived their authors.

**Rejected: let the implementer self-review instead.** Measured across two
branches: self-review caught typos and never caught a defect the author had
reasoned themselves into. Three implementer reports asserted something false in
good faith.

**Rejected: parallel implementers in one worktree.** Concurrent edits to one
checkout, with nobody to untangle the result. Parallelism belongs at the helper
level, where the role model already isolates it.

**Rejected: adopting `spec-driven-development` wholesale.** It is headless by
construction and forbids asking anything. That is right for an unattended
overnight run and wrong as this repo's default, where the loop routinely
surfaces decisions about the canonical record that belong to a human. We take
its structure and keep our interaction model.

**Open:** the P0 branch's gate now cannot re-detect the rows it vacated, because
its own commit messages name them and the evidence rule is a commit-message
mention with no code-touch filter. The same class of question applies to any
gate this loop introduces: *what does this gate stop being able to see once it
has run once?* Worth asking of `check-gate-commands.py` before it lands.
