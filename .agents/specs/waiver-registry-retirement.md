# Retire the waiver registry; a commit documents its own exception

Issue: [#281](https://github.com/mudler/vllm.cpp/issues/281)
Row: `POLICY-WAIVER-RETIRE`

## Why

`AGENTS.md` opens with **"There is no state log. Git is the history, and it
cannot disagree with the tree."** `.agents/waivers.csv` is a state log. It holds
facts about changes somewhere other than the changes themselves, needs its own
expiry bookkeeping, and can drift from the tree without anything noticing.

The project already made this exact move once: `0f3e44ee` deleted `policy.csv`
and its checker, leaving `AGENTS.md` as the whole policy. This finishes the same
thought for exceptions.

Developer decision, 2026-08-10: exceptions are enforced through git. A commit
that needs one argues for it in its own message, where the reason is attached to
the diff it excuses, carries an author and a date, and cannot be edited later
without rewriting history.

## What is actually there (measured before touching anything)

The registry is already vestigial, which is why this is a deletion rather than a
migration.

`check-pr-size.py` takes `waivers` and `waiver_scope` parameters and **never
reads either one**. They are threaded from `main()` into `change_errors()` and
dropped. That is left over from the same-day retirement of the per-class line
budgets (`check-pr-size.py:50-54`): once there was no budget, there was nothing
to waive, but the plumbing stayed.

So every entry in the file is inert:

| Waiver | Scope | Target checker | PR state | Can it fire? |
|---|---|---|---|---|
| `WAIVER-PR-SIZE-001` | `pr:128` | `check-pr-size.py` | MERGED | No — checker ignores waivers |
| `WAIVER-PR-SIZE-002` | `pr:166` | `check-pr-size.py` | MERGED | No — same |
| `WAIVER-PR-SIZE-003` | `pr:196` | `check-pr-size.py` | MERGED | No — same |

The only live consumer in the tree is one line:

    scripts/check-commit-trailers.py:264
        if exact_waiver(waivers, CHECKER, f"commit:{commit}") is not None:

and no `commit:` waiver has ever been written to the file.

**Consequence: removing the registry changes no gate outcome today.** That is the
whole risk assessment, and it is why this can ship as a straight deletion.

## Design

No replacement escape hatch, deliberately.

For the trailer contract there is nothing sensible to waive: a commit either
carries `FOLLOWING_AGENTS_PROTOCOL` and its trailers or it does not, and a commit
waiving its own trailer requirement via a trailer is circular. Historical commits
that predate the contract are already handled by the `--cutover` mechanism, which
is git-native — it names a commit, and ancestry decides.

If a future gate genuinely needs an exception, the commit message states the case
and a human merges it or does not. That is the git registry: `git log --grep`
finds every exception ever taken, with its reason and author, and the record
cannot disagree with the tree because it *is* the tree.

## Scope

Delete: `.agents/waivers.csv`, `scripts/waivers.py`, `tests/scripts/test_waivers.py`,
`tests/scripts/test_policy_waivers.py`.

Strip waiver plumbing from: `check-commit-trailers.py` (import, `validate_waiver_targets`,
the suppression at 264), `check-pr-size.py` (import, dead params, the
`.agents/waivers.csv` path entry, the stub text), `agent-integration.py`,
`agent-preflight.sh`, `.github/workflows/ci.yml`.

Prose: the `AGENTS.md` waiver paragraph under "Changing the rules or a checker",
and `.agents/prompts/implementer.md:27` (`deviations_and_waivers`).

## Gates

`AGENTS.md` requires a spec plus red-before/green-after for a checker semantics
change, and forbids turning a red gate green by deleting an assertion. This
deletes an assertion-*suppressor*, which moves in the safe direction: strictly
more can fail after than before, never less.

- **Red before**: with a `commit:<sha>` waiver present, `check-commit-trailers.py`
  reports OK on a commit with a malformed trailer block. That is the behaviour
  being removed, and it is demonstrated first so the removal is evidenced rather
  than asserted.
- **Green after**: the same malformed commit is reported, with the waiver file
  still present and then absent. A stale registry must not be able to suppress
  anything, including by accident.
- No checker may start passing something it previously failed. Full preflight
  green.

## Stop conditions

If any consumer turns out to read a waiver in a way that changes a live gate
result, stop and report instead of deleting: that would make this a migration
with a real replacement mechanism, not a retirement.
