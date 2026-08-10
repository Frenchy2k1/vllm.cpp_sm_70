# Every unit of work happens in its own worktree (2026-08-09)

**Kind:** governance + checker-semantics change · **Developer-directed**
(2026-08-09). **Affects:** `AGENTS.md` "Landing work",
`scripts/check-role-discipline.py`, `scripts/check-pr-size.py` evidence map,
new `tests/scripts/test_check_role_discipline.py`.

## Gap

Nothing in `AGENTS.md` requires that work happen off the shared checkout. The
word "worktree" does not appear in it. `.agents/workflow.md` mentions worktrees,
but only as campaign method for *parallel* agents, and a task guide can never
add a rule.

Worse, the rule that does exist points the other way. "Landing work" states:

> Integration paths — `scripts/`, `.agents/`, `docs/`, `.github/`, `AGENTS.md`
> — may be pushed directly so a gate or record can be repaired without a round
> trip.

`check-role-discipline.py:184` implements exactly that: `policy_commit_violations`
filters governed paths through `not is_integration_path(path)`, so a policy,
docs, script, CI, or `AGENTS.md` change may be committed straight onto the
shared checkout and every gate passes.

Two consequences, both observed on this machine on 2026-08-09:

1. The shared checkout drifts. It sat on branch `pr178`, 40 commits behind
   `origin/main`, with four stray PNGs and a nested `.review-w5/` worktree
   untracked in it — so it was neither current nor safe to branch from.
2. Worktrees are created and never removed. ~190 were registered and ~150 live
   under `.claude/worktrees`, at roughly 200 MB of checkout each, filling the
   root filesystem to 100% (23 MB free of 447 GB) and blocking every gate that
   needs a temp file. A creation rule with no teardown rule produces this.

## Design

**Rule (AGENTS.md).** A new "Work happens in a worktree" section: every unit of
work gets its own linked worktree on its own task branch (`row/<ID>` for a
claimed row); the shared checkout stays on `main`, clean, and is never a work
surface; the worktree and its branch are removed once the work merges, closes,
or is abandoned.

**Rule (AGENTS.md, revised).** "Landing work" drops the direct-push allowance.
Work reaches `main` from its task branch through a reviewed PR, or through a
local merge commit naming that branch when the operator holds recorded merge
authority. The repair-without-a-round-trip case is still served: an authorized
operator merges their own branch locally in one step, without opening a PR.

**Checker.** `policy_commit_violations` gains `govern_integration`. When set,
the `is_integration_path` exemption does not apply and every tracked path must
arrive via a task branch. `is_integration_path` is retained, because pre-cutover
history was created under the direct-push rule and must stay green.

**Activation.** `WORKTREE_DISCIPLINE_SINCE` mirrors `ROLE_DISCIPLINE_SINCE`:
`None` means report-only; set to the cutover SHA it enforces from that commit
onward. A commit cannot contain its own SHA, so the landing commit sets it to
the SHA of the commit that introduced the behavior, both on this branch. This is
the same pattern and the same reasoning as the 2026-08-05 cutover — rewriting
past judgement retroactively would redden honest history.

## Risk

Tightening arrival for integration paths would fail every historical direct
push if applied retroactively. The cutover constant is what prevents that; the
red-before test asserts pre-cutover behavior is unchanged.

## Tests

New `tests/scripts/test_check_role_discipline.py`, the file this checker's
evidence override in `check-pr-size.py` will point at (today it points at
`test_check_pr_size.py`):

- **Red before:** an integration-path commit with no branch reference is
  governed under `govern_integration=True`. Fails on current `main`, where the
  parameter does not exist and the exemption is unconditional.
- Legacy path: same commit under `govern_integration=False` stays exempt.
- Feature paths are governed in both modes.
- `arrives_via_row_pr` still accepts a `row/*` merge commit, a squash carrying
  `(#N)`, and a synthetic PR merge whose second parent names the row.
- Traversal-ish and malformed paths stay classified as feature paths.

## Gates

`python3 -m pytest tests/scripts/test_check_role_discipline.py
tests/scripts/test_check_pr_size.py -q`, then
`python3 scripts/check-role-discipline.py --base origin/main --head HEAD`, then
`scripts/agent-preflight.sh --staged`.

## Stop conditions

If enforcing arrival for integration paths would redden any commit already on
`main`, stop and report rather than widening the exemption or moving the
cutover backwards.

## Outcome

Pending.
