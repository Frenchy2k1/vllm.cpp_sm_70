# The operator lock becomes a RECORD of who is coordinating where

User-directed 2026-08-10, issue
[#285](https://github.com/mudler/vllm.cpp/issues/285). Row `ENG-OPERATOR-RECORD`
(tooling and policy; it owns no claim-matrix row and no product code).

The developer's words: *"I dont want push force main, never. the operator have
at max the way to merge directly PRs and heavily dispatch sub-agents with
separate worktrees ( == coordinator )"*, and, on the lock itself: *"let's keep
it as a record for who is working where"*.

## Scope

`scripts/agent-role.py` stops refusing a second operator. The file in the git
common dir stays, and stops being a lock: it becomes a set of records, one per
worktree, that says who is coordinating where. `show` reports the other live
coordinators. `release` removes only the caller's record. The 2-hour TTL and
stale pruning stay exactly as they are.

Out of scope: the helper role, the role marker, the `read-only` answer,
`--headless`, `check-role-discipline.py`'s row/PR rule, and anything about how
work lands. Nothing here weakens a gate: no refusal is added, and the one
refusal removed is the subject of the issue.

## Our baseline — why the exclusivity no longer holds

`scripts/agent-role.py:28-37` justifies a repo-wide exclusive lock with *"the
shared case is the operator's primary checkout, where one role is the correct
answer anyway."* That premise is gone. `AGENTS.md` § "Work happens in a
worktree" now requires **every** unit of work to take its own linked worktree,
and § "Landing work" requires everything to reach `main` from a task branch. So
there is no shared checkout to protect and no unsynchronised writer to exclude.

What an operator actually is, per the developer: a **coordinator**. Its maximum
powers are merging PRs directly and dispatching sub-agents into separate
worktrees. It never rewrites shared history — `main` is never force-pushed, with
no `--force` and no `--force-with-lease` — so a plain `git push` refuses any
non-fast-forward and **git itself is the interlock**. Two coordinators racing to
land serialise on that refusal: the loser fetches, re-merges, re-gates and
pushes again. A JSON file in `.git/` never provided that guarantee and cannot.

What the exclusivity costs is measured, not hypothetical.
`LOCK_TTL_SECONDS = 2 * 60 * 60`, so a session killed mid-flight — one was on
2026-08-10, by a host disk cleanup, leaving a dead pid and a frozen heartbeat —
blocks **all** coordination for up to two hours, and the only remedy is
hand-deleting a file inside `.git/`. A claim refused at 78 minutes is the TTL
working correctly; that it was refused at all is the defect.

## Design

**Representation.** One file per worktree in a directory,
`<git-common-dir>/vllm-cpp-operators/<sha256(worktree)[:16]>.json`, instead of
one shared `vllm-cpp-operator.lock`. Still in the git common dir, so it is
shared by every worktree and can never be committed.

That shape is what makes concurrency safe: **a writer only ever touches its own
record**, because the filename is derived from the identity ownership already
keys on. Two coordinators claiming at the same instant write two different
paths, so neither can lose the other's record — there is no read-modify-write of
a shared file anywhere in the design. Each write is `write temp + os.replace`
inside the same directory, which is atomic on POSIX, so a reader sees the old
record or the new one and never a half-written one. A single JSON array or a
JSONL log would both have required rewriting a shared file to release or prune,
which is exactly where a concurrent writer's record gets dropped.

`O_CREAT|O_EXCL` goes: it existed to make the second claimant fail, and the
second claimant must now succeed.

**Ownership still keys on the worktree** (`git rev-parse --absolute-git-dir`),
unchanged from the 2026-08-06 correction, and `record_is_ours` keeps the legacy
session fallback for a record written before that correction.

**Staleness.** `RECORD_TTL_SECONDS` keeps the 2-hour value and the heartbeat
semantics. Another worktree's stale record is filtered out of every display and
unlinked on the next `claim`, which is a write path; `show` and `resolve` never
unlink, because `agent-preflight.sh` documents itself as never writing anything.
A stale record can no longer refuse anybody, so breaking one is no longer an
event: the NOTE that announced it goes with the refusal it explained.

The prune skips THIS worktree's own record (`keep_canonical`, third review
round). Our own record is stale on any ordinary re-claim — the TTL is two hours
and a session re-claims at the top of its next tool call — and pruning it before
republishing it is the same unlink-then-create the second round removed from
`drop_our_record`, only reached by the commoner path. It costs nothing to skip:
`write_our_record` republishes that exact path immediately, and `resolve` matches
our own record by ownership with no staleness filter, so an aged own record was
never displayed as a live coordinator in the first place. The consequence of not
skipping is measured, not theoretical: a session killed inside the window turned
a worktree that resolved `role=operator` into `role=UNDECLARED` exit 3.

**Migration.** A pre-#285 `vllm-cpp-operator.lock` is read as one more record,
so a session that claimed operator before this change still resolves as operator
instead of silently becoming UNDECLARED mid-flight. Its next `claim` or
`release` removes it, which heals the repo the first time either runs.

**The front doors stop blocking.** `scripts/agent-onboard.py` and
`scripts/agent-start.py` told a session that `claim operator` would fail and
instructed it not to run "a known-failing claim". That claim no longer fails, so
`blocked_by_other_operator` is replaced by `operator_peers` — the live records
that are not this worktree's — and both surfaces report them as information
beside the ordinary claim command.

## Port map

None. This is repository tooling with no upstream vLLM counterpart; the porting
inventory's §9 (written from scratch) is where role machinery has always sat.

## Upstream chain

Not applicable — vLLM has no agent-role protocol. The authority for this change
is the developer's direction in issue #285.

## Tests to port

None to port. `tests/scripts/test_agent_role.py` grows the new behaviour, and
the tests that pinned the removed refusal are rewritten to pin its replacement
rather than deleted:

| Was | Becomes |
|---|---|
| `test_second_operator_is_refused` | `test_a_second_coordinator_is_recorded_not_refused` |
| `test_one_operator_per_repo_holds_across_worktrees` | `test_show_lists_the_other_live_coordinators` |
| `test_stale_lock_is_broken_but_reported` | `test_a_stale_record_is_pruned_and_never_blocks` |
| `test_a_legacy_lock_cannot_produce_two_operators` | `test_a_legacy_lock_file_is_adopted_as_this_worktrees_record` |
| `test_an_operator_marker_beaten_to_the_lock_reports_the_lockout` | `test_an_operator_whose_record_vanished_is_told_to_re_claim` |

New: concurrent claims from many worktrees lose no record; `release` removes
only the caller's; `show` reports worktree, session, host and heartbeat age for
each peer.

## Gates

- `python3 tests/scripts/test_agent_role.py` (focused, RED first)
- `python3 tests/scripts/test_agent_onboard.py`, `test_agent_start.py`
- `scripts/agent-preflight.sh --quiet` green before and after the commit
- Mutation: every guard the new tests claim to pin is deleted or inverted, the
  focused suite is shown RED, and the guard is restored and shown GREEN.

## Dependencies

None. Python, docs and tests only; no GPU, no build, no network.

## Work breakdown

Single unit — the tool, its suite, the two front doors that consumed the
refusal, and the documents that asserted it. Splitting it would leave the repo
in a state where the tool permits a second coordinator and the docs still forbid
one.

## Risks/decisions

- **Risk: a directory of files is harder to inspect than one file.** Accepted;
  `show` renders it, which is the point of keeping records at all.
- **Risk: nothing now prevents two coordinators merging to `main` at once.**
  That is the decision, not an oversight: git's non-fast-forward refusal is the
  real interlock and the lock never was one. It holds only while `main` is never
  force-pushed, which `AGENTS.md` now states as a rule.
- **Decision: keep the TTL.** It is correct and already works. Staleness now
  only prunes a display entry instead of gating a claim.
- **Decision: no `--force` variant is ever added to any script here.**

## Outcome

Landed 2026-08-10 on `row/ENG-OPERATOR-RECORD`. `claim operator` records and
succeeds alongside live peers; `show` lists every other live coordinator with
worktree, session, host and heartbeat age; `release` removes only the caller's
record; a stale record is pruned from the display and blocks nothing. The
refusal path and its `already held` message are gone from the tool, the two
front doors, and the documents.

Three review rounds. Round 2 scoped `drop_our_record` so a re-claim replaces its
record instead of unlinking it, and pinned the atomic publish. Round 3 found the
same window still open through `prune_stale_records`, which the round-2 comment
had asserted was closed: the prune runs first and unlinks any stale record,
including ours. It is now scoped the same way, and
`test_a_reclaim_never_unlinks_its_own_record` runs both ages of record because
only the fresh one had been covered. Round 3 also pinned what the publish tests
missed — a publish that unlinks the name and recreates it leaves the hardlink
witness and the residue check both green, so the NAME is now watched directly —
and pinned the `legacy` field a pre-#285 peer adds to `--probe --json`.

Still declined, unchanged: `prune_stale_records` targets a path rather than the
inode it read, so a PEER that republishes inside the window loses that record.
Its remedy is `claim operator`, which is never refused. With `keep_canonical`
that declination is now confined to peers; this worktree's own record is no
longer exposed to it.
