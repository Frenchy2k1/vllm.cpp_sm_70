# Task guide — coordinating parallel work

How the operator runs several rows at once without agents colliding. The rules
are in [`AGENTS.md`](../AGENTS.md); this is the method.

## The shape of a campaign

The operator holds the plan, the GPU, and main. Everything else is delegated to
fresh agents with bounded briefs. The operator does not implement work that
should be independently reviewed — writing it and reviewing it in one context
defeats the review.

For each row: confirm the issue, commit the spec, dispatch an implementer,
dispatch a *different* reviewer, return findings to a new implementer, rerun the
gate yourself, then merge or close.

## Isolation

Parallel agents get their own worktree. Never let two agents write the same
tree.

Stage explicit paths. `git add -A` mid-write picks up another agent's
half-finished work, and the resulting commit is very hard to unpick afterwards.

Pin the base SHA when you create the worktree and push with a lease. A landing
built on `origin/main` resolved at push time can silently absorb — and revert —
another session's merge.

Brief every dispatched agent to wait in the foreground and return once.
Backgrounded builds surface as a completed agent with no result, which reads
exactly like success.

## Claims

`.agents/coordination.md` holds who is doing what *now*. It is overwritten, not
appended. The canonical hierarchy it defines is one-way: the roadmap orders big
areas, area matrices hold the exhaustive row inventory, specs hold the analysis,
this file holds the live claim, and the matrices plus the parity ledger hold the
evidence. Detailed status prose never flows upward.

If the remote cannot be queried, the claim state is `REMOTE_UNVERIFIED`. Unknown
is not absence, and it authorizes no cleanup.

## Handing off unfinished work

Refresh `.agents/NOW.md` — live claims, current gate, next actions, stamp — and
update the coordination entry. Put the durable findings in the row's spec
`## Outcome`.

Record the immutable head, the exact gates run, the evidence roots, any
prohibitions, the blocker, and the first resume command. The next session should
need `NOW.md`, the row's spec, and `git log` — nothing else.

Routine discussion and git housekeeping are not checkpoints and do not need a
record.

## Landing

Land squashes or a local `--no-ff` merge commit carrying the trailers. GitHub
authors its own message for `gh pr merge --merge`, which loses them.

Verified PRs merge in-session. Obsolete PRs close with the reason recorded.
Never end a session with a verified, unmerged PR.
