# The trailer gate rejects correct commits because of paragraph placement

Issue: [#406](https://github.com/mudler/vllm.cpp/issues/406).
Row: `ENG-TRAILER-MERGE-ARTIFACTS`. Related: [#274](https://github.com/mudler/vllm.cpp/issues/274).

`main` is red on the `agent-record` job. The trailer gate is failing on **how
commits land**, not on how they are written, and one of its failure modes rejects
a commit that is entirely correct.

## Scope

**In scope.** How `scripts/check-commit-trailers.py` locates the trailer block.

**Out of scope, deliberately.** The uniqueness rule, the forbidden-AI-trailer
rule, and every other assertion the checker makes. No rule is relaxed; only the
*location* of the block changes. No product source is touched.

## Upstream chain

None. vLLM has no counterpart to this protocol machinery, so the mirror rule does
not apply and there is no upstream `file:line` to port from. Governed by
`AGENTS.md` §"Changing the rules or a checker", which requires a spec, a
red-before test or mutation, and green-after evidence, and forbids turning a red
gate green by deleting an assertion or widening a scope.

## Our baseline — five failures, only one of which is the gate's fault

`check-commit-trailers.py` reads trailers through `git interpret-trailers
--parse`, which by design treats **only the final paragraph** as the trailer
block. Measured on `main`:

```console
$ git show -s --format=%B dbd0d51c | git interpret-trailers --parse
Co-authored-by: Ettore Di Giacinto <mudler@localai.io>
```

That commit carries a complete, correct trailer block. Git cannot see it, because
GitHub appended `Co-authored-by:` as a separate trailing paragraph. The checker
counts zero and reports the trailers missing.

**13 of the last 30 commits on `main` fail this check.** They did not red CI at
the time only because those runs were cancelled (#274) — the cancellation hid the
defect rather than causing it.

The five distinct shapes, each verified against the real commit:

| Commit | Shape | Verdict |
|---|---|---|
| `dbd0d51c` | human `Co-authored-by:` appended below the block | **gate defect** — correct commit rejected |
| `f64f2b71` | bot `Co-authored-by:` appended below the block | **real violation**, previously hidden by the parse |
| `87308dea` | GitHub's `---------` separator between block and co-author | malformed message |
| `b8293c88` | multi-commit squash doubled the whole block | malformed message |
| `b580452d` | "Merge pull request #N", no trailers at all | real violation |

Only the first is the gate's fault. That is the one this row fixes.

## Port map

| Item | Local anchor | Motion |
|---|---|---|
| `join_trailing_trailer_paragraphs`, `_is_trailer_paragraph` | `scripts/check-commit-trailers.py` | new |
| `parsed_trailers` | same file | fuse before parsing |
| uniqueness, marker placement, forbidden-AI-trailer rules | same file | **unchanged** |

## Design

**Fuse consecutive trailing trailer-shaped paragraphs before parsing.** A
paragraph qualifies only if every line is trailer-shaped (`Key: value`, or an
indented continuation). A prose paragraph still terminates the block, so trailers
buried mid-message remain invalid — the looseness this gate exists to prevent is
untouched.

Nothing is relaxed. The block must still exist, the marker must still sit above
it, each declaration must still appear exactly once, and an AI co-author is still
forbidden. The change is that the block is *found* where a merge tool actually
left it.

**What was rejected, and why it matters.** The first attempt also collapsed
identical duplicate trailers, to fix the `b8293c88` squash case. That is a
relaxation of the uniqueness rule, and an existing test
(`test_protocol_and_ai_declarations_are_unique_and_exact`) pins it. Rewriting an
assertion to suit the change is exactly what `AGENTS.md` forbids, and the
distinction is real: a doubled block is a genuinely malformed message, fixable at
source by writing the squash body or landing a single-commit PR, whereas the
`Co-authored-by` case is a correct commit defeated by the parser. It was reverted
in full; `b8293c88` stays red on purpose.

**A consequence to state plainly:** making the block visible also makes
previously-hidden `Co-authored-by` lines visible to the forbidden-AI-trailer
rule. `f64f2b71` names a bot co-author and now fails where it silently passed.
That is the gate working, not a regression, but it means merges that attribute a
bot will red until the merge method stops adding them.

## Tests to port

None upstream, for the reason in Upstream chain. Written from scratch in the
checker's **paired** suite, `tests/scripts/test_check_commit_trailers.py`:

1. RED-BEFORE: a human `Co-authored-by:` appended below a valid block passes.
2. A doubled trailer block still **fails** (`b8293c88` stays red by design).
3. Contradictory `AI-Assisted` declarations still fail.
4. A "Merge pull request" message with no trailers still fails.
5. Prose after the trailer block still fails — proving the fusion is bounded.

Cases 2–5 are the guards that keep the fusion from becoming a hole; all four were
already green before the change and must stay green.

## Gates

- `scripts/agent-preflight.sh` and `--staged`.
- `tests/scripts/test_check_commit_trailers.py`, read by test-case COUNT.
- The five real `main` commits above, each re-checked by
  `check-commit-trailers.py --range '<sha>~1..<sha>'`, with the verdict table
  reproducing exactly.
- `python3 scripts/agent-integration.py --base origin/main`.
- No CUDA, GPU or SACRED gate is implicated: no product source is touched.

## Evidence

The `git interpret-trailers --parse` output above; the 13-of-30 count on `main`;
and the per-commit verdict table before and after, which is the binding result.

## Dependencies

None.

## Work breakdown

| ID | Work | Done when |
|---|---|---|
| W1 | RED-BEFORE case for the appended human co-author | fails before the fix |
| W2 | `join_trailing_trailer_paragraphs` + `parsed_trailers` | W1 green, guards still green |
| W3 | Re-verify the five real commits | table reproduces |

## Risks / decisions

- **Fusion could hide a malformed block.** Bounded by requiring every line of
  every fused paragraph to be trailer-shaped, and pinned by the prose-after guard.
- **This does not make `main` green on its own.** Four of the five shapes are
  merge-method artifacts that remain red by design. The durable fix for those is
  how commits land, which is a process change and not this row's scope.

## Stop conditions

Return `NEEDS_DECISION` rather than widening scope if closing the remaining four
shapes would require relaxing uniqueness, the marker placement rule, or the
forbidden-AI-trailer rule. Those are the guarantees the gate exists to hold.

## Now

W3 done; the row is landing. Next: decide the merge-method change that closes the
four remaining shapes, which is process, not checker.

## Outcome

Pending.
