# The forge's `Co-authored-by` is attribution, not an authorship claim

Issue: [#418](https://github.com/mudler/vllm.cpp/issues/418).
Row: `ENG-FORGE-COAUTHOR`. Follow-up to
[#406](https://github.com/mudler/vllm.cpp/issues/406) /
[trailer-merge-artifacts.md](trailer-merge-artifacts.md).

#406 fixed how the trailer block is *found*. This fixes the one rule that fix
exposed: GitHub's auto-generated `Co-authored-by` names the account that opened
the pull request, most of which here are bots, and the AI-identity rule rejects
it.

## Scope

**In scope.** The forbidden-AI-trailer rule in `check-commit-trailers.py` as it
applies to `Co-authored-by`, and the matching sentence in `AGENTS.md`.

**Out of scope.** `Signed-off-by`, which keeps its rule with no exemption. The
`AI-Assisted` / `Assisted-by` declarations, which are untouched. Every other
assertion the checker makes. No product source.

## Upstream chain

None. Local protocol machinery, so the mirror rule does not apply and there is no
upstream `file:line` to port from. Governed by `AGENTS.md` §"Changing the rules or
a checker", which requires a spec, a red-before test or mutation, and green-after
evidence, and forbids turning a red gate green by deleting an assertion.

## Our baseline — the rule catches the wrong thing

`AGENTS.md` states *"AI tools never add `Signed-off-by` or `Co-Authored-By`."*
The checker enforces it by rejecting any `Co-authored-by` whose value matches an
AI identity token (`bot`, `agent`, `claude`, `codex`, …).

GitHub composes squash-merge messages itself and appends the submitting account:

```
Co-authored-by: localai-org-maint-bot <306269227+localai-org-maint-bot@users.noreply.github.com>
```

Most pull requests here are opened by `localai-bot` or
`localai-org-maint-bot`, so that line is written on nearly every squash and the
commit fails the gate. Real instance on `main`: `f64f2b71`. It was invisible
until #406 repaired the parse, which is why it reads as a new failure and is not
one.

**The rule exists so an AI cannot claim it wrote the code.** That concern is
real and stays. But GitHub is not making an authorship claim — it is recording
which account pressed the button, and that account has an audit trail. The honest
statement about AI involvement is already carried, separately and explicitly, by
`AI-Assisted: true` and `Assisted-by: …`, which sit in the same block and are not
touched here. Conflating attribution with authorship reds `main` for correctly
authored work while the actual declaration sits one line above, unread.

## Port map

| Item | Local anchor | Motion |
|---|---|---|
| `FORGE_ACCOUNT_EMAIL` | `scripts/check-commit-trailers.py` | new |
| forbidden-trailer loop | same file | skip forge-addressed `Co-authored-by` |
| `Signed-off-by` handling | same file | **unchanged** |
| `AI-Assisted` / `Assisted-by` rules | same file | **unchanged** |
| the authorship sentence | `AGENTS.md` | add the distinction |

## Design

Accept a `Co-authored-by` whose address is a GitHub account noreply address
(`…@users.noreply.github.com`), even when the name matches an AI identity token.

**Keyed on the forge's own domain, not on the name.** That is what stops the
exemption being borrowed: a hand-written `Co-authored-by: Claude
<claude@anthropic.com>` still fails, because a model crediting itself does not
get to pick GitHub's address space. `Signed-off-by` is excluded from the
exemption entirely — a sign-off is a legal assertion about provenance, not
attribution, and there is no reading under which a bot should make one.

`AGENTS.md` records the same distinction in the same change, so the prose and the
checker do not drift apart.

## Tests to port

None upstream, for the reason in Upstream chain. Written from scratch in the
checker's **paired** suite. Because this **loosens** an attribution rule, the
guards matter more than the relaxation and are asserted explicitly:

1. RED-BEFORE: a bot `Co-authored-by` at a GitHub noreply address passes — the
   `f64f2b71` shape.
2. GUARD: a hand-written `Co-authored-by: Claude <claude@anthropic.com>` still
   fails.
3. GUARD: `Signed-off-by` at the same noreply address still fails — the forge
   address must not buy an AI a sign-off.
4. GUARD: a human `Co-authored-by` still passes, so #406's fix is not disturbed.

Guards 2–4 were green before the change and must stay green; that is what makes
this a narrowing of one rule rather than its removal.

## Gates

- `scripts/agent-preflight.sh` and `--staged`.
- `tests/scripts/test_check_commit_trailers.py`, read by test-case COUNT.
- `check-commit-trailers.py --range 'f64f2b71~1..f64f2b71'` on the real commit.
- `python3 scripts/agent-integration.py --base origin/main`.
- No CUDA, GPU or SACRED gate is implicated: no product source is touched.

## Evidence

The generated line from `f64f2b71`; that commit passing after the change; and the
three guards staying red/green as stated.

## Dependencies

#406, landed at `157080c8`. Without the parse fix this rule never fires, because
the trailer block is invisible in the first place.

## Work breakdown

| ID | Work | Done when |
|---|---|---|
| W1 | Red-before + three guards | W1 fails, guards green |
| W2 | `FORGE_ACCOUNT_EMAIL` and the loop skip | all four green |
| W3 | `AGENTS.md` distinction | prose matches the checker |

## Risks / decisions

- **The exemption could be borrowed** by a tool that writes a GitHub noreply
  address for itself. Accepted and bounded: such an address belongs to a real
  account with an audit trail, and guard 2 keeps every other spelling closed.
- **It loosens an attribution rule.** Deliberate and developer-approved, and the
  narrowest form found: one trailer, one address form, no change to sign-off or
  to the AI declarations.

## Stop conditions

Return `NEEDS_DECISION` rather than widening if closing the remaining merge-method
shapes (the `---------` separator, or the doubled block from a multi-commit
squash) would need the uniqueness rule relaxed. That was already attempted and
reverted under #406 and remains refused.

## Now

`ACTIVE`; W1-W3 implemented. Next: land, then close the row.

## Outcome

Pending.
