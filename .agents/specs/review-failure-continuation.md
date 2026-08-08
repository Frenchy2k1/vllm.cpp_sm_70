# Review-failure continuation

User-approved 2026-08-08. Governance task: strengthen the existing
`POL-REVIEW-NO-REPAIR` contract so an actionable review failure cannot be
converted into a terminal result by an orchestration retry budget.

## Verified gap

The policy registry and workflow already require reviewer findings to return
to a fresh implementer, followed by focused and full gates and a fresh scoped
reviewer. They do not state that this cycle continues until `PASS`, and the
operator prompt still permits a generic `BLOCKED` result. A generic attempt or
retry budget can therefore stop work while an actionable, in-scope finding is
still correctable.

The branch also inherits a cutover-record defect from the policy-history
squash. `.agents/policy-cutover` names commit
`00927ed611f4c5b720ceb158f6174be1e5470b03`, which exists locally but is not an
ancestor of the current head. The fail-closed trailer checker correctly rejects
that unreachable anchor. This spec commit is intentionally strict and becomes
the replacement reachable cutover anchor; a following strict commit updates
only the marker.

## Binding design

Strengthen `POL-REVIEW-NO-REPAIR` rather than introduce an overlapping rule.
Every actionable, in-scope reviewer `FAIL` starts this cycle:

1. send the bounded finding and evidence to a fresh implementer;
2. rerun the focused and full gates on the resulting immutable head; and
3. send that head to a fresh scoped reviewer.

Repeat the cycle until the reviewer returns `PASS`. Attempt and retry budgets
are scheduling controls, never terminal blockers for correctable findings.
The cycle stops short of `PASS` only on explicit developer direction or a
precise external authority or resource blocker. The coordinating/operator
session continues to coordinate and independently verify; it does not repair
the finding itself.

## Affected surfaces

- `.agents/policy.csv`: strengthen the authoritative
  `POL-REVIEW-NO-REPAIR` requirement.
- `.agents/workflow.md`: bind the exact continuation procedure.
- `AGENTS.md`: refresh the generated compact T0 projection.
- `.agents/prompts/operator.md`: close the operator output/stop grammar so a
  correctable finding cannot be reported as terminal `BLOCKED`.
- `scripts/check-protocol-consistency.py` and
  `tests/scripts/test_check_protocol_consistency.py`: require and mutation-test
  the continuation semantics.
- `tests/scripts/test_policy_contract.py` and
  `tests/scripts/test_check_prompt_contract.py`: cover the synchronized policy
  projection and closed prompt contract where applicable.
- `.agents/state.md`, `.agents/NOW.md`, `docs/STATUS.md`, and
  `docs/BENCHMARKS.md`: record the governance checkpoint without changing a
  feature/model/backend/quantization surface.
- `.agents/policy-cutover`: replace the unreachable SHA with this spec commit's
  full SHA in a separate strict commit, without weakening any checker.

## Verification and evidence

The cutover repair must prove the replacement anchor is reachable and all
commits at or after it use strict Git-parsed trailers:

```sh
cutover=$(tr -d '\n' < .agents/policy-cutover)
git merge-base --is-ancestor "$cutover" HEAD
python3 scripts/check-commit-trailers.py \
  --range origin/main..HEAD --cutover "$cutover"
```

The policy change uses red-before mutations for the continuation phrases and
the operator prompt row, then runs:

```sh
python3 scripts/check-policy.py
python3 scripts/check-prompt-contract.py
python3 scripts/check-protocol-consistency.py
python3 -m unittest \
  tests.scripts.test_policy_contract \
  tests.scripts.test_check_prompt_contract \
  tests.scripts.test_check_protocol_consistency
scripts/agent-preflight.sh
```

No checker is relaxed, no attempt budget can terminate an actionable review
loop, and no remote operation is part of this task.
