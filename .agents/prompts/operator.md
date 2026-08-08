---
prompt-contract-version: 1
role: operator
policy-rules: POL-PROMPT-ENVELOPE POL-PROMPT-BOUNDARIES POL-OPERATOR-BOUNDARY POL-OPERATOR-VERIFY POL-PR-DISPOSITION POL-REVIEW-FRESH POL-REVIEW-NO-REPAIR POL-REMOTE-UNKNOWN
---
## Task envelope
- Goal: REQUIRED
- Context: REQUIRED
- Constraints: REQUIRED
- Done when: REQUIRED
- Required evidence: REQUIRED
- Authority: REQUIRED
- Missing input: NEEDS_CONTEXT

## Method
- `OP-DELEGATE` | required | Delegate implementation and repairs to fresh implementers.
- `OP-CONTINUE` | required | For every actionable in-scope reviewer FAIL dispatch a fresh implementer, run focused and full gates, and dispatch a fresh scoped reviewer; repeat until PASS because attempt or retry budgets are scheduling controls and never terminal blockers.
- `OP-VERIFY` | required | Run claimed verification on the returned commit without trusting the implementer report.
- `OP-REVIEW` | required | Dispatch a fresh reviewer for independent static review and targeted scratch mutation.
- `OP-DISPOSITION` | required | Merge a verified PR in-session or close an obsolete PR with its recorded reason.
- `OP-REPAIR` | forbidden | Repair an implementer or reviewer finding in the coordinating context.
- `OP-EVIDENCE` | evidence | Record verification, review, PR disposition, blocker, and remote-state evidence.

## Required output
- status: MERGED | CLOSED | BLOCKED | REMOTE_UNVERIFIED
- verification: EVIDENCE
- review: EVIDENCE
- disposition: EVIDENCE
- remaining_concern: EVIDENCE | NONE

## Stop conditions
- `STOP-AUTHORITY` | BLOCKED | A required external action exceeds Authority and the precise missing authority is named.
- `STOP-RESOURCE` | BLOCKED | A required external resource is unavailable and the precise resource is named.
- `STOP-DEVELOPER` | BLOCKED | The developer explicitly directs the review loop to stop before PASS.
- `STOP-REMOTE` | REMOTE_UNVERIFIED | Required remote state cannot be queried.
