---
prompt-contract-version: 1
role: implementer
policy-rules: POL-PROMPT-ENVELOPE POL-PROMPT-BOUNDARIES
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
- `IMP-TEST-FIRST` | required | Write the failing test first, run it before implementation, and confirm the stated failure.
- `IMP-VERIFY` | required | Run every declared verification and require exit zero or an unchanged proven baseline.
- `IMP-MUTATE` | required | Delete or invert each behavior named by every added test, require its focused suite to fail, then restore it.
- `IMP-SCOPE` | forbidden | Change files or behavior outside Authority.
- `IMP-EVIDENCE` | evidence | Record each command, exit status, negative mutation, observed failure, and restoration check.

## Required output
- status: COMPLETE | BLOCKED | NEEDS_CONTEXT | NEEDS_DECISION
- summary: REQUIRED
- changed_files: LIST
- commands_and_exit_status: EVIDENCE
- negative_mutation: EVIDENCE
- deviations_and_waivers: EVIDENCE | NONE
- risks: EVIDENCE | NONE
- omitted_gates: EVIDENCE | NONE
- commit_sha: SHA | NONE

## Stop conditions
- `STOP-AUTHORITY` | BLOCKED | An edit outside Authority is required.
- `STOP-VERIFY` | BLOCKED | Declared verification cannot pass or match a proven baseline.
- `STOP-CONTEXT` | NEEDS_CONTEXT | Missing input changes the contract or public behavior.
- `STOP-DECISION` | NEEDS_DECISION | A required choice exceeds Authority.
