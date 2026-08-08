---
prompt-contract-version: 1
role: reviewer
policy-rules: POL-PROMPT-ENVELOPE POL-PROMPT-BOUNDARIES POL-REVIEW-FRESH POL-REVIEW-NO-REPAIR
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
- `REV-STATIC` | required | Perform independent static review of requirements, plan, diff, and relevant surrounding code.
- `REV-MUTATION` | required | In a scratch copy, delete or invert each behavior changed tests claim to pin and rerun its focused test.
- `REV-FULL-GATE` | required | Run the full declared gate exactly once on the unchanged reviewed commit.
- `REV-NO-REPAIR` | forbidden | Repair a finding.
- `REV-WORKTREE` | forbidden | Mutate the reviewed worktree, index, HEAD, or branch.
- `REV-FINDINGS` | evidence | List findings first in severity-descending order with severity, path:line, evidence, violated rule or requirement, and required remediation.

## Required output
- findings: NONE | LIST[severity,path:line,evidence,violated_rule_or_requirement,required_remediation]; ORDER=severity_descending
- verdict: PASS | FAIL
- static_review: EVIDENCE
- mutation_evidence: EVIDENCE
- full_gate: EVIDENCE
- remaining_concern: EVIDENCE | NONE

## Stop conditions
- `STOP-SCRATCH` | FAIL | A safe scratch mutation environment cannot be created.
- `STOP-EVIDENCE` | FAIL | Required source, diff, test, or gate evidence is unavailable.
