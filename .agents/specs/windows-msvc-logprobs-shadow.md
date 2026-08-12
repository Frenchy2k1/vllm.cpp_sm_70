# Native MSVC LogprobsTensors slice shadow repair

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#465](https://github.com/mudler/vllm.cpp/issues/465)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Predecessor repairs:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md) and
[windows-msvc-deepseek-probe.md](windows-msvc-deepseek-probe.md)

Status: `ACTIVE`. This repair starts from exact PR #446 head
`b7e04d7afdda36b092352965304a8ea2917df0d6` and preserves the native MSVC
`/W4 /WX` contract.

## Scope

Remove the one native MSVC C4458 diagnostic in
`LogprobsTensors::slice_request` without changing the method signature's types,
visibility, return shape, selected rows, public ABI, inference behavior,
release assets, or warning strictness. Rename only the shadowing parameter in
the declaration and definition, and explicitly initialize the output member
from that renamed value. Do not suppress C4458 and do not expand this repair to
unrelated naming or formatting changes.

## Observed baseline and root cause

Hosted run `31590520904` failed native CPU job `94094182639` at
`src/vllm/v1/outputs.cpp:24` and native Vulkan job `94094182635` at the same
line. The complete logs have the identical unique diagnostic set: C4458
promoted through C2220, with the compiler pointing to
`include/vllm/v1/outputs.h:37` as the hidden member. No second warning or error
family appears in either log.

The issue intake calls the site a `RequestOutput` constructor, but the executing
source and git history establish the actual site as
`LogprobsTensors::slice_request(int req_idx, int num_positions) const`. The
parameter introduced with the slice method has the same spelling as
`LogprobsTensors::num_positions`; unqualified uses in the body refer to the
parameter while `out.num_positions` refers to the output object's member. The
runtime values are correct, but MSVC intentionally diagnoses this lexical
hiding at warning level 4 and `/WX` makes it a release-build error.

## Design and tests

Add a structural regression over the real `slice_request` definition that
rejects a parameter named `num_positions` in that method's scope. Capture RED
on the pinned head with the exact shadowing signature. Add a semantic focused
case that constructs a multi-row `LogprobsTensors`, slices a nonzero request
offset for more than one row, and verifies the output dimensions and all three
selected payload arrays against hand-derived values. This protects the behavior
the parameter rename must preserve.

After RED, rename the parameter to `request_num_positions` in the installed
header and implementation, assign `out.num_positions` explicitly from it, and
use the renamed value for the end offset. The parameter name is not part of the
C++ function type or symbol, so this is ABI-neutral.

## Gates

1. RED structural source regression at exact pinned head, reporting the
   shadowing `slice_request` signature.
2. GREEN structural regression plus the focused LogprobsTensors semantic test.
3. Direct Windows portability checker and its complete mutation suite.
4. Clean CPU and Vulkan source-closure builds and the relevant output/request
   tests when feasible locally.
5. Exact PR #446 range gate from `a170c81c` to the immutable candidate.
6. Full unstaged, staged, and post-commit `scripts/agent-preflight.sh`.
7. Native CPU and Vulkan reruns under `/W4 /WX`; Linux cannot substitute for
   this compiler gate.

## Risks and stop conditions

The risk is accidentally using the object member (the batch-wide source shape)
where the requested slice length belongs, changing the selected range. The
semantic multi-row/nonzero-offset case must catch that substitution. Stop with
`NEEDS_DECISION` if the fix requires a warning suppression, signature-type or
ABI change, or runtime/release behavior change. Stop with `NEEDS_CONTEXT` if a
complete native log contains any other diagnostic family.

## Outcome

Pending RED, implementation, and immutable gate evidence.
