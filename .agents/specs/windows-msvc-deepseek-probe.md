# Native MSVC DeepSeek V4 expert-probe narrowing repair

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#464](https://github.com/mudler/vllm.cpp/issues/464)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Predecessor repairs:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md) and
[windows-msvc-central-nominmax.md](windows-msvc-central-nominmax.md)

Status: `ACTIVE`. This repair starts from PR #446 head
`7747ac097c56fc020bfc92ad493cafa7dbcd17a0` and preserves the native MSVC
`/W4 /WX` contract.

## Scope

Make the deterministic DeepSeek V4 expert-probe input sequence intentionally
float-typed. Preserve its `0.5 * sin(frequency * (i + 1))` shape and the two
frequencies (`0.017`, `0.013`), and do not alter model inference, weights,
release assets, ABI, or warning strictness. Do not hide the conversion with a
result cast; select the float `std::sin` overload by making the literals and
index conversion float before the call.

## Observed baseline and root cause

Hosted run `31586472591` failed native CPU job `94081367518` and Vulkan job
`94081367597` at `src/vllm/model_executor/models/deepseek_v4.cpp:2398-2399`.
The complete logs have the same unique diagnostic set: C4244 promoted through
C2220, and no other warning/error family. Each assignment stores into
`std::vector<float>`, but `0.017`/`0.013` and `(i + 1)` select double
multiplication and the double `std::sin` overload; multiplying its double
result by `0.5f` remains double before the narrowing assignment.

## Design and tests

Extract the sequence construction into an internal model helper used by the
expert probe. Its implementation uses `0.5f`, a float frequency, and
`static_cast<float>(i + 1)`, so overload resolution remains float throughout.
Add a focused DeepSeek scaffold unit that verifies exact float bit patterns for
the first four `0.017f` values and representative `0.013f` values. The second
and later `0.017f` samples deliberately differ by one ULP from the old
double-then-narrow sequence, so the test proves the intended arithmetic rather
than accepting a warning-hiding result cast. Add a source-contract assertion to
the Windows portability suite so reintroducing either double literal is caught
before hosted MSVC.

RED must be captured before implementation: the semantic test links against a
declared but absent helper, and the source-contract test rejects both current
double literals. GREEN requires both focused tests, the relevant DeepSeek V4
test target, and the complete Windows portability suite.

## Gates

1. Focused RED for the semantic helper and both source expressions.
2. Focused DeepSeek V4 scaffold test and complete Windows portability suite.
3. Clean CPU and Vulkan source-closure builds when feasible locally.
4. Exact PR #446 range gate from `a170c81c` to the immutable candidate.
5. Full unstaged, staged, and post-commit `scripts/agent-preflight.sh`.
6. Native CPU and Vulkan reruns under `/W4 /WX`; Linux cannot substitute for
   this compiler gate.

## Risks and stop conditions

The float overload changes a few samples by one ULP relative to double then
narrowing; that is intentional because the stored sequence is a float probe and
the compiler contract forbids the implicit narrowing. Stop with
`NEEDS_DECISION` if preserving double evaluation is required product behavior,
or if the repair would need a warning suppression or public ABI change. Stop
with `NEEDS_CONTEXT` if either complete hosted log contains another diagnostic
family.

## Outcome

Pending implementation and immutable gate evidence.
