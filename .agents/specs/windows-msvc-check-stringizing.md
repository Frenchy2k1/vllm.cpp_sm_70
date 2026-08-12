# Native MSVC CHECK stringizing repair

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#474](https://github.com/mudler/vllm.cpp/issues/474)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Predecessor repairs:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md) and
[windows-msvc-logprobs-shadow.md](windows-msvc-logprobs-shadow.md)

Status: `ACTIVE`. This repair starts from exact PR #446 head
`2bf417f3ed80360e1adcd52224fd738c0a28eb90` and preserves the native MSVC
`/W4 /WX` contract.

## Scope

Make the byte-exact Windows command-line expectation compile under MSVC without
weakening its payload assertion. Bind the expected command line to a named
`const std::wstring` outside the doctest `CHECK`, use a non-empty custom raw
string delimiter, and compare the production result to that identifier. Keep
the argv inputs and every expected character unchanged. Do not change
`BuildWindowsCommandLine`, suppress any warning, relax `/W4 /WX`, or include the
unrelated Vulkan embedding failures tracked by #461.

## Observed baseline and root cause

Hosted run `31597279008` compiled the project library successfully and then
failed native CPU job `94115788855` and Vulkan job `94115788793` while parsing
`tests/vllm/entrypoints/openai/test_api_server.cpp:2594`. Both complete job logs
have the same direct diagnostics: C4129, C2017, and C3688, followed by cascading
C2661 and `/WX` wrapper C2220. No production command-line-builder diagnostic is
present.

The initial issue hypothesis blamed the default raw-string delimiter, but the
literal is standard-valid: its payload ends with `""` followed by the one
closing `)` and does not itself contain the delimiter token `)"`. The actual
boundary is doctest's macro stringizing of the full `CHECK` expression, which
contains the backslash-bearing raw literal. Microsoft's official
[stringizing operator documentation](https://learn.microsoft.com/en-us/cpp/preprocessor/stringizing-operator-hash?view=msvc-170)
states that MSVC C++ stringizing is incorrect for strings containing escape
sequences and emits C2017; its
[C2017 documentation](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2017?view=msvc-170)
shows the same stringize-plus-backslash failure class. C4129/C3688/C2661 are
downstream parser/macro fallout after that boundary is corrupted.

## Design and tests

Add a structural regression over the real named test case. It must extract the
custom-delimited wide raw literal from a named `const std::wstring expected`,
assert its payload equals the hand-derived byte sequence
`"ffmpeg" "two words" "C:\\path\\\\" "a\\"b" ""`, and require `CHECK` to
compare `BuildWindowsCommandLine(argv)` with the identifier only. It must reject
the pinned shape where the backslash-bearing literal is directly in `CHECK`.

Capture RED before implementation. Then make only this source transformation:

1. introduce `const std::wstring expected = LR"cmd(... )cmd"` beside `argv`;
2. change `CHECK(BuildWindowsCommandLine(argv) == literal)` to
   `CHECK(BuildWindowsCommandLine(argv) == expected)`.

The custom delimiter is defensive clarity, not the root-cause fix; moving the
literal outside macro stringizing is load-bearing. The existing runtime test
continues to compare the complete production result, so no byte or semantic
assertion is dropped.

## Gates

1. RED structural portability regression at exact pinned head, diagnosing the
   absent named expectation/direct literal inside `CHECK`.
2. GREEN focused structural regression and direct production portability
   checker, then the complete Windows portability suite.
3. Build and execute `test_openai_api_server` in the relevant local CPU and
   Vulkan configurations when proportionate; #461 failures remain separately
   attributed rather than repaired here.
4. Full unstaged, staged, and post-commit `scripts/agent-preflight.sh`.
5. Exact PR #446 range gate from `a170c81c` to the immutable candidate.
6. Native CPU and Vulkan reruns under `/W4 /WX`; Linux cannot substitute for
   MSVC macro/preprocessor behavior.

## Risks and stop conditions

The primary risk is making the test compile by changing or shortening the
expected command-line bytes. The structural regression therefore extracts the
payload independently, while the C++ test still executes the full production
comparison. Stop with `NEEDS_DECISION` if the repair requires changing the
builder, argv contract, warning policy, or expected bytes. Stop with
`NEEDS_CONTEXT` if either complete native log exposes a distinct direct
diagnostic outside this test site.

## Outcome

Pending RED, implementation, and immutable gate evidence.
