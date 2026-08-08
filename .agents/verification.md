# Verification — gates, evidence, review, and performance

[Policy](policy.csv) supplies the rules. This procedure describes how to prove
them without promoting a report into evidence.

## Gate sequence

Start with the smallest deterministic test that can falsify the spec. Preserve
the intended red result, make it green, run the declared focused gate, then run
the full repository preflight. A gate report records the immutable SHA,
command, environment, exit status, and relevant output/evidence path.

Review occurs only after implementation gates pass. Static review checks the
spec, diff, tests, error paths, ownership boundaries, and policy mapping;
scratch mutation review temporarily removes or corrupts each critical guard
and proves the focused test fails. Restore the reviewed tree byte-for-byte
after every mutation.

<!-- policy-procedure:begin -->
[POL-GROUND-CHAIN] For parity conclusions, inspect and cite the applicable vLLM path plus every executing dependency layer—such as FlashInfer, CUTLASS, cuBLASLt, DeepGEMM, torch/Inductor, generated code, and the local dispatch path. Dump the generated kernel before calling a lever unreachable.

[POL-TRACE-SAME-TOOL] Before throughput comparison, trace both implementations on the identical workload with the same profiler and relevant graph-node tracing; source inspection establishes candidates, while matching traces establish what ran.

[POL-GEMV-CONTRACT] A GEMM/GEMV invocation-parity claim proves in the same tool: output/C dtype, compute and scale type, entry point and algorithm policy, and resolved kernel-template dtypes. Cross-tool traces cannot satisfy this claim.

[POL-ORACLE] Correctness and performance gates use the pinned oracle, identical model artifacts, prompts, token counts, batching/concurrency, sampling settings, and measured workload on both sides.

[POL-CORRECTNESS-GATE] Establish the declared token-exact gate—or an explicitly ratified distributional gate—before accepting performance. Never trade correctness for throughput.

[POL-PERF-EVERY-AXIS] Record both values and the ratio for every required throughput, latency, memory, model, and workload axis; any below-floor axis remains an open gap.

[POL-REPRODUCE] Record the exact build and run recipe, revisions, model hashes, environment, clocks/contention state, raw output, and same-binary A/B; rerun the claimed result on an idle device before acceptance.

[POL-NO-CEILING] Treat an apparent same-architecture performance ceiling as an unresolved implementation difference. Keep the gap open and name the next traceable hypothesis rather than declaring completion.
<!-- policy-procedure:end -->

## Evidence and independent review

Evidence distinguishes observed facts from inference and names source roots,
versions, file-and-line anchors, commands, artifacts, and limitations. Failed
attempts and refuted hypotheses remain in append-only records; public documents
contain only the keyed current projection.

A reviewer reports `PASS` only after both static and mutation review on the
same immutable head. Findings include severity, violated spec or policy ID,
reproduction, and expected behavior. The operator then runs the declared gate
independently before disposition; a reviewer's green report is not a substitute.

Specialized methods remain in
[`.agents/parity-lever-protocol.md`](parity-lever-protocol.md), the benchmark
record, gate-specific specs, and environment registry. They are technical
evidence and recipes, not additional policy registries.
