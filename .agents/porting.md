# Porting — mirror, upstream, tests, and shared seams

This procedure turns the policy registry into the repeatable vLLM porting
method. The pinned upstream source and current parity state remain in
[`.agents/upstream-sync.md`](upstream-sync.md) and the owning matrices.

## Port cycle

Verify the recorded gap against the current pinned upstream and local head.
Enumerate the complete upstream surface and tests, classify each mode, commit a
spike with source and dependency anchors, then implement the smallest coherent
vertical slice. Keep local names and structure mechanically traceable to
upstream, record necessary C++ adaptations, and advance the parity pin only
after all affected rows and gates are reconciled.

The implementation phase is spec-driven and test-first: port the upstream
failure before behavior, obtain focused green, run full validation, then follow
the fresh static/mutation review and operator-verification cycle in
[workflow](workflow.md) and [verification](verification.md).

<!-- policy-procedure:begin -->
[POL-MIRROR-VLLM] When vLLM defines behavior, mirror every applicable mode, default, error, and edge case. Escalate only a genuine product or scope decision; do not ask how a mirrored feature should behave.

[POL-SEAM-FUSION] Route model fusion through `vt::FusedChain`. If the shared seam cannot represent the upstream behavior, extend it or record one exact tracked exception rather than hand-rolling a parallel path.

[POL-SEAM-MERGED-GEMM] Route mergeable MLP projections through `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`; extend the shared family or declare one exact tracked exception.

[POL-SEAM-RUNNER] Route decode through `ModelRegistry::Forward`, `dense_attn::AttnBlock`, and on-device sampling; examples and servers consume the shared library path instead of owning another runner.

[POL-PORT-TESTS] Port the applicable upstream test module in the same change as its code, preserving parameters, modes, fixtures, tolerances, failure cases, and upstream revision anchors; document only unavoidable harness adaptation.

[POL-TABLE-INVENTORY] Give every inventory item a stable ID and record upstream source, local anchor, tests/evidence, committed spike, lifecycle state, and owner in the correct matrix.
<!-- policy-procedure:end -->

## C ABI and evidence boundary

A user capability is a library capability: reusable behavior is reachable
through `include/vllm.h`, while examples and servers remain thin consumers.
The owning spec identifies ABI additions and tests alongside engine behavior.

Use the execution-chain and same-tool trace method from verification for
performance or kernel claims. The parity ledger records each introduced change
and its upstream reference; matrices hold current coverage; append-only records
retain attempts and forensic detail. Do not duplicate those facts here.
