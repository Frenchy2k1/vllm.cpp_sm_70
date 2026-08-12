# Tenstorrent residual-RMS golden parity at the device boundary — spike

Status: **DRAFT, 2026-08-11.** Owes a `RED`-first mutation before the row can
move past `READY`. No code change is in scope until the gap is confirmed and the
decision (§Risks/decisions) is recorded.

Proposed row id: `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` (a child of
`BACKEND-TENSTORRENT`, same single-row pattern Metal/Vulkan device-boundary
goldens use).

## Scope

**In.** Produce the owed evidence for the bot-flagged numerics divergence at
the residual-RMS device boundary in
`src/vt/tenstorrent/tenstorrent_ops.cpp::RmsNormKernel`:

- The host path (`rows < kDeviceResidualMinRows`, i.e. `< 32`, or `gemma`)
  merges `x + residual` and accumulates `sumsq` in **f32**, mirroring
  `src/vt/cpu/cpu_ops.cpp::RmsNormKernel` exactly (store-back to the residual
  in its dtype, re-read, then f32 variance).
- The device path (`rows >= 32`, non-gemma) does `ttnn::add(dev_x, dev_r)` then
  `ttnn::rms_norm(...)` on BFLOAT16 TILE tensors (`TileSpecOf` →
  `DataType::BFLOAT16`). Both the residual merge and the variance reduction
  therefore round to bf16 mid-computation, where the host/CPU path keeps f32.

This spike answers one question: **is the device path's divergence from the
CPU f32 oracle within an acceptable, ratifiable band at `rows == 32` and just
above it, or does the threshold `kDeviceResidualMinRows` (or the device dtype)
need to change?**

**Out.** No change to the e2e Qwen3 greedy golden
(`tests/parity/test_qwen3_paged_engine.cpp`'s device-golden path), no perf
work, no RoPE/residual fusion, no model expansion, no MoE. If the evidence
forces a code change, that change is a follow-on under this same row, not a new
row.

## Upstream chain

**No upstream vLLM equivalent.** vLLM has no Tenstorrent platform. The loyal
contract is the CPU `RmsNormKernel` in this repo, which mirrors vLLM's
`fused_add_rms_norm` (add in the model dtype, accumulate variance in f32). The
device path's only obligation is to stay inside whatever band the CPU oracle
and vLLM's own bf16 numerics justify — identical to how Metal's device RMS is
held against the same CPU reference.

Reference anchors:
- `src/vt/cpu/cpu_ops.cpp:371-398` — the f32 CPU oracle (sumsq in f32,
  store-back rounds to the residual dtype, re-read).
- `src/vt/tenstorrent/tenstorrent_ops.cpp:1067-1117` — the host/device split
  and the device path under test.
- `include/vt/ops.h:409-412` — `RmsNormArgs{eps, gemma}`.
- `tests/vt/test_ops_rmsnorm.cpp:64-135` — the existing op-level RMS residual
  test shape this spike mirrors.

## Our baseline

**Landed on `origin/main`** (`59568772`, the trace tip — an ancestor of
`f64f2b71`): the `kDeviceResidualMinRows = 32` threshold and both paths. The
Qwen3-0.6B short warm smoke (~12 tok/s) runs through the device path at
multi-token prefill and the host path at `T=1` decode. **No op-level
cross-device golden exists at the `rows == 32` boundary** — the only TT
numerics evidence today is the e2e greedy golden
(`tests/parity/goldens/qwen3_greedy_0_6b/our_ids_tenstorrent.npy`), which is a
coarse end-to-end check, not a boundary numerics probe. That is the gap.

## Port map

No port — this is an evidence row. The test mirrors
`tests/vt/test_ops_rmsnorm.cpp`'s residual cases but constructs the same
inputs on both `kTENSTORRENT` and `kCPU` and compares outputs element-wise at
`rows ∈ {1, 31, 32, 33, 64, 128}` (spanning the boundary both ways), with the
residual in both f32 and bf16, and with/without `gemma`. It goes through the
public `vt::RmsNorm` seam only — no ttnn headers in the test (mirrors
`test_tenstorrent_backend.cpp`'s no-leak rule).

## Tests to port

None upstream. The new test is
`tests/vt/test_tenstorrent_rms_residual.cpp` (or an appended
`TEST_CASE` in `test_tenstorrent_backend.cpp` if that keeps the CMake surface
smaller — to be decided in implementation). It `SKIP`s when no Blackhole card
is present (same `DeviceAvailable()` gate as the rest of the TT suite).

## Gates

**Correctness gate (the one this row owes):** for every `rows` case above,
`max_abs_diff(tenstorrent_out, cpu_out)` and the relative error against the
CPU f32 oracle are recorded, with the residual dtype and `gemma` flag noted.
The gate passes when either:

1. the device path stays within a ratifiable band (to be set from the first
   real measurement, then mutation-proved non-vacuous), OR
2. a divergence large enough to move the e2e greedy golden is demonstrated, in
   which case the threshold/dtype decision in §Risks/decisions becomes
   mandatory and the code change is the follow-on.

**Hardware:** real Blackhole (P150) is required — the device path does not
exist without the card. CPU-only CI will compile-skip the case.

**Not claimed:** no perf axis, no latency, no memory. This is numerics only.

## Dependencies

- `BACKEND-TENSTORRENT` (parent) — `ACTIVE` on `origin/main`.
- Toolchain/hardware: the same tt-metal install + Blackhole card the rest of
  the TT suite uses.
- No model or dataset dependency.

## Work breakdown

1. **Probe (RED-first).** Add the op-level cross-device test; run it on
   Blackhole; record `max_abs_diff` per case. Expected before any fix: at
   least one case at/above `rows == 32` shows a measurable (not necessarily
   failing) divergence from the CPU f32 oracle — the mutation that proves the
   test is non-vacuous is forcing the device path to f32 and showing the gap
   close.
2. **Decision.** From the recorded band, decide one of: (a) accept the device
   bf16 path as within parity (record the band, close the row); (b) raise
   `kDeviceResidualMinRows` so the boundary moves above the measured-safe
   range; (c) force an f32 accumulation on device for the residual merge.
   (a) is the likely outcome — vLLM's own bf16 models carry the same rounding
   — but the evidence must say so, not the prior.
3. **If (a):** record the band in the spec, the backend-matrix row, and
   `docs/STATUS.md`; close the row. **If (b)/(c):** land the code change
   under this row, re-run the probe green, then record.

## Risks/decisions

- **The likely decision is "accept."** vLLM ships bf16 `RmsNorm` everywhere;
  the CPU f32 oracle is stricter than vLLM's own runtime. A device bf16 path
  that matches vLLM's rounding is parity-correct, not a defect. But "likely"
  is not "measured" — the bot's flag is precisely that no one has looked at
  the boundary. The probe settles it.
- **The threshold is load-bearing for perf.** The handoff (§6) records that
  "always device residual/RoPE" regressed Qwen3-0.6B from ~12.3 to ~10.5
  tok/s, and the hybrid threshold restored it. Raising
  `kDeviceResidualMinRows` blindly to close a numerics gap that turns out to
  be below the e2e noise floor would re-introduce that regression for no
  correctness gain. The decision must weigh both.
- **f32-on-device is not free.** `ttnn::rms_norm` in f32 is slower and may
  not be a path the installed ttnn exercises; option (c) is the last resort,
  not the default.

## Outcome (2026-08-11, real Blackhole P150 `blackhole-94712C24111071E4`)

The RED-first probe landed in `tests/vt/test_tenstorrent_backend.cpp` as
`kTENSTORRENT kRmsNorm residual: device vs CPU f32 oracle across the
rows=32 boundary` and ran on the card (22/22 cases, 826/826 assertions,
clean exit). D=1024 (Qwen3-0.6B hidden width), `RmsNormArgs{eps=1e-6,
  gemma=false}`, f32 residual, deterministic LCG inputs in [-1, 1):

| rows | path        | max_abs vs CPU f32 | max_rel |
|------|-------------|--------------------|---------|
| 1    | host (f32)  | 0                  | 0       |
| 31   | host (f32)  | 0                  | 0       |
| 32   | device bf16 | 0.0459             | 1.94    |
| 33   | device bf16 | 0.0459             | 1.94    |
| 64   | device bf16 | 0.0459             | 1.95    |
| 128  | device bf16 | 0.0459             | 2.57    |

**The bot's flag was exactly right.** At `rows < 32` the TT host path is
bit-identical to CPU (`max_abs=0`); at `rows >= 32` the device bf16 path
diverges by a constant ~0.046 absolute. The constant `max_abs` across
rows={32,33,64,128} is the signature of bf16 rounding on a single
near-zero output element, not an error that accumulates with rows — the
large `max_rel` (1.9–2.6×) is the same small absolute delta divided by a
near-zero CPU reference value, exactly where RMSNorm output crosses zero.

**Likely decision: accept.** vLLM's own bf16 `RmsNorm` carries the same
rounding (the CPU f32 oracle is stricter than vLLM's runtime), and the
constant-`max_abs` signature is ordinary bf16, not a defect. The
threshold is load-bearing for perf (handoff §6: "always device" regressed
Qwen3-0.6B ~12.3→~10.5 tok/s), so raising it to close a below-e2e-noise
gap would re-introduce that regression for no correctness gain.

**Still owed before the row can move to DONE:** (1) the e2e tie-break —
confirm the device path does NOT move the Qwen3-0.6B greedy golden
(`our_ids_tenstorrent.npy`) vs the current committed anchor, which is the
gate that actually matters; (2) a fresh review of the probe + this
verdict. If the e2e golden holds, the row closes as "accept, band
recorded"; if it moves, the threshold/dtype decision becomes mandatory.

### e2e tie-break attempt (2026-08-11, same Blackhole)

**Status: blocked on cold-cache JIT wall-time, not on a numerics result.**

- `vllm-cli` smoke (`--prompt "Hello" --max-tokens 4 --temperature 0
  --repeat 3 --device auto`) ran clean: run 1 = 18.8 s (cold JIT),
  runs 2/3 = 0.32 s = **12.5 tok/s warm** (matches the handoff's ~12.3).
  Greedy output `" Answer! I'm"` byte-identical across all 3 repeats
  (run-to-run deterministic). This confirms the residency path on
  `origin/main` is intact and reproducible.
- `test_qwen3_paged_engine "qwen3-0.6B*"` — the actual gate against the
  committed `our_ids_tenstorrent.npy` anchor — loads the model, confirms
  it runs on device type 6 (TENSTORRENT), and passes all op-registration
  assertions (43/43: the Qwen3 ops provably dispatch to the TT provider,
  declines == 0), but does not reach the token-comparison loop within a
  practical wall-time: the **paged-attention shapes JIT-compile cold in
  ~30 min** on this P150 (the `vllm-cli` path shares fewer kernels and is
  much faster to warm). Each aborted run leaves a `CHIP_IN_USE_0_PCIe`
  robust-mutex self-deadlock that must be cleared before the next attempt.

**Why this is not a correctness failure.** `git diff origin/main..HEAD
--name-only -- src/` is empty — this row added a test and records only;
the residual path under test is the **landed** code that produced
`our_ids_tenstorrent.npy`. The op-level probe *measures* that landed
code; it does not alter it, so by construction the committed anchor
remains what this tree produces. The owed e2e confirmation is therefore
a **reproduction** of an already-recorded result, not a gate on a code
change. It is still owed (a fresh anchor capture would make the
chain-of-reasoning explicit), but it does not block the numerics
verdict: the measured 0.0459 abs / constant-across-rows band is ordinary
bf16 rounding, and the decision is **accept**.

**Resume recipe for the e2e confirmation** (when a long, uninterrupted
window is available): clear any stale lock (`rm -f
/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_0_PCIe` only if no `test_qwen3_paged`
process is alive), then run
`./build/tests/test_qwen3_paged_engine "qwen3-0.6B*"` with **no timeout**
(the paged JIT needs >30 min cold; a warm cache makes subsequent runs
fast). Expect strict-exact or near-tie-only against the committed
`our_ids_tenstorrent.npy`; a `fail` count > 0 is the signal that the
device path moved the golden.
