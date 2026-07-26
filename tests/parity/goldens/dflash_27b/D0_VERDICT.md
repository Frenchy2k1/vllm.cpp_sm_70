# DFlash D0 — oracle readiness verdict: **UNBLOCKED, golden captured** (measured 2026-07-26)

`CLAIM-DFLASH-D0D1`. Base `origin/main` `bc415a3e` (the pin-advance commit). Pinned
vLLM `/home/mudler/_git/vllm` @ `555967922`; dgx oracle `~/venvs/vllm-oracle` ->
`vllm-oracle-next` (**vLLM 0.26.0.dev0+g5559679**, from-source sm_121a build).

**This supersedes the 2026-07-26 ORACLE-BLOCKED verdict on vLLM 0.25.0**
(`d0_blocked_traceback.txt`, retained as evidence). The pin advance to 0.26.0.dev0
resolves upstream **vllm#40898** (mixed sliding/full attention `layer_types` for
DFlash drafts, under `VLLM_USE_V2_MODEL_RUNNER=1`): the z-lab mixed-attention draft
now CONSTRUCTS and the propose/verify/rejection loop RUNS.

## The decisive check (RUN-verified, not config-construct)

Ran on dgx under `flock $HOME/gpu.lock`, GPU sole-owner idle, with the new oracle:

```python
# env VLLM_USE_V2_MODEL_RUNNER=1   (REQUIRED for the vllm#40898 mixed-attn path)
LLM(model="unsloth/Qwen3.6-27B-NVFP4",            # NVFP4 parity target
    trust_remote_code=True,
    speculative_config={"method": "dflash",
                        "model": "z-lab/Qwen3.6-27B-DFlash",
                        "num_speculative_tokens": 16,   # = block_size 16
                        "max_model_len": 4096},
    enforce_eager=True, gpu_memory_utilization=0.30, max_model_len=4096,
    max_num_seqs=4, disable_log_stats=False,
    limit_mm_per_prompt={"image": 0, "video": 0})   # skip vision-encoder profiling
```

**NO draft attention backend is pinned** (per the W0-W2 finding): auto-selection
resolves `decode_backend=flashinfer-native, kv_cache_dtype=torch.float8_e4m3fn,
arch=sm121` — NOT FLASH_ATTN. The anticipated "DFlash needs FLASH_ATTN" soft risk
did NOT materialize; flashinfer-native serves the non-causal draft on GB10.

### MEMORY — the vision-encoder OOM reboot (recorded)

The FIRST D0-redo attempt (gpu_memory_utilization=0.55, mm ENABLED) **HARD-REBOOTED
dgx** — the multimodal `Qwen3.6-27B-NVFP4` (`ForConditionalGeneration`) profiles the
vision encoder cache with one max-feature-size dummy image at startup, which spiked
the 119 GiB UNIFIED pool (the exact `gb10-unified-memory-oom-reboots-box` hazard).
`limit_mm_per_prompt={"image":0,"video":0}` (DFlash is a text path) + gpu_util 0.30
fixed it: startup used 26.2 GiB (weights) + KV 8.85 GiB, well within the pool.

## RESULT — the DFlash drafter is ALIVE (acceptance > 0; the dead-drafter trap cleared)

Greedy, `max_tokens=32`, fixed 4-prompt battery (mirrors the MTP gate prompts).
Acceptance from `vllm:spec_decode_num_drafts` / `_num_accepted_tokens`
(`acceptance_len` = 1 + accepted/drafts, per `test_spec_decode.py::compute_acceptance_len`):

| prompt | acceptance_len | drafts | accepted |
|---|---|---|---|
| "The capital of France is" | **2.214** | 14 | 17 |
| "def fibonacci(n):" | **8.800** | 5 | 39 |
| "Q: What is 17 * 23?\nA:" | **4.750** | 8 | 30 |
| "The three laws of robotics are" | **4.571** | 7 | 25 |

acceptance_len > 1 on EVERY prompt = drafts are genuinely accepted (a dead drafter
gives exactly 1.0). Content-dependence matches the spec §5 B5 note (code >> prose).
Goldens: `dflash_27b_spec_on.json` (tokens + acceptance), `dflash_27b_spec_off.json`
(baseline greedy). Coherent decoded text (Paris; a fibonacci function; the
distributive property for 17×23; the three laws) confirms a real run.

## Gate FORM (by measurement) — STRICT, MODE-MATCHED (NOT the MTP three-way identity)

Two measured facts set the D5 gate form:

1. **vLLM-DFlash-ON greedy is RUN-DETERMINISTIC.** K>=3 spec-on runs (the stats
   golden + `k3/` + `d1/`) are TOKEN-IDENTICAL and acceptance-identical on all 4
   prompts. (Run `k2/` failed engine init with "No available memory for the cache
   blocks" — a host-RAM contention artifact from a concurrent CUDA build, NOT a
   DFlash issue; the uncontended runs all succeed.)
2. **vLLM-DFlash-ON greedy is NOT token-identical to vLLM-spec-OFF greedy** (3 of 4
   prompts diverge). This is expected for a k=16 BLOCK verify: the target processes
   the (1+k)-token block in ONE batched forward, whose logits differ from
   sequential single-token decode at bf16 near-ties, flipping the argmax. Greedy
   spec-decode is output-equivalent only in the no-near-tie limit.

**Therefore the DFlash D5 correctness gate is NOT the MTP-style three-way identity**
(`our-ON == our-OFF == vLLM-ON`), because vLLM itself has `vLLM-ON != vLLM-OFF`.
The D5 gate is **STRICT and MODE-MATCHED**: `our-DFlash-ON == vLLM-DFlash-ON` greedy,
token-for-token (single-request, deterministic — the golden here is the reference),
plus nonzero acceptance; the `spec-OFF byte-identical SACRED` obligation is a
SEPARATE inertness gate (unchanged). If a later run reveals near-tie run-nondeterminism
at c>1, the `near-tie-distributional-gate` applies as in MTP I7 (exact identity at c1).

## D0 deliverables — all cleared

- Draft `z-lab/Qwen3.6-27B-DFlash` cached (3.3 GiB, `target_layer_ids=[1,16,31,46,61]`);
  the oracle log confirms `eagle3_utils.py:28 Using Eagle3 auxiliary layers from
  config: (2, 17, 32, 47, 62)` — i.e. `target_layer_ids + 1` (the eagle3 shift),
  exactly the keys DF-AUX-TAPS captures.
- `num_speculative_tokens` resolves to **16** (= block_size).
- Capture tool: `scripts/spec/d0_dflash_oracle_capture.py` (extended: `disable_log_stats=False`
  for the acceptance liveness proof, `limit_mm_per_prompt` + gpu_util 0.30 for the
  unified-pool OOM guard).

Evidence: `dflash_27b_spec_{on,off}.json` (this dir); `k3/`, `d1/` determinism runs
+ `spec_on*.log`, `sanitizer_d1.log`, `mtp_e2e.log`, `sacred_27b.log` on dgx under
`~/work/dflash-d0d1/`.
