# DFlash D0 — oracle readiness verdict: **ORACLE-BLOCKED on vLLM 0.25.0** (measured 2026-07-26)

`CLAIM-DFLASH-D0`. Base `origin/main` `8a379182`. Pinned vLLM `/home/mudler/_git/vllm`
@ `e24d1b24`; dgx oracle `~/venvs/vllm-oracle` -> `vllm-oracle-v0.25.0-stage` (vLLM 0.25.0).

## The decisive check (RUN-verified, not config-construct — per [[oracle-gateability-model-runs-not-config-constructs]])

Ran, on dgx under `flock $HOME/gpu.lock`, GPU sole-owner idle:

```python
LLM(model="unsloth/Qwen3.6-27B-NVFP4",           # the spec's NVFP4 parity target (25 GiB; safer than
    trust_remote_code=True,                       #   the 69 GiB bf16 Qwen/Qwen3.6-27B under the OOM-reboot rule)
    speculative_config={"method": "dflash",
                        "model": "z-lab/Qwen3.6-27B-DFlash",
                        "num_speculative_tokens": 16,   # block_size 16 -> the e2e-fixture value (NOT 15)
                        "max_model_len": 4096},
    enforce_eager=True, gpu_memory_utilization=0.55, max_model_len=4096, max_num_seqs=4)
```

Config keys are grounded in the pinned oracle
(`tests/v1/e2e/spec_decode/test_spec_decode.py::dflash_config`, `config/speculative.py`).

## RESULT — vLLM 0.25.0 does NOT serve DFlash for the z-lab Qwen3.6 drafts.

The DFlash speculative_config is **accepted** (method resolves to `"dflash"`,
`SpeculativeConfig(method='dflash', model='z-lab/Qwen3.6-27B-DFlash', num_spec_tokens=16)`,
`parallel_drafting=True`), and the **NVFP4 target loads fully** (FLASHINFER attn,
FlashInferCutlassNvFp4 GEMM, Triton/FLA GDN). But constructing the **DFlash draft model**
aborts, BEFORE any propose/verify/rejection loop runs — so there are NO emitted tokens and
NO acceptance rate to capture:

```
File ".../vllm/model_executor/models/qwen3_dflash.py", line 93, in _resolve_layer_attention
    raise NotImplementedError(
NotImplementedError: DFlash does not yet support mixed sliding/full attention via layer_types;
see https://github.com/vllm-project/vllm/issues/40898.
-> RuntimeError: Engine core initialization failed.
```

(Full trace: `d0_blocked_traceback.txt`, this dir.)

### Root cause (measured, verified in pinned source)

`qwen3_dflash.py:_resolve_layer_attention` (`e24d1b24`, identical in the installed 0.25.0)
hard-raises when the draft's `layer_types` mixes sliding and full attention:

```python
if any_sliding and not all_sliding:
    raise NotImplementedError("DFlash does not yet support mixed sliding/full attention ...")
```

The **z-lab/Qwen3.6-27B-DFlash** draft config has
`layer_types = [sliding_attention ×4, full_attention ×1]` -> `any_sliding=True,
all_sliding=False` -> **raises**. The **z-lab/Qwen3.6-35B-A3B-DFlash** draft is
`[sliding_attention ×5, full_attention ×1]` -> **also mixed -> also blocked**.

Only "standard" all-full-attention drafts (`z-lab/Qwen3.5-9B-DFlash`) or all-SWA
(`use_swa`) drafts construct on 0.25.0. Both of OUR gate-model drafts are the
mixed-attention kind the pin refuses. This is the D5 "no oracle -> honestly blocked"
case — same class as Gemma-4 (no `gemma4` in transformers) and OLMo-3 (nested rope).

### Draft download / config (D0 deliverable #1 — CLEAR)

`z-lab/Qwen3.6-27B-DFlash`: **ungated + downloadable** (unauthenticated HF fetch OK),
arch `DFlashDraftModel` -> registry `("qwen3_dflash","DFlashQwen3ForCausalLM")` confirmed,
`config.json` present, `model.safetensors` **3.46 GiB** on disk (= 1.73 **B params** bf16;
the spec's "1.73 GB" is the param count, not the file size). 5 layers (4 SWA-2048 + 1 full),
hidden 5120, block_size 16, mask_token_id 248070.
`target_layer_ids = [1, 16, 31, 46, 61]` (5 taps), `num_target_layers = 64`.

**Reconciliation:** the D0 brief's "`[1,6,11,16,22,27,32,37]` (8 taps / 40 target layers)"
is the **35B-A3B** draft's config (verified: 35B = `[1,6,11,16,22,27,32,37]`, 40 layers,
mask 248077), NOT the 27B. The **spec §2 table is correct**; the brief transposed the two
drafts' tap sets. Target `Qwen3.6-27B` NVFP4 has `num_hidden_layers=64` -> matches the
draft's `num_target_layers=64`.

## What would UNBLOCK DFlash

A vLLM version **> 0.25.0** that resolves upstream **vllm#40898** (per-layer causal
metadata + multiple KV-cache groups for mixed sliding/full DFlash drafts) — i.e. a **pin
advance**, exactly the Gemma-4/OLMo-3 unblock mechanism. There is no all-full-attention
DFlash draft published for either Qwen3.6 gate model, so the config-swap fallback the spec
§0/§8 hoped for ("worst case a config/version bump") does not exist today — it is a
genuine version gate.

## Consequence

**STOP.** The D-series (D1 `DF-AUX-TAPS` -> D2 `DF-DRAFT-MODEL` -> D3 `DF-DRAFT-KV-PREP` ->
D4 `DF-ENGINE-INTEGRATION` -> D5 correctness -> D6 throughput) is **oracle-BLOCKED**: the D5
three-way gate (our-DFlash-ON == our-spec-OFF == vLLM `--speculative-config dflash`) has no
vLLM arm on the pin, so the drafter cannot be verified. Do NOT implement the drafter blind.
No golden fabricated. The capture script `scripts/spec/d0_dflash_oracle_capture.py` is the
ready D0 tool for re-running the instant the pin advances.
