# Laguna-S-2.1 (`LagunaForCausalLM` / `laguna`) — W4 CUDA-build verification + W5-blocker record

**Row:** `MODEL-TEXT-laguna-laguna-for-causal-lm` — stays **ACTIVE** (NOT RUNNABLE).
**Base:** W4 commit `8788d68d` (git-archived to the DGX, md5 `14ec2435…a32f`, verified).
**Date:** 2026-07-31. **HW:** dgx.casa GB10 (sm_121a), 119 GiB unified.

## STEP 1 — CUDA build (the prior-agent stall point): CLEAN, no fixes needed
- Configured `RelWithDebInfo -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
  -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DVLLM_CPP_TRITON=ON`; `configure.log`
  `cutlass not found`=0.
- `cmake --build build-cuda -j8` → **BUILD_RC=0**, 1059/1059 targets, **0 error: / 0
  warning:** in build.log. All four laguna TUs compiled (laguna.cpp / laguna_ops.cpp /
  laguna_registry.cpp / laguna_weights.cpp `.o`). The W3 code was CPU-only; this proves
  the W4 tree also builds clean under the full CUDA fast-path config. No W4 build error
  to fix. (Prior agent stalled by BACKGROUNDING the build; the build itself is fine.)

## STEP 2/3 — real-GGUF forward + gate: BLOCKED, the GGUF path is UNIMPLEMENTED (W5)
The task framing ("only build + forward + gate remain; run the W4 tower materialization")
does not match the committed code. The W4 spec's own "What REMAINS (W5 close)" is unbuilt:

- `LoadLagunaFromGguf` (laguna_weights.cpp:250) is a **`VT_CHECK(false)` throwing stub**
  ("GGUF keep-quant tower materialization is a W3 residual"). `LoadLagunaForCausalLMWeights`
  likewise throws. `test_laguna_scaffold` explicitly asserts these loaders THROW.
- There is **no run entrypoint** (no `laguna_gen` example; no engine/test path that loads
  the real GGUF into the arch). `LagunaModel::Forward` is a reference **f32** path
  (`ReadF32`+`MatmulNK`) that cannot consume keep-quant blocks and is fed only synthetic
  weights in the unit test. So OUR engine cannot yet run Laguna on the real bytes — this is
  an UNBUILT code path, not a numeric bug in a working forward.

### Additional blockers discovered this increment
1. **No multi-shard GGUF reader.** `GgufFile::Open` opens ONE file; the UD-Q4_K_XL model
   ships as 3 shards (shard-1 = 3.6 MB header only). ds4's GGUF was single-file, so this
   never surfaced. Fix is loader-local (open all 3 shards, route each tensor to its shard —
   no reader change, no merge, no extra disk).  llama-gguf-split `--merge` was tried and is
   NOT viable here: it ENOSPC-failed (the box filled).
2. **DGX disk pressure.** The full test-inclusive `build-cuda` tree is **137 GiB**; with the
   69 GiB model the box hit 100 %. (Build tree cleaned at end of this run — box restored.)

## W5 plan (fully scoped — all templates identified, ~600 lines, mechanical)
- `LoadLagunaFromGguf`: mirror ds4 `V4GgufCtx` — open the 3 shards, `Mw` (attn q/k/v/o/gate
  Q8_0, dense-L0 ffn, shared Q8_0, lm_head → `OwnGgufQuantBlocks` keep-quant) + `Sew`
  (`ffn_{gate,up,down}_exps` [E,out,in] kept COMPRESSED) + `Vec` (norms / `ffn_gate_inp`
  router / `exp_probs_b.bias` / `token_embd` → f32). Build `LagunaParams` from the GGUF KV
  (keys confirmed: `laguna.{block_count,embedding_length,feed_forward_length,
  attention.head_count[ARRAY per-layer],attention.head_count_kv,attention.key_length,
  expert_{count,used_count,feed_forward_length,shared_feed_forward_length,weights_scale,
  gating_func},leading_dense_block_count,attention.sliding_window,attention.
  layer_norm_rms_epsilon,rope.{freq_base,freq_base_swa,dimension_count,dimension_count_swa,
  scaling.{type,factor,original_context_length,yarn_attn_factor,yarn_beta_fast,
  yarn_beta_slow}}}`). Derive `layer_types` from the per-layer head_count array
  (48→full_attention / 72→sliding_attention).
- `LagunaForwardGguf(w, q, tokens, positions, logits_idx)`: COPY `LagunaModel::Forward`,
  swap the ~9 `MatmulNK(ReadF32(w))` sites for ds4-style `Gemm`/`GemmRowSlice`
  (`vt::MatmulBT`, dispatches keep-quant → CPU/CUDA `kMatmulBTQuant`). All glue (dual-RoPE,
  per-head QK-RMSNorm, sigmoid ungrouped router + `moe_routed_scaling_factor` 2.5, softplus
  out-gate, shared expert) stays identical to the unit-gated reference.
- `examples/laguna_gen/main.cpp`: mirror `examples/deepseek_v4_gen/main.cpp` (open GGUF,
  keep-quant load, greedy full-recompute loop, argmax, tokenizer decode) + CMake target.
- Gate: greedy "The capital of France is" vs the llama.cpp-Poolside reference on the identical
  UD-Q4_K bytes (llama.cpp gives "…Paris." @ 27.8 tok/s). Coherence = gate (a); token/near-tie
  vs llama.cpp = gate (b).

## Decision
Row stays **ACTIVE**. W4's build is CUDA-clean; the real-GGUF keep-quant forward + dual-oracle
gate is the W5 close (unchanged from the W4 spec). Not fabricated as done.
