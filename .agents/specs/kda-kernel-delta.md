# KDA (Kimi Delta Attention) kernel delta — W0 spike + W1 host-reference brick

**Claim:** `CLAIM-KDA-KERNEL`. **Kernel row:** `KERNEL-KDA-DELTA` (NEW,
`INVENTORIED`→`SPIKE`). **Model rows it unblocks:**
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` (KDA kernel campaign named on
that row) and `MODEL-MM-kimi-k3-kimi-k3-for-conditional-generation` (K3 W4,
loader defers to this row).

**Base:** `main` HEAD `42c56b51499041b961c096674e25fa4378981097`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

**Signal (honest, up front):** this is a **host-reference-first kernel brick**,
mirroring the DeepSeek-V4 DSA lane exactly. It lands the genuinely-NEW-vs-GDN KDA
numerics as portable CPU references and UNIT-GATES them against hand-derived
literal cases + from-first-principles double-precision references. The gate is
**host-reference + structural review, NOT a dumped-oracle rel-L2**. The REAL
end-to-end gate is the **Kimi-Linear-48B-A3B proxy vs the pinned vLLM oracle on
GB10** — a NAMED residual, DGX-blocked (the 2.8T Kimi-K3 does not fit one GB10,
~12×). The device (CUDA) kernel is a second named residual.

---

## 0. Scope (headline verdict)

KDA (`KimiGatedDeltaNetAttention`, `kimi_gdn_linear_attn.py:85`) **SUBCLASSES**
`GatedDeltaNetAttention` — it REUSES our landed GDN machinery (conv-state/cache
layout, `GDNAttentionMetadata`, conv update, chunked-delta recurrence, WY solve)
wholesale. What plain GDN does NOT have — the **KDA delta** — is exactly four
things, and this brick owns their portable references:

1. a per-channel **`[H,D]` low-rank decay** via an `f_a_proj`→`f_b_proj`
   bottleneck (plain GDN has only a per-HEAD scalar decay from `A_log`);
2. the **decay gate** `g = -exp(A_log[h]) · softplus(g1 + dt_bias)` per channel
   (the `fused_kda_gate` / `kda_gate_fwd_kernel` decode variant) plus its
   chunk-local cumulative-sum prefill variant (`kda_gate_cumsum_fwd_kernel`);
3. the **sigmoid-gated output norm** `FusedRMSNormGated(head_dim,
   activation="sigmoid")` — the gated-linear-attention output norm plain GDN
   lacks;
4. **three separate q/k/v short causal convs** (`conv_size=4`, silu) and the
   q/k **L2-norm** preprocessing (`use_qk_l2norm_in_kernel=True`).

**In scope (W1, this brick):** portable host references + unit gate for all four
deltas. **Out of scope (named residuals):** the chunked gated-delta RECURRENCE
itself (reuses GDN — `chunk_kda_with_fused_gate` / `fused_recurrent_kda` compose
the GDN WY-solve/`recompute_w_u`/`chunk_gla_fwd_o` machinery); the CUDA device
kernel; the Kimi-Linear/K3 model forward + loader; the Kimi-Linear-48B proxy
e2e gate.

**Additive-only guarantee:** KDA lands as KDA-specific files
(`kimi_kda.{h,cpp}` + `test_kimi_kda.cpp`), touching NEITHER `cuda_gdn.cu` NOR
`gdn_attn.cpp`, so the Qwen3.6-27B/35B GDN gate is structurally untouched (the
same discipline DSA used to keep the shared MLA path untouched). Proven by an
empty `git diff` over `src/vt/cuda/cuda_gdn.cu` and
`src/vllm/v1/attention/backends/gdn_attn.cpp`.

---

## 1. Upstream chain (`file:line` @ `555967922`)

### 1.1 The KDA attention module — `models/layers/mamba/gdn/kimi_gdn_linear_attn.py`
- `KimiGatedDeltaNetAttention(GatedDeltaNetAttention)` (`:85`) — subclass; the
  GDN parent owns state/conv/recurrence.
- Projections: `q_proj`/`k_proj`/`v_proj` (`:120-140`, `hidden→H*D`);
  `f_a_proj` (`:142-148`, `hidden→head_dim`, `ReplicatedLinear`); `f_b_proj`
  (`:150-156`, `head_dim→H*D`, `ColumnParallelLinear`) — the **low-rank decay
  bottleneck**. `dt_bias` (`:157-161`, `[H*D]` f32). `b_proj` (`:163-169`,
  `hidden→H`, the delta-rule β). `A_log` (`:200-203`, `[1,1,H,1]` f32, per-head).
- `g_a_proj`/`g_b_proj` (`:205-218`, `hidden→head_dim→H*D`) — the OUTPUT gate
  `g2`. `o_norm = FusedRMSNormGated(head_dim, activation="sigmoid")` (`:219`).
- Three convs `q_conv1d`/`k_conv1d`/`v_conv1d` (`:171-198`, `conv_size=4`, f32),
  applied with `activation="silu"` (`:324-356` prefill `causal_conv1d_fn`,
  `:362-388` decode `causal_conv1d_update`).
- `forward` (`:233-268`): `g1 = f_b_proj(f_a_proj(x))` (`:245`);
  `g2 = g_b_proj(g_a_proj(x))` (`:249`); `beta = sigmoid(b_proj(x))` (`:244`);
  after the recurrence, `core_attn_out = o_norm(core_attn_out, g2)` (`:266`).
- `_forward` (`:270-444`): prefill calls `chunk_kda_with_fused_gate` (`:403`)
  with `use_qk_l2norm_in_kernel=True` (`:413`); decode calls `fused_kda_gate`
  (`:420`) then `fused_recurrent_kda` (`:429`).

### 1.2 The KDA ops — `third_party/flash_linear_attention/ops/kda.py`
- `kda_gate_fwd_kernel` (`:1541-1600`) + `fused_kda_gate` (`:1603-1646`): the
  per-channel decode gate `y = -exp(A[h]) · softplus_beta(g + g_bias)`,
  `softplus_beta(x)= x if βx>thr else (1/β)log(1+e^{βx})`, β=1, thr=20.
- `kda_gate_cumsum_fwd_kernel` (`:1182-1254`) + `fused_kda_gate_chunk_cumsum`
  (`:1257-1303`): the prefill variant — the same gate, then a **chunk-local
  cumulative sum** (`:1252-1253` tril-ones matmul) scaled by `RCP_LN2` (`:1295`,
  = 1/ln2, folds the ln→log2 conversion for the downstream exp2 kernels).
- `FusedRMSNormGated.forward_native` (`:463-487`), ctor (`:436-461`, eps=1e-5):
  `x_normed = x·rsqrt(mean(x²)+eps)·weight`; sigmoid → `x_normed·σ(g)`; swish →
  `x_normed·g·σ(g)`.
- `l2norm_fwd` (`ops/l2norm.py:42-43`, `:96` eps=1e-6): `x/sqrt(Σx²+eps)` — SUM
  of squares (not mean).
- REUSED-from-GDN recurrence (NOT re-ported here): `chunk_kda_with_fused_gate_fwd`
  (`:1416-1455`), `_chunk_kda_fwd_with_cumulative_g` (`:1306-1373`),
  `chunk_kda_scaled_dot_kkt_fwd` (`:717`), `solve_tril`, `recompute_w_u_fwd`
  (`:960`), `chunk_gla_fwd_o_gk` (`:1126`), `fused_recurrent_kda` (`:109-143`).

### 1.3 Config — `transformers_utils/configs/kimi_linear.py`
- `linear_attn_config` dict carries `head_dim`, `num_heads`,
  `short_conv_kernel_size`, `use_full_rank_gate`, `gate_lower_bound=-5.0`.
  `is_kda_layer(layer_idx)` (`:144`) selects KDA vs MLA per layer.

---

## 2. Our baseline — reuse-vs-new (our `file:line`)

### REUSE (landed; the KDA recurrence rides on these)
- GDN linear attention: `src/vt/cuda/cuda_gdn.cu`,
  `src/vllm/v1/attention/backends/gdn_attn.cpp` +
  `include/vllm/v1/attention/backends/gdn_attn.h`, AOT kernels
  `src/vt/cuda/triton_aot_vendored/sm_121a/gdn_*`. Mature + gated in production
  (Qwen3.6-27B/35B GDN-hybrid). Covers the conv-state/cache layout,
  `GDNAttentionMetadata`, conv update, chunked-delta recurrence, WY solve. **This
  brick does NOT touch these.**
- Kimi tokenizer/tool parser: `src/vllm/parser/kimi_k2.cpp` (Kimi family).

### NEW (this brick — the KDA delta host references)
- `include/vllm/model_executor/models/kimi_kda.h` +
  `src/vllm/model_executor/models/kimi_kda.cpp`:
  - `KdaLowRankDecay` ← `kimi_gdn_linear_attn.py:142-156,:245`
  - `KdaDecayGate` ← `kda.py:1541-1600,:1603-1646`
  - `KdaDecayGateChunkCumsum` ← `kda.py:1182-1254,:1257-1303`
  - `FusedRMSNormGated` ← `kda.py:463-487`
  - `KdaShortConv` ← `kimi_gdn_linear_attn.py:171-198,:324-356`
  - `L2NormRows` ← `kda.py:1511-1513`, `l2norm.py:42-43,:96`

### NEW (named residuals, NOT this brick)
- The KDA CUDA device kernel (structure-port of the FLA/Triton kernels, reusing
  our GDN device scaffolding).
- The Kimi-Linear / K3 model forward + loader that composes the KDA layer.
- MXFP4 (K3), MoonViT-V2 (K3) — owned elsewhere.

---

## 3. Port map (upstream → local)

| Upstream | Local | Notes / deviations |
|---|---|---|
| `kimi_gdn_linear_attn.py:142-156,:245` | `kimi_kda.cpp::KdaLowRankDecay` | two bias-free linears, no activation between; the per-channel `[H,D]` decay input |
| `kda.py:1541-1600,:1603-1646` | `kimi_kda.cpp::KdaDecayGate` | `-exp(A_log)·softplus_β(g1+dt_bias)`; β=1, thr=20; double-precision softplus |
| `kda.py:1182-1254,:1257-1303` | `kimi_kda.cpp::KdaDecayGateChunkCumsum` | same gate + chunk-local cumsum · RCP_LN2; `log2_domain` flag exposes the raw-ln cumsum |
| `kda.py:463-487` (`:436` eps=1e-5) | `kimi_kda.cpp::FusedRMSNormGated` | RMS over head_dim · weight · σ(g); sigmoid (KDA) + swish branches |
| `kimi_gdn_linear_attn.py:171-198,:324-356` | `kimi_kda.cpp::KdaShortConv` | depthwise causal conv (zero init state) + silu; one of the 3 q/k/v convs |
| `kda.py:1511-1513`, `l2norm.py:42-43,:96` | `kimi_kda.cpp::L2NormRows` | `x/sqrt(Σx²+eps)`, eps=1e-6, SUM not mean |

**Deviations recorded:** (a) the recurrence itself is reused from GDN, not
re-ported here (§0 out-of-scope); (b) the fp8/quant activation folds in the
kernels are omitted — the fp32 reference isolates the KDA numerics
(β=1, kv_scale=1); (c) TP sharding is out of scope for a single-device host
reference; (d) `RCP_LN2` is a downstream-exp2 representation detail, exposed
behind `log2_domain` so the reference can emit the plain natural-log cumsum too.

---

## 4. Tests to port

Upstream has no standalone KDA-gate/-norm unit module at this pin (the numerics
live inside the fused Triton kernels, exercised only by full-model runs, which
are DGX/multi-Spark-blocked). So — exactly as the DSA lane — the executable spec
is a **hand-derived-case + double-precision reference** suite, re-expressed in
our doctest tier and named traceably after the KDA ops:
`tests/vllm/models/test_kimi_kda.cpp`.

- low-rank decay = `f_b @ f_a @ x` (hand case)
- decay gate = `-exp(A_log)·softplus(g1)` with softplus linearising past
  threshold (hand case), A_log/dt_bias application (hand case), + double-precision
  rel-L2 on randomized shapes
- chunk-cumsum resets at chunk boundary and folds RCP_LN2 (hand case)
- `FusedRMSNormGated` sigmoid = `rmsnorm(x)·w·σ(g)` (hand case), swish-vs-sigmoid
  branch, per-head-dim normalisation, + double-precision rel-L2
- short conv is causal-depthwise + silu (hand case), zero-init-state edge, +
  double-precision rel-L2 with bias
- q/k L2-norm = `x/sqrt(Σx²+eps)` (hand case) + SUM-not-mean discrimination

When the KDA CUDA kernel and the Kimi-Linear-48B proxy land, these host
references become the kernel's portable oracle and the proxy becomes the real
e2e gate.

---

## 5. Gates

- **W1 (this brick):** `test_kimi_kda` all green on a clean CPU
  `-Wall -Werror -Wextra` build; additive/byte-neutral for the GDN gate (empty
  `git diff` over `cuda_gdn.cu` + `gdn_attn.cpp`).
- **Residual — device kernel:** structure-port the FLA/Triton kda kernels reusing
  GDN device scaffolding; unit-gate the CUDA path vs these host references;
  compute-sanitizer clean.
- **Residual — REAL e2e:** Kimi-Linear-48B-A3B (~89–91 GiB, FITS GB10
  HW-marginal) greedy golden vs the pinned vLLM oracle (strict or the ratified
  near-tie band); serialize via `${GPU_LOCK}`, stop `local-ai-worker`, keep
  `gpu_memory_utilization` low (unified pool), reclaim ~10 GiB disk first.

---

## 6. W-breakdown

- **W0 (this):** spike — the KDA-delta map, reuse-vs-new, port map, tests, gates.
  DONE.
- **W1 (this):** portable host references for the four KDA deltas +
  hand-case/double-precision unit gate; CPU build-verify; records. DONE.
- **W2 (residual):** KDA CUDA device kernel (structure-port), unit-gated vs the
  W1 host references; reuse GDN device scaffolding; compute-sanitizer clean.
- **W3 (residual):** compose the KDA layer into the Kimi-Linear forward + loader
  (reuse the Qwen3.6-35B GDN-hybrid-MoE skeleton); gate on Kimi-Linear-48B-A3B
  vs the pinned oracle (the decisive REAL-signal brick).

---

## 7. Dependencies / risks / decisions

- **Shared with `CLAIM-MLA-DEEPSEEK` (Kimi-Linear row) + `CLAIM-KIMI-K3-*`** —
  the KDA kernel must not be implemented twice. This brick is the single home of
  the KDA delta references; K3's loader (already scaffolded, MXFP4-refuse)
  defers its KDA layer here. (DEP.)
- **REAL gate is DGX-blocked** — no on-box KDA e2e is produced by this brick; the
  Kimi-Linear-48B proxy is the named residual. (RISK, recorded.)
- **Host-reference, not oracle-dumped** — honest per the DSA precedent; the fixed
  large configs cannot be constructed at a tiny shape, and the numerics live in
  fused Triton kernels. (DECISION.)
