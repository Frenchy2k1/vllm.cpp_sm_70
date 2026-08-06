# Spike: the Parakeet / FastConformer audio encoder on `vt::`

Status: SPIKE (records only). No implementation row may go `ACTIVE` off this
document until the work breakdown below is claimed in `coordination.md`.

Motivating question: can `parakeet.cpp` (the LocalAI team's standalone C++/ggml
NeMo Parakeet ASR engine) be folded into vllm.cpp by supporting the same models,
rather than re-basing `parakeet.cpp` onto `vt::`? This spike answers what is in
scope, what the port actually costs, and what is NOT vLLM-defined.

## Scope

**In.** The conformer audio ENCODER family and the CTC head, on `vt::`, CPU and
CUDA:

| Row ID | What |
|---|---|
| `KERNEL-CPU-CONV2D-SUBSAMPLE` | Conv2d subsampling stack (the encoder front end) as a real `vt::` op, CPU + CUDA |
| `KERNEL-DEPTHWISE-CONV1D` | Non-causal depthwise Conv1d (the conformer convolution module) |
| `KERNEL-ATTN-RELPOS` | Relative-position multi-head attention (Transformer-XL style), the conformer attention |
| `MODEL-AUDIO-PARAKEET-ENCODER` | `ParakeetEncoder` + `ParakeetForCTC` end to end |

**Out, and why.**

- **RNN-T / TDT transducer decode.** `parakeet.cpp`'s joint network, prediction
  network, greedy transducer search and TDT duration heads have NO upstream in
  either vLLM or HF transformers (grep-verified: no `rnnt`, `transducer` or
  `joint_net` symbol in `vllm/model_executor/models/parakeet.py`,
  `conformer_encoder.py`, or `transformers/models/parakeet/`). Adding them is a
  genuine PRODUCT call, in the same class as the conscious llama.cpp and SGLang
  source deviations, and is deliberately not decided here.
- **Streaming / cache-aware encoding and `<EOU>`/`<EOB>`.** NeMo-only, same
  reasoning.
- **The mel front end** beyond what `ParakeetExtractor` already specifies.
- **x86 SIMD quant (G5).** Unrelated row, and the 2026-08-06 op-dispatch profile
  ranks it below these.

**Dispatch behavior.** The three kernel rows register as ordinary `vt::` ops
through `op_provider`, so they take the portable CPU tier by default and a CUDA
provider when one is registered, exactly like every existing op. No new seam.

## Upstream chain

This is the section that changed the plan, so it is stated precisely.

**vLLM does NOT implement the Parakeet encoder.** It wraps HuggingFace's:

- `vllm/model_executor/models/parakeet.py:37` imports
  `from transformers import ParakeetEncoder as HFParakeetEncoder`.
- `parakeet.py:62` (`ProjectedParakeet.__init__`) instantiates
  `HFParakeetEncoder(self.config)`; `:66-75` (`forward`) calls it and applies a
  vLLM-native `ParakeetProjection` (`parakeet.py:27`).
- `parakeet.py:138` `ParakeetExtractor` owns the mel feature extraction.
- Config: `vllm/transformers_utils/configs/parakeet.py:8` `ParakeetConfig`
  extends HF `ParakeetEncoderConfig`.
- It is NOT a standalone registry architecture: it is the audio encoder
  component of `nano_nemotron_vl.py` (`registry.py:511-513`,
  `NemotronH_Nano_VL_V2` / `NemotronH_Nano_Omni_Reasoning_V3` /
  `NemotronH_Super_Omni_Reasoning_V3`).

**So the real mirror source is HF transformers**, which is squarely inside "the
whole execution chain" T0 already binds us to. transformers 5.3.0,
`transformers/models/parakeet/modeling_parakeet.py`:

| Upstream class | Line |
|---|---:|
| `ParakeetEncoderRelPositionalEncoding` | 51 |
| `ParakeetEncoderFeedForward` | 101 |
| `ParakeetEncoderConvolutionModule` | 116 |
| `ParakeetEncoderAttention` | 259 |
| `ParakeetEncoderSubsamplingConv2D` | 357 |
| `ParakeetEncoderBlock` | 426 |
| `ParakeetEncoder` | 549 |
| `ParakeetForCTC` | 675 |

`ParakeetForCTC` matters: **the CTC decode path DOES have an upstream**, so CTC
is mirror work, not a scope deviation. Only the transducer heads are not.

**A second, closer mirror already exists in vLLM's own tree.**
`vllm/model_executor/models/conformer_encoder.py` is a NATIVE vLLM conformer
(`Conv2dSubsampling:18`, `Swish:50`, `RelPositionalEncoding:55`,
`ConformerFeedForward:81`, `RelPosMultiHeadAttention:170`,
`ConformerConvolution:220`, `RelPosEmbConformerBlock:265`,
`ConformerEncoder:289`). Its docstring says it is "shared by FireRedASR2 and
FireRedLID", so it is a DIFFERENT model family, but it is structurally the same
conformer and it is vLLM-native rather than HF-delegated. It needs the identical
three kernels.

Runtime trace plan: `ParakeetEncoderAttention` selects its attention path
dynamically in HF; before claiming parity, dump the actual module path with a
torch hook on a real checkpoint rather than reading the source, per T0 "trace
the execution, not just the code".

## Our baseline

Honest gaps, verified 2026-08-06 by grep and by the parakeet.cpp probe in
`benchmarks/vt_probe/` of that repo:

| Need | `vt::` today | Gap |
|---|---|---|
| Conv2d (subsampling) | NONE as a device op. The only Conv2d in the tree is a host `std::vector<float>` loop, `src/vllm/model_executor/models/gemma4_audio.cpp:92` `Conv2dK3S2P1`, a correctness reference for a small audio prefix | Full kernel, CPU + CUDA |
| Depthwise Conv1d, non-causal | `CausalConv1dFwd` / `Update` / `SpecUpdate` only (`include/vt/ops.h:787-807`), Mamba-shaped and causal | Non-causal variant |
| Relative-position attention | NONE. Every attention path is RoPE plus paged or flash KV (`ops.h:1932-2292`) | Full kernel |
| Log-mel front end | NONE | Extractor port |
| GEMM, LayerNorm, elementwise | Present and tuned | None |

Measured context (parakeet.cpp probe, same shapes, both runtimes), CORRECTED
2026-08-06:

- **GB10 CUDA, 16-bit:** `vt::MatmulBT` is **1.4x to 3.5x FASTER** than ggml at
  this encoder's real GEMM shapes (2.03x weighted, all 12 cases numerically
  verified against an f64 host reference).
- **GB10 Arm CPU, q8_0 with the G7 repack tier engaged:** `vt` is **1.38x to
  4.97x FASTER** than ggml (628 to 1962 GFLOP/s vs 394 to 456).
- **GB10 Arm CPU, 16-bit (f32 activations, f16 weight):** ggml is ahead
  **1.14x to 2.00x** (403 to 444 vs 141 to 242 GFLOP/s).
- **x86 CPU:** ggml ahead everywhere, because `QuantRepackEligible` is false off
  i8mm and G5 is unimplemented, so `vt` runs a portable scalar tier.

An earlier revision of this spike said ggml wins CPU outright. That was a
benchmark defect: `Tensor.repacked` was left false, and since the G6 mmla tier
only engages when M AND N are both even, the conformer's odd M (131, 261, 1)
fell through to the portable tier. Production repacks at load
(`qwen3_5_gguf_weights.cpp:104-107`), so the corrected numbers above are the
binding ones.

## Port map

| Upstream | Local | Notes |
|---|---|---|
| `modeling_parakeet.py:357` `ParakeetEncoderSubsamplingConv2D` | new `src/vt/cpu/cpu_conv2d.cpp`, `src/vt/cuda/cuda_conv2d.cu`, op `kConv2d` | Replaces the `gemma4_audio.cpp:92` host loop, which becomes a caller |
| `:116` `ParakeetEncoderConvolutionModule` | new depthwise path in `cpu_conv1d`/`cuda_conv1d`, op `kDepthwiseConv1d` | Sibling of the existing `CausalConv1d`, NOT a modification of it |
| `:259` `ParakeetEncoderAttention` | new `kAttentionRelPos` | Encoder self-attention, no KV cache, no paging |
| `:51` `ParakeetEncoderRelPositionalEncoding` | host-side table, reuses existing ops | No new kernel |
| `:101` `ParakeetEncoderFeedForward` | existing `MatmulBT` + `LayerNorm` + activation | No new kernel |
| `:426` `ParakeetEncoderBlock`, `:549` `ParakeetEncoder` | new `src/vllm/model_executor/models/parakeet_encoder.cpp` + header | Must route the three MUST-route seams or take a conscious allowlist entry |
| `:675` `ParakeetForCTC` | new CTC head + greedy collapse | Upstream-defined, in scope |
| `vllm/.../parakeet.py:27` `ParakeetProjection` | same file | vLLM-native, small |
| `vllm/.../parakeet.py:138` `ParakeetExtractor` | new mel extractor | Deviation to record: ours is C++, no torchaudio |

**Recorded deviation.** We port HF's module where vLLM delegates to it. Every
ported file carries the upstream-commit header per `discipline.md`, citing the
transformers version and path, not a vLLM path, because that is the honest
provenance.

## Tests to port

| Upstream | Local tier | Note |
|---|---|---|
| `transformers/tests/models/parakeet/test_modeling_parakeet.py` | `tests/vllm/models/test_parakeet_encoder.cpp` | The executable spec for the encoder |
| per-op numerics | `tests/vt/test_ops_conv2d.cpp`, `test_ops_conv1d_depthwise.cpp`, `test_ops_attn_relpos.cpp` | Gate is byte-identity vs an in-test scalar reference, matching `test_ops_matmul_elem.cpp` discipline, NOT NMSE |
| e2e CTC | `tests/vllm/models/test_parakeet_ctc_engine.cpp` | Greedy transcript vs the HF oracle on a fixed clip |

Initially blocked: nothing. All three kernels are unit-testable on CPU with no
checkpoint.

## Gates

- **Correctness.** Per-op byte-identity against an independent in-test scalar
  reference across dtype x shape x thread-count, including ragged K/N, on x86-64
  AND dgx aarch64. This is the E1-E4 / G6 bar and it is met on x86 even though
  x86 is void for timing.
- **e2e.** Greedy CTC transcript token-exact against the HF `ParakeetForCTC`
  oracle on a pinned clip, on CPU and CUDA, tokens byte-identical between them.
- **Performance.** GB10 only (`GATE_HOST=dgx.casa`), one `flock $HOME/gpu.lock`,
  same binary, 3 reps, medians, idle box. Floor: **parakeet.cpp on the same
  clip and box**, since that is the incumbent and vLLM does not run this
  standalone. No x86 timing number is binding, per
  `CLAIM-KERNEL-CPU-ELEM-GEMM-1`.
- **Memory.** Peak RSS against parakeet.cpp, same clip.
- **Architectures/backends.** CPU + CUDA `sm_121a` in the first pass. Metal and
  Vulkan explicitly deferred.

Exact commands go in the implementation rows, not here; a spike does not run
gates.

## Dependencies

- Rows: none blocking. `KERNEL-GEMM-CPU-ELEM` and `QUANT-GGUF-CIQ-GEMM` are
  siblings, not prerequisites.
- Toolchain: existing. No new third-party.
- Models: an `nvidia/parakeet-*` checkpoint plus the HF oracle for the token
  gate. Licences are permissive (CC-BY-4.0 family) but must be confirmed per
  checkpoint before any is vendored.
- Hardware: dgx.casa for the speed gate. CPU-only boxes suffice for correctness.
- Data: one pinned audio clip, committed or hash-pinned.

## Work breakdown

Small, non-overlapping, claimable in parallel. Each owns disjoint files.

| # | Row | Owns | Parallel with |
|---|---|---|---|
| P1 | `KERNEL-CPU-CONV2D-SUBSAMPLE` | `cpu_conv2d.*`, `cuda_conv2d.*`, `test_ops_conv2d.cpp`, op registration | P2, P3 |
| P2 | `KERNEL-DEPTHWISE-CONV1D` | depthwise files + its test | P1, P3 |
| P3 | `KERNEL-ATTN-RELPOS` | relpos attention files + its test | P1, P2 |
| P4 | `MODEL-AUDIO-PARAKEET-ENCODER` | `parakeet_encoder.{h,cpp}`, extractor, CTC head, model tests, matrix rows | after P1-P3 |
| P5 | FireRedASR2 `ConformerEncoder` | reuses P1-P3 unchanged | after P1-P3 |

P5 is listed because it is the cheap proof that the three kernels are general
rather than Parakeet-shaped, and because it is a native-vLLM mirror rather than
an HF one.

## Risks and decisions

**Product calls, for the human:**

1. **Does the transducer (RNN-T / TDT) belong in vllm.cpp at all?** It has no
   upstream anywhere in the chain. Folding parakeet.cpp in FULLY requires it;
   folding in the encoder plus CTC does not. Until this is decided, P1-P5 stand
   on their own and parakeet.cpp keeps the transducer.
2. **Is CPU a target for this encoder, and at which dtype?** Measured, and the
   answer differs by dtype rather than being uniform. On Arm, `vt` is 1.38x to
   4.97x FASTER than ggml for q8_0 weights (i8mm repack tier) and 1.14x to 2.00x
   SLOWER for 16-bit weights, where it has no integer tier to reach for. So a
   quantized Arm CPU encoder is attractive, an f16 Arm CPU encoder is a
   regression, and x86 is a regression at every dtype until G5 lands. If the
   fold targets CPU, it should ship q8_0 as the CPU default rather than f16.

**Engineering risks, not reopened:**

- HF's encoder may change shape between transformers releases. Pin the version
  in the file headers and gate the drift.
- `ParakeetEncoderAttention` dispatches dynamically; source reading is not
  sufficient evidence of what runs (T0).

**Not a risk:** the three kernels are well-understood, have exact upstream
references, and are unit-testable without a checkpoint.
