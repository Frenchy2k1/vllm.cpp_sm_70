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

**In, added 2026-08-07 (row P6) after the correction below:**

| Row ID | What |
|---|---|
| `MODEL-AUDIO-PARAKEET-TRANSDUCER` | `ParakeetForRNNT` + `ParakeetForTDT`: LSTM prediction network, joint network, TDT duration head, greedy transducer decode |

**CORRECTION, 2026-08-07: the transducer DOES have an upstream.** The
paragraph this replaces said the RNN-T / TDT stack had "NO upstream in either
vLLM or HF transformers", and moved it out of scope as a product call. That was
measured against the transformers version installed on the spike box, **5.3.0,
which ships only `ParakeetForCTC`**: the grep was correct about that tree and
wrong about upstream. Current transformers `main` implements the whole stack:

| Upstream class | File | Line |
|---|---|---:|
| `ParakeetRNNTDecoder` (LSTM prediction network) | `modeling_parakeet.py` | 831 |
| `ParakeetRNNTJointNetwork` | `modeling_parakeet.py` | 879 |
| `ParakeetForRNNT` | `modeling_parakeet.py` | 922 |
| `ParakeetTDTJointNetwork` (duration head) | `modeling_parakeet.py` | 1035 |
| `ParakeetForTDT` | `modeling_parakeet.py` | 1052 |
| `ParakeetRNNTDecoderCache` | `generation_parakeet.py` | 23 |
| `ParakeetRNNTGenerationMixin` (greedy loop) | `generation_parakeet.py` | 125 |
| `ParakeetTDTGenerationMixin` (duration-aware loop) | `generation_parakeet.py` | 271 |
| `ParakeetRNNTConfig` / `ParakeetTDTConfig` | `configuration_parakeet.py` | 136 / 188 |

So the transducer is MIRROR work with citable `file:line`, exactly like the
encoder, and NOT a product call. It landed on CPU as row P6. **Method note worth
keeping:** a grep against the locally installed package is not evidence about
upstream. Pin and state the version, and check `main` before recording anything
as unmirrored.

**Out, and why.**

- **Streaming / cache-aware encoding and `<EOU>`/`<EOB>`.** Genuinely NeMo-only:
  it needs a causal/limited-context encoder (causal depthwise conv, causal
  downsampling, LayerNorm instead of BatchNorm, `att_context_style:
  chunked_limited`) plus per-layer cache tensors, and `ParakeetEncoderConfig`
  (`configuration_parakeet.py:23`) has NO field for attention context, causal
  convolution or streaming at all. See "Checkpoint reach" below.
- **The mel front end** beyond what `ParakeetExtractor` already specifies.
- **x86 SIMD quant (G5).** Unrelated row, and the 2026-08-06 op-dispatch profile
  ranks it below these.

## Checkpoint reach

Which published `nvidia/parakeet-*` checkpoints the port can actually load,
recorded 2026-08-07. The axis that decides this is not the architecture, it is
the CONTAINER: our loader reads an HF `config.json` + safetensors, so a model
published only as `.nemo` is out until something converts it.

| Checkpoint | Container | Head | State |
|---|---|---|---|
| `parakeet-ctc-0.6b` | HF safetensors | CTC | **IN**, transcript verified |
| `parakeet-ctc-1.1b` | HF safetensors | CTC | **IN**, transcript verified (same architecture, 42 layers instead of 24) |
| `parakeet-rnnt-0.6b` | HF safetensors | RNN-T | **IN**, transcript verified, ids exact vs HF `generate()` |
| `parakeet-rnnt-1.1b` | HF safetensors | RNN-T | **IN by construction** (same architecture, not run) |
| `parakeet-tdt-0.6b-v3` | HF safetensors | TDT | **IN**, transcript verified, ids exact vs HF `generate()` |
| `parakeet-tdt-0.6b-v2` | `.nemo` only | TDT | **OUT until converted**: architecture is in scope, container is not |
| `parakeet-tdt-1.1b` | `.nemo` only | TDT | **OUT until converted**: same |
| `parakeet-tdt_ctc-110m` | `.nemo` only | hybrid TDT + aux CTC | **OUT**: container, AND no hybrid class upstream |
| `parakeet-tdt_ctc-1.1b` | `.nemo` only | hybrid TDT + aux CTC | **OUT**: same |
| `parakeet_realtime_eou_120m-v1` | `.nemo` only | streaming RNN-T | **OUT**: a different ENCODER, see below |

**The `.nemo` container, and why we are not writing a reader.** A `.nemo` file is
an uncompressed tar holding `model_config.yaml` (OmegaConf YAML) and
`model_weights.ckpt` (a plain `torch.save` zip: pickle + raw storage blobs). A
C++ reader for the narrow tensor-rebuild protocol is genuinely small: a zip
reader plus the `persistent_load` / `_rebuild_tensor_v2` opcode subset, on the
order of several hundred lines. It is still the wrong call, because the reader
is not what is missing: we would also owe an OmegaConf-flavoured YAML parser,
and, for the streaming model, a reimplementation of NeMo's
`encoder.setup_streaming_params()` derivation (chunk size, shift size, cache
sizes are COMPUTED from `att_context_size` and the subsampling, not stored).

**The cheaper route, and its real limits.** transformers `main` ships
`src/transformers/models/parakeet/convert_nemo_to_hf.py` (577 lines), which
untars a `.nemo`, remaps the weights and emits `ParakeetForCTC` /
`ParakeetForRNNT` / `ParakeetForTDT` selected by a required
`--model_type {ctc,tdt,rnnt}`. Running it offline turns "supported" into
"supported after upstream's own converter", which is the honest and cheap answer
for `parakeet-tdt-0.6b-v2` and `parakeet-tdt-1.1b`: plain TDT, so the converter
covers them. It does NOT rescue the other three:

- **The `tdt_ctc` hybrids lose their CTC head.** `--model_type` is exclusive
  across the three, there is no hybrid HF class, and the converter's CTC mapping
  regex is `decoder\.decoder_layers\.0\.(weight|bias)`, which does not match the
  hybrid's `ctc_decoder.decoder_layers.0.*`. The TDT half converts; the aux CTC
  head is silently dropped.
- **`parakeet_realtime_eou_120m-v1` would convert to a SILENTLY WRONG model.**
  `convert_encoder_config` puts `att_context_size`, `att_context_style`,
  `causal_downsampling`, `conv_context_size` and `conv_norm_type` on its IGNORE
  list, and `ParakeetEncoderConfig` has no field for any of them. The weights
  load and the model runs as a full-context, non-causal, BatchNorm FastConformer.
  No error, just wrong output. Supporting it needs encoder work (causal depthwise
  convolution, causal downsampling, LayerNorm in the convolution module,
  chunked-limited attention, per-layer cache tensors) that has no upstream in
  transformers at all, plus `<EOU>` / `<EOB>` decoder-state resets.

**Recommendation.** Take the two plain TDT `.nemo` models via upstream's
converter if they are wanted; leave the hybrids and the streaming EOU model OUT
and say so, rather than growing a `.nemo` reader for them.

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
is mirror work, not a scope deviation. **Since 2026-08-07 the same is true of the
transducer heads**: see the correction in § Scope. The table above is the 5.3.0
tree; the transducer classes are in `main` at the lines listed there.

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
| `main:831` `ParakeetRNNTDecoder` | new `src/vllm/model_executor/models/parakeet_transducer.cpp`: embedding + stacked LSTM cell + projector, with the cache's blank fast path | P6 |
| `main:879` `ParakeetRNNTJointNetwork`, `:1035` `ParakeetTDTJointNetwork` | same file: `head(act(enc + dec))`, widened by `len(durations)` for TDT | P6 |
| `main:922` `ParakeetForRNNT`, `:1052` `ParakeetForTDT` | same file + `encoder_projector` | P6 |
| `generation_parakeet.py:125` / `:271` generation mixins | the greedy transducer loop in the same file | P6; greedy only, which is upstream's whole supported surface (`_supported_generation_modes`, `main:925`) |
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
| e2e transducer | `tests/vllm/models/test_parakeet_transducer.cpp` | Emitted sequence and per-step durations vs a HF `ParakeetForRNNT` / `ParakeetForTDT` oracle, plus a PRETRAINED arm gated on a real `generate()` run |

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
| P6 | `MODEL-AUDIO-PARAKEET-TRANSDUCER` | `parakeet_transducer.{h,cpp}`, the transducer half of `parakeet_weights.cpp`, `test_parakeet_transducer.cpp`, the two `scripts/mm/p5_parakeet_transducer_*.py` generators | after P4 |

P5 is listed because it is the cheap proof that the three kernels are general
rather than Parakeet-shaped, and because it is a native-vLLM mirror rather than
an HF one.

## Risks and decisions

**Product calls, for the human:**

1. ~~**Does the transducer (RNN-T / TDT) belong in vllm.cpp at all?** It has no
   upstream anywhere in the chain.~~ **WITHDRAWN 2026-08-07: the premise was
   false.** transformers `main` implements the whole transducer stack (§ Scope),
   so this was never a product call, it was mirror work behind a stale version
   check. Landed as P6.
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
