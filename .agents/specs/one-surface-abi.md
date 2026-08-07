# ONE SURFACE — every capability ships through the C ABI

Row: `ARCH-ONE-SURFACE`. Status: **AUDIT DONE; remediation IN PROGRESS — ROW 1 (Parakeet ASR / audio transcription) LANDED 2026-08-07: ABI v11 `vllm_transcribe`, live `/v1/audio/transcriptions`, registry refuse-by-task, example folded, ratchet 12 -> 11. ROW 2 (MiniMax-H3 video+audio generation) LANDED 2026-08-08 (`row/H3-VIDEO-ABI`, task #283): ABI v12 `vllm_video_engine_load`/`vllm_video_generate`/`vllm_video_result_free` + `vllm_video_mux_argv` over the `MiniMaxH3VideoEngine` library seam, `/v1/videos` routed through the SAME seam, both H3 examples rewritten as `vllm.h` clients byte-identical to the pre-fold binary, ratchet 11 -> 9. GB10 real-video re-verification is a NAMED RESIDUAL (box on the Kimi campaign).**

## The defect

`include/vllm.h` (`VLLM_ABI_VERSION 10`, 19 symbols) exposes **text completion and
chat only**: `vllm_engine_load/free`, `vllm_complete{,_stream}`, `vllm_chat{,_stream}`,
`vllm_request_*`, `vllm_sampling_params_default`, `vllm_model_params_default`,
`vllm_abi_version`, `vllm_version`, `vllm_last_error`, `vllm_string_free`,
`vllm_completion_free`.

Everything else this project can do lives in `examples/`, where no embedder can
reach it. Found when LocalAI's `vllm-cpp` backend, which `dlopen`s this ABI, needed
video generation: the capability was complete, gated and shipping, and from the
outside it did not exist.

## Audit — this is NOT only video

**CORRECTED 2026-08-07 (`row/SURFACE-COVERAGE-AUDIT`).** The original claim here —
"None of these four architectures appear in `model_registry.cpp`" — is FALSE for 3 of 4:
`LagunaForCausalLM` (`laguna_registry.cpp:131`), `KimiLinearForCausalLM`
(`kimi_linear_registry.cpp:142`) and `DeepseekV4ForCausalLM` (`deepseek_v4_registry.cpp:150`)
ARE registered; only MiniMax-H3 is off-registry. Registration did not make the capability
reachable, though — each registered arch still fails ONE SURFACE differently, so the rows
stay OPEN. The complete, code-grounded matrix (all 30 archs + every off-registry lane) is
`.agents/specs/surface-coverage-2026-08-07.md`.

| capability | registered? | why still off-surface | only real path via | example size |
|---|---|---|---|---|
| MiniMax-H3 video+audio gen | NO (diffusion lane) | **CLOSED (ROW 2)** — `vllm_video_*` on ABI v12; `/v1/videos` drives the library seam | library seam `MiniMaxH3VideoEngine`; examples are thin clients | 216 lines (was 1293) |
| Laguna | YES | keep-quant/NVFP4 decode example-only; registry forward `VT_CHECK`s non-bf16; GGUF dispatch unreachable | `examples/laguna_gen` | 415 lines |
| Kimi-Linear | YES | registry leg is a stateless recompute reference; §18/§19 incremental entry points are private | `examples/kimi_linear_gen` | 318 lines |
| DeepSeek-V4 | YES | KV spec is a "never exercised" stub; registry forward discards attn_meta/kv | `examples/deepseek_v4_gen` | 240 lines |

This table also UNDER-COUNTS: it omits Parakeet ASR (a 5th off-surface capability, landed
`fd2259d8` — **CLOSED 2026-08-07 by ROW 1**: `vllm_transcribe` on ABI v11, live
`/v1/audio/transcriptions`, registry refuse-by-task, example rewritten as a `vllm.h`
client, ratchet 12 -> 11) and the partial internal-reachers `minimax_h3_mux`, `bench`,
`tokenize`, `dump_container`, `dequant_nvfp4`, `quant_gemm_bench` — 11 of 13 example
binaries still include non-public headers (`examples/cli` and `examples/parakeet_transcribe`
are the clean ABI clients). The full list is the audit spec.

`examples/server/main.cpp` also carries ~32 internal includes (54 direct `MiniMaxH3`
references), i.e. it re-implements wiring rather than consuming a library entry point. So
HTTP and FFI can drift, and have.

Three of these own real generation logic, not just argv parsing:
`kimi_linear_gen`, `minimax_h3_gen`, `server` all reference `DenoiseLoop` /
`GenerateT2va` / `ForwardDevice` directly.

## Why the gates never caught it

Every one of these paths is well tested. `test_minimax_h3` alone runs 75 cases and
55,000+ assertions. **No gate asks whether a CONSUMER can reach a capability**, so a
second-class path stays green forever. That is the hole this row closes.

## Target state

llama.cpp is the model: a library of reusable pieces callable from anywhere, with
`llama-cli` / `llama-server` as consumers rather than privileged holders of
behaviour.

1. Each capability gets a C ABI entry point, appended so zero values preserve
   existing behaviour, `VLLM_ABI_VERSION` bumped, `vllm_abi_version()` truthful.
2. The library does the work. `src/vllm/` still spawns NOTHING — the ratified
   ffmpeg boundary stands: the library produces artifacts and argv, the caller
   invokes.
3. `examples/` is REFACTORED onto the entry point. Two implementations of one
   capability is the defect, not the fix.
4. `examples/server` routes through the SAME entry point, so HTTP and FFI cannot
   drift.

## Proposed ABI shape for video (first slice) — SHIPPED (ROW 2, ABI v12)

Mirrors the existing engine idiom: an opaque handle, a params struct with a
`_default()`, an explicit free, `vllm_last_error` for diagnosis.

**As-shipped deltas from the proposal below (each argued during ROW 2,
2026-08-08; `include/vllm.h` is the binding text):**
- `vllm_video_model_params` GAINS `prompt_embeds_path` (without an encoder
  there is NO conditioning path — both pre-fold consumers had this arm),
  `partition` (the #77 guard refuses every full render without a declared
  partition, so omitting it would make the ABI unable to render at all) and
  `fp4_resident` (the gated NVFP4 Marlin arm); `encoder_max_layers` stays a
  C++-seam knob.
- `vllm_video_params` GAINS `output_dir` (the result needs a destination) and
  DROPS `duration_seconds`/`task` (derivable: `num_frames` expresses duration,
  and the task is resolved from the references exactly as upstream
  `_resolve_task` does; both remain on the C++ seam for the server).
- `vllm_video_result.mux_argv` is `char**`, NULL-terminated (execvp-ready),
  freed via `vllm_video_result_free`.
- ADDED `vllm_video_mux_params(_default)` + `vllm_video_mux_argv(_free)`: the
  engine-free composer `minimax-h3-mux` needs to be a `vllm.h` client (W4);
  the encoding contract stays the library's, the caller execs.
- NAMED RESIDUALS of the first slice: ONE `ref_image` (multi-image ref2va is
  C++-seam-reachable only), and the pre-fold example's diagnostic modes
  (`--denoise-only`, `--dump-params`, `--encoder-only`/`--save-embeds`,
  `--decode-latent`, `--roundtrip`, `--prompt-image`, `--cond-image`,
  `--dry-run`) were deleted with the private pipeline — the capabilities they
  probed are gated by `test_minimax_h3`/`test_minimax_h3_video_fold`, and the
  GB10 speed recipe moves to the seam/ABI (re-verification residual below).
- The pre-fold CPU host-f32 GGUF arm (the example default with NEITHER
  `--keep-quant` nor `--dequant-bf16`) is not on the ABI: `dequant_bf16=0` is
  keep-quant (the gated arm the fold goldens were captured on).

```c
typedef struct vllm_video_engine vllm_video_engine;

typedef struct {                  /* all paths; empty means "not supplied" */
  const char* dit_path;           /* GGUF or sharded dir */
  const char* encoder_path;       /* GGUF or sharded dir */
  const char* tokenizer_path;
  const char* video_vae_path;     const char* video_vae_config_path;
  const char* audio_vae_path;     const char* audio_vae_config_path;
  int32_t device;                 /* 0 cpu, 1 cuda */
  int32_t dequant_bf16;           /* 0 keep-quant, 1 stream bf16 */
} vllm_video_model_params;

typedef struct {
  const char* prompt;
  int32_t width, height, num_frames, steps;
  uint64_t seed; int32_t has_seed;
  const char* first_frame;  const char* last_frame;   /* fl2va keyframes */
  const char* ref_image;    const char* ref_video;    /* ref2va */
  const char* ref_audio;
  float noise_aug;
} vllm_video_params;

typedef struct {                  /* the library WRITES these, spawns nothing */
  const char* frame_dir;          /* frame_%06d.ppm */
  const char* audio_path;         /* 16-bit PCM WAV */
  int32_t frame_count, width, height, fps, sample_rate;
  const char* const* mux_argv;    /* argv the CALLER may exec (ffmpeg) */
  int32_t mux_argc;
} vllm_video_result;
```

`mux_argv` is what keeps the ffmpeg boundary intact while still letting an
embedder produce an MP4 without reinventing the command: the library composes it
(`MiniMaxH3BuildMp4MuxArgs` already does exactly this), the caller executes it.

## Order of work

1. **This row**: the T0 directive in `AGENTS.md` + `.agents/directives.md`, and this
   audit. Documentation only.
2. **Video ABI slice**: the entry points above, `minimax_h3_gen` refactored onto
   them, `examples/server` `/v1/videos` routed through them, ABI bumped to 11.
3. **A GATE** so this cannot regress: a check that every capability named in
   `docs/FEATURES.md` as supported has an ABI path, or is explicitly listed as
   embedder-unreachable with a reason. Without this, the directive is a note.
   **Operator-directed 2026-08-07: the gate runs in `scripts/agent-preflight.sh`
   — before every commit/push — as well as in CI.** Same for the
   include-boundary companion check (examples/* may include only the public
   exported headers). Being CI-only would let a whole session build on an
   unreachable capability before the first PR run says no.
   **LANDED 2026-08-07 (`row/SURFACE-COVERAGE-AUDIT`):** `scripts/check-surface-coverage.py`
   (two axes — the FEATURES capability-reachability check bound to `include/vllm.h`, and the
   examples/* include-boundary check with the public set DERIVED from the CMake install
   rules), wired into BOTH `scripts/agent-preflight.sh` and CI, with 46 mutation tests. No
   permanent dev-tool exemption: every internal-reaching example is a transition-tracker row
   in `scripts/example-abi-allowlist.txt` pointing at its fold row, shrink-only ratchet. NB:
   `check-supported-models.py:22-24` is NOT a substitute — it deliberately tolerates
   non-registered lanes (the exact class Parakeet occupies), so it cannot enforce
   reachability.
4. The remaining three (Laguna, Kimi-Linear, DeepSeek-V4), which most likely want
   registry entries rather than bespoke entry points.

## NOT claimed

The video ABI has not been reviewed against an embedder other than LocalAI.
TWO slices are landed: audio transcription (ROW 1, 2026-08-07 — ABI v11
`vllm_transcribe`, the `ParakeetTranscriber` seam, task-conditional
`/v1/audio/transcriptions`, example as a thin client, byte-identical
transcripts) and video+audio generation (ROW 2, 2026-08-08 — ABI v12
`vllm_video_*`, the `MiniMaxH3VideoEngine` seam, `/v1/videos` through it, both
H3 examples as thin clients, frames+WAV byte-identical to the pre-fold binary
on the committed fold fixture; three-arm gate `test_minimax_h3_video_fold` +
the v12 `test_capi` section). ROW 2 residuals, named: (1) GB10 real-video
re-verification (real checkpoints through the v12 ABI + the folded server;
the box is running the Kimi campaign — CPU fold gates are the landed
evidence); (2) DISCLOSED server-path numeric deltas (no goldens existed): the
pre-fold `/v1/videos` runner drew single-stream legacy-uniform noise and
defaulted to the host-f32 GGUF arm — it now shares the ratified recipe
(dual-stream splitmix64 Gaussian, keep-quant default), which kills the exact
HTTP-vs-CLI drift this row exists to prevent; (3) multi-image ref2va on the
ABI. Laguna/DeepSeek/Kimi fast decode, embeddings and multimodal input remain
open rows of this program.
