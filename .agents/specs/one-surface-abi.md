# ONE SURFACE — every capability ships through the C ABI

Row: `ARCH-ONE-SURFACE`. Status: **AUDIT DONE; remediation IN PROGRESS — ROW 1 (Parakeet ASR / audio transcription) LANDED 2026-08-07: ABI v11 `vllm_transcribe`, live `/v1/audio/transcriptions`, registry refuse-by-task, example folded, ratchet 12 -> 11.**

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
| MiniMax-H3 video+audio gen | NO | no arch, no video C-ABI; served via example-injected `VideoRunner` | `examples/minimax_h3_gen`, `examples/server` `/v1/videos` | 1293 lines |
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

## Proposed ABI shape for video (first slice)

Mirrors the existing engine idiom: an opaque handle, a params struct with a
`_default()`, an explicit free, `vllm_last_error` for diagnosis.

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

The VIDEO ABI shape above is a proposal, not an implementation, and it has not
been reviewed against an embedder other than LocalAI. The ONE landed slice is
audio transcription (ROW 1, 2026-08-07): `vllm_transcribe` + params/result
structs on ABI v11, the `ParakeetTranscriber` library seam, task-conditional
`/v1/audio/transcriptions`, registry refuse-by-task, and the example as a thin
`vllm.h` client — gated byte-identical to the pre-fold transcripts. Video,
Laguna/DeepSeek/Kimi fast decode, embeddings and multimodal input remain open
rows of this program.
