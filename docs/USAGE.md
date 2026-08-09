# Using vllm.cpp

The complete surface: the CLI, the OpenAI-compatible server, and the library
(C ABI and C++). The [README](../README.md) carries the quickstart; this page is
the reference behind it. Per-capability lifecycle state is
[docs/STATUS.md](STATUS.md); measured numbers are
[docs/BENCHMARKS.md](BENCHMARKS.md).

## Confirming which CUDA architecture a build targets

`CMakeCache.txt` is now a reliable answer. Configuring with
`-DVLLM_CPP_CUDA_ARCHITECTURES=<arch>` writes that value into
`CMAKE_CUDA_ARCHITECTURES` in the cache, so the two agree:

```sh
grep '^CMAKE_CUDA_ARCHITECTURES' build-cuda/CMakeCache.txt
```

It previously reported the toolkit's detected default (typically `75`) no matter
what was requested, because the project set the variable without writing it back
to the cache. Only the report was wrong — the emitted gencode always followed the
requested value — but it sent a contributor looking in the wrong place
([#168](https://github.com/mudler/vllm.cpp/issues/168)). The `build.ninja`
gencode line remains the ground truth if you want to double-check.

## Starting an agent-assisted contribution

Run `scripts/agent-start.py` first. It reports an inherited worktree role or,
for a new contributor with no declared role or explicit intent, prints the
welcome that the agent should relay. An explicit request can use
`--intent operator|helper|read-only` and a helper `--row ID`. Follow its printed
claim action, rerun it after declaration, then run `scripts/agent-preflight.sh`.
The entrypoint is non-interactive and does not mutate the checkout.

## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source:
[`examples/cli/main.cpp`](../examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--repeat N` | `1` | Load once, then run N blocking completions. Use it to read a warm decode tok/s without paying model load each time. Not supported with `--stream`, which falls back to 1 |
| `-h`, `--help` | | Print usage and exit |

Two more example binaries ship alongside it:

- `vllm-bench` ([`examples/bench/main.cpp`](../examples/bench/main.cpp)), a
  throughput/latency harness taking `--model`, `--dataset-path`,
  `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`,
  `--max-num-batched-tokens`, and `--num-blocks`.
- `tokenize` ([`examples/tokenize/main.cpp`](../examples/tokenize/main.cpp)), a
  tokenizer smoke tool taking `<tokenizer.json | model.gguf> <corpus.txt>`.

### How much memory a Vulkan load needs

On a unified-memory device (a DGX Spark) the Vulkan heap and system RAM are the
same bytes, so budget roughly **the checkpoint size plus about 5%**, plus your KV
pool. Measured on GB10: Qwen3.6-27B bf16 (50.89 GiB on disk) peaks at 53.4 GiB of
process RSS. Reading the checkpoint also fills the page cache with about the file
size; that is reclaimable and does not need to be budgeted, but it does make
`MemFree` look alarming during a load. Use `MemAvailable`, not `MemFree`, to
decide whether a model fits. `VT_VULKAN_ALLOC_STATS=1` prints the running device
total and the `/proc` context if you need to see where it goes.

A Vulkan build (`-DVLLM_CPP_VULKAN=ON`) adds three kernel-measurement binaries.
They exist so a Vulkan tuning knob can be A/B'd in ONE binary, which is this
project's benchmark protocol, and each one prints WHICH kernel variant it ran so
a silent fallback cannot post a plausible number:

- `vulkan-gemm-ab`, cooperative-matrix versus the portable scalar GEMM
  (`VT_VULKAN_COOPMAT=0` picks the arm). Takes `M K N [reps]`.
- `vulkan-dispatch-floor`, one op swept across a 65,536x range of element counts,
  to separate per-dispatch overhead from real kernel cost.
- `vulkan-gemv-ab`, the decode GEMV swept over the (k, n) shapes a 27B model
  actually dispatches, with `VT_VULKAN_GEMV_ROWS` / `VT_VULKAN_GEMV_PACK` /
  `VT_VULKAN_GEMV_UNROLL` selecting the arm. Takes `[reps] [warmup] [GB/s roof]`
  and reports GB/s against that roof. Set `VT_VULKAN_DISPATCH_STATS=1` so it
  reports GPU-timestamp time rather than wall clock; see
  [ENVIRONMENT.md](ENVIRONMENT.md) for what each knob does and what it measured.

### Quantized checkpoints: which weight forms load

Publishers do not agree on how weights are stored, and a single repo can change
it between revisions (one 27B "NVFP4" repo silently became FP8 throughout).
The table below is about `lm_head`; the same three forms are accepted for the
attention, MLP and `linear_attn` projections, in both compressed-tensors
(`weight_packed` + `weight_global_scale`) and ModelOpt (`weight` +
`weight_scale_2`) naming. For the Qwen3.6 dense family we accept all three
forms in use, so pick a checkpoint by its quality, not by its head:

| `lm_head.weight` | Companion tensors | Seen in |
|---|---|---|
| `BF16` | none | `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` |
| `F8_E4M3` | `lm_head.weight_scale` (per-output-channel or per-tensor) | `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` |
| `U8` NVFP4 | `lm_head.weight_scale` + `weight_scale_2` (ModelOpt) or `weight_global_scale` (compressed-tensors) | `nvidia/Qwen3.6-27B-NVFP4` |

The head is dequantized to BF16 at load, so all three cost the same memory once
running. Any other dtype fails at load with a message naming what it saw.

## OpenAI-compatible server

`vllm-server` is a small HTTP server speaking the OpenAI API. Source:
[`examples/server/main.cpp`](../examples/server/main.cpp) and
[`src/vllm/entrypoints/openai/`](../src/vllm/entrypoints/openai/).

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```

The install component and deterministic archive target both stage from install
rules rather than copying the build tree:

```sh
cmake --build build --target vllm-server-stage
cmake --build build --target vllm-server-archive
build/release/stage/bin/vllm-server --help
```

The basic CMake archive under `build/release/` includes the version, configured
backend, OS, and host architecture in its name. It is a developer package. The
release workflow separately produces host-ABI-specific archives with a
manifest, `VERSION`, SPDX SBOM, notices, licenses, and detached checksum and
provenance sidecars; no release download is claimed until that workflow has
completed on a release tag.

To reproduce the W1 heterogeneous CUDA archive candidate, configure the exact
release architecture set. Portable translation units compile for all ten SMs;
architecture-specific kernels compile only for their supported intersection:

```sh
cmake -S . -B build-cuda-fat -G Ninja \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES='80;86;87;89;90a;100a;103a;110;120a;121a' \
  -DVLLM_CPP_CUTLASS_FETCH=ON -DVLLM_CPP_TRITON=OFF
cmake --build build-cuda-fat --target vllm
python3 scripts/check-cuda-fat-gencode.py \
  --compile-commands build-cuda-fat/compile_commands.json \
  --library build-cuda-fat/libvllm.a
```

The release workflow applies this audit to independently linked x86_64 and
arm64 host executables, packages each as a preview `cuda-fat` archive, and then
runs the extracted-archive validator. Each archive must contain all ten SM
images and the six available exact-SM Triton AOT namespaces; the manifest keeps
runtime evidence separate per SM. These build-only preview candidates are not
a downloadable release claim until the tagged workflow publishes them.

The complete primary download matrix and its runtime boundaries are documented
in [RELEASES.md](RELEASES.md). A manual workflow dispatch runs all eight tuples
without publication. An exact version tag runs the same build, produces
`release-index.json` and `RELEASE_INDEX.md` from the verified archive manifests,
attests the archive bytes, and publishes every archive/checksum/provenance
triplet through the protected release environment.

### Selecting an x86 CPU ISA tier

The x86_64 CPU library is one adaptive binary: portable, SSE2,
SSE2+F16C, AVX2, and AVX-512 elementwise matmul kernels are isolated in their
own translation units and selected only after CPUID plus the required XCR0 OS
state are checked. Leave `VT_CPU_MATMUL_TIER` unset for automatic selection, or
set it to `portable`, `sse2`, `sse2+f16c`, `avx2`, or `avx512` for a same-binary
correctness/performance check. A forced tier that the current CPU or OS cannot
execute fails closed instead of silently narrowing or risking an illegal
instruction. Release builds never use `-march=native`.

On arm64, leave the same variable unset to select between portable and NEON
elementwise matmul, or force `portable`/`neon`. DotProd and i8mm kernels are
independently selectable with `VT_CPU_Q8_DOT`, `VT_CPU_QUANT_MMLA`, and
`VT_CPU_QUANT_REPACK`; `auto` uses Linux HWCAP/HWCAP2 or Darwin feature sysctls,
while an unavailable forced tier fails closed. The exact accepted values are
listed in [ENVIRONMENT.md](ENVIRONMENT.md).

### Validating a staged release archive

Release verification reads only a freshly extracted archive, never files from
the build tree. Pass the archive together with its final-byte SHA256 and SLSA
provenance sidecars:

```sh
python3 scripts/validate-release-archive.py \
  --archive vllm.cpp-0.0.1-cpu-linux-x86_64.tar.gz \
  --checksum vllm.cpp-0.0.1-cpu-linux-x86_64.tar.gz.sha256 \
  --provenance vllm.cpp-0.0.1-cpu-linux-x86_64.tar.gz.provenance.json \
  --forbid-path "$PWD/build"
```

The validator checks the content allowlist, executable and host ABI, manifest,
`VERSION`, SPDX SBOM, licenses, ELF dependencies and RPATH/RUNPATH, extracted
`--help`/`--version` smokes, and backend-specific CUDA or adaptive-CPU claims.
The digest and provenance are sidecars because both describe the final archive
bytes; placing either inside those bytes would create a self-reference.

The CPU release helper is the reproducible entry point used by CI. It requires
an explicit artifact tuple, architecture, channel, build directory, libc ABI,
a feature-poor QEMU userspace emulator, and a feature-rich runner. x86_64 uses
the SHA256-pinned Intel SDE installed by `scripts/install-intel-sde.sh` so the
AVX-512 tier is really executed even when the host lacks AVX-512. The gate then
executes the baseline and proves rich-tier refusal under the feature-poor QEMU
model before metadata can be generated:

```sh
SOURCE_SHA=$(git rev-parse HEAD) \
VERSION=0.0.1 \
SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD) \
EVIDENCE_URL=https://github.com/mudler/vllm.cpp/actions/runs/EXAMPLE \
scripts/build-cpu-release.sh \
  linux-x86_64-glibc-cpu x86_64 stable build-release-cpu-x86 \
  2.39 /usr/bin/qemu-x86_64 /tmp/intel-sde/sde64
```

The corresponding arm64 tuple is `linux-aarch64-glibc-cpu`. The only literal
static tuple is the CPU-only `linux-x86_64-musl-cpu-static` experiment; normal
CPU and accelerator archives are static-core bundles with audited host runtime
dependencies.

To exercise the release pipeline without publishing anything, trigger its
manual entry point:

```sh
gh workflow run release.yml --ref main
```

Manual runs are always dry runs. Publication additionally requires an exact
`v<project-version>` tag, a release matrix whose required lanes are all marked
ready, successful verification and attestation jobs, and approval of the
protected `release` environment. Build and verification jobs have read-only
repository permissions; only attestation receives OIDC authority, and only the
final protected job receives `contents: write`.

Any OpenAI client works by pointing its `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

### Endpoints

Registered in
[`src/vllm/entrypoints/openai/api_server.cpp`](../src/vllm/entrypoints/openai/api_server.cpp).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET, POST | `/ping` | Liveness probe (200, mirrors `/health`) |
| GET | `/version` | Engine version |
| GET | `/metrics` | Prometheus metrics (`vllm:*` names, text format 0.0.4) |
| POST | `/tokenize` | Tokenize a `prompt` to token ids (optional `token_strs`) |
| POST | `/detokenize` | Detokenize token ids back to text |
| GET | `/server_info` | Server info (`vllm_config`, `vllm_env`, `system_env`) |
| POST | `/reset_prefix_cache` | Reset the prefix cache; returns `{"success": bool}` |
| POST | `/v1/videos` | Start a video generation job, returns `{id, status}` (MiniMax-H3) |
| POST | `/v1/videos/sync` | Same, but runs to completion before answering |
| GET | `/v1/videos/{id}` | Job status |
| GET | `/v1/videos/{id}/content` | The finished MP4 (`video/mp4`) |

The four `/v1/videos` routes are registered **only** when the server was started
with `--video-dit`; without it they are absent (404) and the server is identical
to one built without video support. See
[MiniMax-H3: video + audio generation](#minimax-h3-video--audio-generation).

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size |
| `--num-blocks N` | `256` | KV blocks |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `32` | Max concurrent sequences (also sizes the HTTP worker pool). Was `8`, which put a c8 client exactly on the batch ceiling; vLLM's own default is 1024, which we do not mirror because this also caps the padded decode-graph set |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [docs/SGLANG-COMPAT.md](SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` | off | Jump-forward decoding for structured output (token-unique subset) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (40 names over 36 families). `auto` detects from the chat template, `none` disables |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`). `auto` detects, `none` disables |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [docs/KV-OFFLOAD.md](KV-OFFLOAD.md) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding (`mtp`, `dflash`, `ngram`), same JSON as vLLM's flag. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--enable-log-requests` / `--disable-log-requests` | on | Log each incoming request. Mirrors vLLM's flag of the same name |
| `--enable-log-outputs` | off | Also log the generated output, not just the request |
| `--max-log-len N` | `256` | Truncate logged prompts and outputs to N characters |
| `--enable-metrics` / `--disable-metrics` | on | Serve the metrics endpoint |
| `--enable-thinking` / `--no-enable-thinking` | off | Set the `enable_thinking` chat-template variable for templates that gate a reasoning block on it (Gemma-4 and friends). Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking` |
| `--verbose`, `-v` | off | Verbose server logging |
| `--cuda-profile-graph-replays N` | `0` (off) | Trace-only diagnostic: arm the CUDA-graph-replay profiler and stop after N replays, printing a pid to signal with `SIGUSR2`. Requires a build with `VT_BENCH_PROFILE_CONTROL` |
| `--cuda-profile-graph-batch N` | `16` when replays are armed | Batch size the profiler traces. Must not exceed `--max-num-seqs` |
| `-h`, `--help` | | Print usage and exit |

For a production deployment, use [LocalAI](https://localai.io), which can embed
engines like this behind a model gallery, multi-model serving, the full OpenAI
API surface, auth, and metrics.

## MiniMax-H3: video + audio generation

### The exact weights (so a render is reproducible)

Five files. The DiT and encoder are community GGUF quantisations; the two VAEs and
the tokenizer come from the official checkpoint.

| file | size | source |
|---|---|---|
| `MiniMax-H3-FL2VA-Q4_K_M.gguf` | 19.9 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14.6 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `vae/diffusion_pytorch_model.safetensors` | 5.2 GB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/video_vae/` |
| `audio_vae/model.safetensors` | 0.6 GB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/audio_vae/` |
| `tokenizer.json` | 7 MB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/tokenizer/` |

Take each VAE's `config.json` from the same directory as its weights: they carry the
per-channel `latents_mean` / `latents_std` and the temporal `clip_length` /
`token_drop`, and the decode is wrong without them.

**Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude
outliers that 3-bit cannot hold. In a controlled A/B (same prompt, seed, code and
VAEs, only the DiT quantisation changed) Q3_K_M gave a murky silhouette under a
visible lattice and Q4_K_M gave a photoreal close-up. The full bf16 release is
66.3 GB across 13 shards if you want to go further.

Higher-precision arms that exist but are not the default: NVFP4
([lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4),
note the pruned variants restructure AdaLN into a timestep lookup table and are
NOT drop-in), and the original bf16 weights under `FL2VA/transformer/`.

**What is actually verified, and what merely exists.** The distinction matters
because a render takes hours before it tells you anything:

| arm | status |
|---|---|
| **Q4_K_M** | **VERIFIED end to end** — every render in this doc. Use this. |
| Q3_K_M | verified BAD (the A/B above): murky silhouette under a lattice |
| bf16 (66.3 GB, 13 shards) | loader + device streamer implemented and gated, but **CPU-only** verification — no end-to-end GPU render has been done |
| NVFP4 | exists; pruned variants are NOT drop-in (see above) |
| Q8 | no published GGUF is known and none has been tested |

### The trap: this checkpoint does not serve every task

**`MiniMax-H3-FL2VA-Q4_K_M.gguf` is the FL2VA partition. It serves `t2va` and
`fl2va` — NOT `ref2va`.** H3 ships two independently-served DiT partitions and
the task must match the one you loaded; upstream's `_resolve_task` raises on the
mismatch.

Pass a reference image against this file and you get a task/partition mismatch.
It does not fail loudly — it renders, and the render is *wrong*: a coloured
diagonal lattice over the whole frame, worse the larger the canvas. Measured on
one prompt and canvas (1344x768 / 124f), as a period-16 seam ratio where 1.15 is
clean:

| configuration | seam ratio |
|---|---|
| ref2va against FL2VA (the mismatch) | 2.28 |
| t2va against FL2VA (correct) | **1.19** |

The small-canvas case is what makes this expensive to spot: at 864x480 the same
mismatch measures 1.15 and looks acceptable, so the bug only becomes obvious at
the resolution you actually want.

Pass `--partition fl2va` explicitly. The driver mirrors upstream's raise, so a
mismatch is rejected at the CLI rather than silently rendered.

For a reference-image render you need the **Ref2VA** partition instead
(`minimax_h3_ref2va_nvfp4_full`, or a REF2VA GGUF). Note that the encoder vision
tower is not yet ported, so image-conditioned renders are not clean even on the
matching partition.

### Writing the prompt (read this first)

Two things decide whether you get what you asked for, and neither is obvious.

**To get SPEECH, ask for it and supply the line.** The model generates video and
audio jointly, so a prompt describing a silent performance produces room tone and
ambience, which is correct but not what most people expect. Say that the character
talks, describe the voice, and put the words in the prompt:

```
It is TALKING to the camera: its mouth moves clearly in sync with its speech,
in a dry, deadpan tone.

It says, clearly and audibly: "Michael scheduled another all-hands.
It is about the printer. Again."

Audio: a single clear voice, close-miked, with quiet room tone underneath.
```

That prompt produced audio an ASR pass transcribed back word for word. A prompt
that only described expressions and sighs produced ambience at about 13 dB lower
level and no speech at all.

**Refer to references BY TAG in the prompt text.** A reference is bound by naming
it, not merely by being passed on the command line. Use `<Picture i>`, `<Video k>`
and `<Audio j>`, numbered from 1 per type, matching the order you pass them:

```
<Picture 1> is a cyan llama mascot wearing white sunglasses.

A talking-head interview. The subject is the llama from <Picture 1>, sitting in a
grey office chair ...
```

Other prompt notes: frame count runs on the 17n+5 grid at 24 fps, and the trained
range is roughly 124 to 362 frames (about 5 to 15 seconds). Text rendered *inside*
the video (signage, wordmarks) is the model's weakest area and will often come out
malformed; composite real logos in afterwards.

`/v1/videos` generates video with sound through the MiniMax-H3 diffusion model.
It speaks **OpenAI's Sora video shape**, so an OpenAI client works against it
unmodified, and it keeps the richer native knobs alongside.

```sh
build/examples/server --model /path/to/Qwen3.6-27B \
  --video-dit /path/to/h3-dit.gguf --video-vae /path/to/video-vae.safetensors \
  --audio-vae /path/to/audio-vae.safetensors \
  --video-vae-config video_vae/config.json --audio-vae-config audio_vae/config.json \
  --video-encoder /path/to/h3-encoder.gguf
```

```python
video = client.videos.create(model="sora-2-pro", prompt="a cat on a skateboard",
                             size="1280x720", seconds="8")
while client.videos.retrieve(video.id).status not in ("succeeded", "failed"):
    time.sleep(5)
open("out.mp4", "wb").write(client.videos.download_content(video.id).read())
```

### Request fields

| Field | Spelling | Meaning |
|---|---|---|
| `prompt` | both | Required. The text conditioning |
| `model` | OpenAI | Recorded and echoed back. A name this server does not serve is a `warning` on the job, never a rejection: the video model is chosen at startup |
| `size` | OpenAI | `"<width>x<height>"`, e.g. `"1280x720"`. Whole pixels, both positive |
| `seconds` | OpenAI | Duration, as a number or a numeric string (`8` and `"8"` both work) |
| `input_reference` | OpenAI | The image the video starts from. A filesystem path or a `data:` URL |
| `metadata` | OpenAI | Free-form string map, passed through untouched. Two keys are acted on: `input_reference_video` and `input_reference_audio` (see below) |
| `width`, `height` | native | Output geometry in pixels |
| `duration` | native | Duration in seconds |
| `task` | native | `t2va`, `fl2va`, `ref2va`; resolved from the inputs when omitted |
| `num_frames`, `num_inference_steps`, `flow_shift`, `audio_flow_shift`, `seed` | native | The H3 generation knobs. Accepted at the top level or nested under `extra_params` |

**Precedence.** When a body carries both spellings of one value, the **native
field wins**: `width`/`height` beat `size`, `duration` beats `seconds`. That
direction keeps every request that parses today meaning exactly what it meant
before. Both spellings are validated either way, so a malformed `size` is a 400
even when explicit `width`/`height` would have overridden it.

**`input_reference` maps to fl2va first-frame conditioning.** OpenAI documents
it as the image the generated video starts from, which is what fl2va expresses:
the supplied image is pinned as frame 0 of the output. H3's other image mode,
ref2va, prepends whole reference images as their own blocks (subject or style
guidance that never becomes a frame), so it stays reachable only through the
native `task` field and the `minimax-h3-gen` CLI. Two limits: the image must be
a **binary PPM (P6)** (no PNG or JPEG codec is vendored, the same residual the
chat multimodal path carries), and it must already be at the output resolution
(no image resampler is vendored). A mismatch is refused with the resolved
geometry in the message.

### Video and audio references (`metadata`)

H3 supports three reference modalities and OpenAI's schema has a slot for one,
so the other two enter through `metadata`, the standard OpenAI free-form string
map. Strict clients tolerate it, and no invented top-level field breaks their
schema validation. Unknown metadata keys are passed through untouched.

```jsonc
{
  "prompt": "the same scene, at dusk",
  "metadata": {
    "input_reference_video": "/tmp/vllm_h3_videos/job0",  // DIR of frame_%06d.ppm
    "input_reference_audio": "/tmp/voice.wav"             // 16-bit PCM WAV, or a data: URL
  }
}
```

`input_reference_video` is a **directory of `frame_%06d.ppm`**, which is exactly
what this server and `minimax-h3-gen` write, so one run's frames chain straight
into the next request. It is not a container file: no demuxer is vendored.

**A video reference is SILENT.** `MiniMaxH3EncodeReferenceVideo` emits a
`kVideoAudio` block with `ref_audio_t == 0`, so the clip contributes no sound of
its own. Supplying `input_reference_audio` alongside it attaches the audio to
that same block (one block carrying both, the layout upstream builds); without
it the reference is picture only. That is a real limitation, not an omission.

**Legal combinations.** fl2va keyframes and ref2va reference blocks are
exclusive in the pipeline itself
([`minimax_h3_pipeline.cpp`](../src/vllm/model_executor/models/minimax_h3_pipeline.cpp)),
so the request parser enforces the same rule and returns a 400 naming the
offending pair rather than dropping a reference you supplied.

| `input_reference` | `metadata.input_reference_video` | `metadata.input_reference_audio` | |
|---|---|---|---|
| (none) | (none) | (none) | t2va, prompt only |
| image | (none) | (none) | fl2va, the image is frame 0 |
| (none) | clip | (none) | ref2va, silent video reference |
| (none) | (none) | WAV | ref2va, audio reference |
| (none) | clip | WAV | ref2va, one block carrying both |
| image | clip and/or WAV | | **400**: keyframe and reference conditioning are exclusive |

The video reference needs `--video-vae` (the encoder half of the same file) and
the audio reference needs `--audio-vae`; both load lazily, once, on the first
request that asks for them.

### The job lifecycle

`POST /v1/videos` returns immediately with `{"id": "vid_1", "status": "queued"}`;
generation is minutes long, so the synchronous twin `POST /v1/videos/sync` exists
for scripts that would rather block. `GET /v1/videos/{id}` reports `queued`,
`running`, `succeeded` (with `output_path`) or `failed` (with `error`).

`GET /v1/videos/{id}/content` returns the finished MP4 with
`Content-Type: video/mp4`. An unknown id is a 404; a job that has not finished is
a **409** naming its current status rather than a truncated file; a failed job is
a 500 carrying its failure; an output that has since vanished from disk is a 500
rather than a 200 with zero bytes.

The library never spawns a process, so generation and muxing enter through a
caller-supplied `VideoRunner` callback (`examples/server/main.cpp` supplies one
that invokes `ffmpeg`, path configurable with `--video-ffmpeg`).

## Consuming it as a library (C ABI)

Link `libvllm` (static or shared) and include [`include/vllm.h`](../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 10`,
19 exported symbols) suitable for `dlopen` / FFI / LocalAI integration.

```c
#include "vllm.h"

vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               /* sp.temperature = 0.0 means greedy */

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

The ABI covers lifecycle, blocking and streaming completion, non-blocking
concurrent requests, memory helpers, and diagnostics. Later ABI versions add:

| ABI | Adds |
|---:|---|
| v2 | Structured output (JSON schema, JSON object, regex, choice, GBNF) |
| v3 | Chat with tools and chat templates |
| v4 | Tool-parser selection |
| v5 | Reasoning-parser selection |
| v6 | Speculative decoding |
| v7 | Prefix caching (tri-state) |
| v8 | Custom logits processors |
| v9 | Engine sizing: chunked-prefill token budget, scheduling policy, external KV connector / LMCache |
| v10 | Jump-forward decoding (tri-state, default off) |

Chat templates render through the vendored google/minja engine, the same
renderer llama.cpp ships.

## Consuming it from C++

The higher-level surface lives under [`include/vllm/`](../include/vllm/).
`LoadedEngine::FromModelDir(...)`
([`entrypoints/model_loader.h`](../include/vllm/entrypoints/model_loader.h))
hands back either the synchronous `LLMEngine`
([`v1/engine/llm_engine.h`](../include/vllm/v1/engine/llm_engine.h)) or the async
`AsyncLLM` ([`v1/engine/async_llm.h`](../include/vllm/v1/engine/async_llm.h)) that
the server itself uses.

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_prefix_caching = true;
ep.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, ep);
```

The underlying portable tensor runtime is `vt::` ([`include/vt/`](../include/vt/)),
which carries no ggml or PyTorch dependency.

## Multimodal input (image, video, audio to text)

Multimodal input is served over the **OpenAI API**, not the CLI. `vllm-cli` is text-only:
`--model --prompt --max-tokens --temperature --top-k --top-p --seed --stream
--speculative-config --tokenizer-config`.

Start the server with a multimodal model, then send content parts on
`/v1/chat/completions`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.chat.completions.create(model="Qwen3.6-27B", messages=[{"role": "user", "content": [
    {"type": "text",      "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}},
]}])
```

Accepted part types (`src/vllm/entrypoints/openai/chat_mm.cpp`):

| part type | modality |
|---|---|
| `image_url` | image |
| `video_url` | video |
| `input_audio` / `audio_url` | audio |

## MiniMax-H3: video + audio generation


Renders an MP4 with a stereo track. Weights: a GGUF DiT (use **Q4_K_M**), the Qwen3-VL-32B
encoder, and both VAEs.

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-FL2VA-Q4_K_M.gguf --dequant-bf16 --partition fl2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "A golden retriever runs across a sunlit beach, waves crashing behind it" \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 768 --width 1344 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

`--partition` is REQUIRED and names the partition the checkpoint you passed
actually serves — see the trap above. This is the command every render in this
document was produced with: Q4_K_M DiT and encoder, `--dequant-bf16`, task
**t2va** (no reference image), the 1344x768 default canvas, 124 frames, 50 steps.

Cost, so you can plan: **~176 s per step** at 1344x768 / 124f on a 20-SM sm_110
device, so a 50-step render is about **2.5 hours** plus roughly 30 minutes of
weight loading. Dropping to 512x512 costs ~15 s/step (~13 minutes end to end),
which is the right canvas for iterating on a prompt before committing to a full
render. `--dequant-bf16` holds the DiT as bf16 (~66 GB resident); `--keep-quant`
is the low-memory arm.

Conditioning modes, all optional and mutually exclusive where noted:

```sh
--first-frame start.ppm --last-frame end.ppm   # pin the first and/or last frame (fl2va)
--ref-image subject.ppm                        # reference image, repeatable (ref2va)
                                               # NOT served by the FL2VA checkpoint above --
                                               # needs a Ref2VA partition (see the trap)
--ref-video prev_workdir/                      # reference clip, reads frame_%06d.ppm
--ref-audio voice.wav                          # reference audio
--noise-aug 0.9                                # how hard a keyframe is pinned (1.0 = exact)
```

Reference frames are binary PPM, which is what this tool also **writes**, so one run's `--workdir`
feeds straight back in as `--ref-video` and clips chain. Convert anything else with
`ffmpeg -i in.png -pix_fmt rgb24 out.ppm`.

Useful for measurement: `--denoise-only` times the DiT loop without loading the VAEs,
`--dump-params` prints the geometry a checkpoint implies (manifest only, no weights),
`--save-embeds` writes the text conditioning so a second run can replay it with
`--prompt-embeds` and compare checkpoints on identical conditioning.

Served over HTTP too: pass `--video-dit` (plus the VAEs and configs) to `examples/server` and
`POST /v1/videos`, `POST /v1/videos/sync` and `GET /v1/videos/{id}` register. Without it the
routes stay unregistered.
