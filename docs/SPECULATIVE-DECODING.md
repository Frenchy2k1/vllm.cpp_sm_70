# Speculative decoding (MTP)

vllm.cpp ships MTP (Multi-Token Prediction) speculative decoding: a small draft
head proposes the next token, the full model verifies it in the same step, and an
accepted proposal skips a full model step. It is **opt-in and off by default**;
with no speculative config the engine is byte-identical to the non-speculative
one.

Selection mirrors vLLM's own CLI: `--speculative-config '<json>'`, taking the
same JSON object vLLM takes, so a config written for vLLM works here.

## What it does and where it works

- **Method:** MTP only, at `num_speculative_tokens` = 1 (k=1).
- **Models:** the Qwen3.5 / 3.6 gate checkpoints that ship an `mtp.*` draft head
  in their safetensors (Qwen3.6-27B and Qwen3.6-35B-A3B). Both are GDN hybrids,
  and the speculative path is wired through the linear-attention (GDN) recurrence
  and short causal convolution as well as the attention layers.
- **Correctness:** at concurrency 1 the speculative-on greedy output is
  token-for-token identical to both the speculative-off output and vLLM's own MTP
  speculative greedy output on the same prompt.
- **Speed:** measured about 1.04x faster than vLLM's own speculative-on decode at
  concurrency 1 (see [Measured result](#measured-result)).

## DSpark (semi-autoregressive block drafting) — in progress

DSpark drafts a whole block in one parallel pass and then adds intra-block
dependency with a small sequential head, so a block draft stops being k
conditionally independent guesses. It is the DFlash drafter plus a low-rank
Markov transition bias.

Current state (`SPEC-DSPARK`): the config, the Markov head and draft model, both
published checkpoint layouts (native `deepseek-ai/dspark_qwen3_*` and
Speculators `RedHatAI/*.dspark`), the sequential sampler and the runner wiring
have landed, and the path runs end to end on the Qwen3.6-35B-A3B gate model.
**It is not working yet.** Tracing shows the drafter proposes eight plausible
tokens every step, but none of them are ever installed on the scheduler or
verified, speculation is 5.5x slower than plain decode, and the speculative-on
output is not stable run to run. No correctness or speed number is claimed. A
GGUF target, and a target architecture with no aux multi-tap, are both refused
by name.

```bash
main --model /models/Qwen3-4B \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":7}'
```

`num_speculative_tokens` is required for a DSpark draft (a native DSpark config
carries no `n_predict`), and it must be at least the checkpoint's block size —
a smaller value produces incorrect output rather than merely lower acceptance,
so it is rejected.

## The flag

On the OpenAI server:

```bash
server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

On the example CLI:

```bash
vllm-cli --model /models/Qwen3.6-27B \
  --prompt "The capital of France is" --max-tokens 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Over the C ABI (`include/vllm.h`, ABI v6):

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/models/Qwen3.6-27B";
mp.speculative_config = "{\"method\":\"mtp\",\"num_speculative_tokens\":1}";
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);   /* NULL/"" speculative_config => no speculation */
```

The JSON is parsed into the same `vllm::SpeculativeConfig` the C++ API takes
programmatically (`EngineParams::speculative_config`). `method` must be `mtp` and
`num_speculative_tokens` must be 1; a malformed document, an unsupported method,
or a checkpoint with no `mtp.*` head fails the load loudly at startup rather than
running silently without speculation.

## Measured result

Concurrency 1, Qwen3.6-27B (NVFP4) on GB10, our speculative-on decode versus vLLM
0.25.0 running the same speculative config:

| Metric | Ours (spec on) | vLLM (spec on) |
|---|---:|---:|
| Time per output token, prose | 66.2 ms | 69.1 ms |
| Time per output token, code | 62.95 ms | 65.3 ms |
| Decode throughput, prose | 15.1 tok/s | 14.4 tok/s |
| Decode throughput, code | 15.7 tok/s | 15.1 tok/s |
| Drafter acceptance rate | 0.85 prose / 0.92 code | 0.84 |

Speculation helps both engines about 1.5 to 1.6x at this operating point, and our
engine is already about 4% faster than vLLM with speculation off, so the lead is
preserved with it on. The extra state speculation needs (a doubled recurrent-state
slot at k=1, plus the draft cache and head) costs about 3.6 GB, well inside the
box's unified memory. The full A/B, including the higher-concurrency numbers, is
in [BENCHMARKS.md](BENCHMARKS.md).

## Concurrency above 1

The concurrent (multi-request) path, where a speculative request shares a
scheduler step with an ordinary prefill, is implemented. The GDN layer splits the
batch into its speculative and ordinary rows, runs each through its own recurrence
over disjoint per-request state, and merges the results back; because the state is
disjoint, the mixed batch's output equals the two requests run alone, bit-for-bit,
at the real 27B and 35B dimensions. Speculation keeps helping at concurrency 2, 4
and 8 on the 27B (about 1.5x our own speculative-off throughput).

## Limitations

- **k=1 only.** `num_speculative_tokens` greater than 1 is not accepted.
- **MTP only.** Other draft methods (n-gram, EAGLE, a separate draft model, the
  DFlash draft) are not wired; MTP is the one that ships.
- **Qwen3.5/3.6 with an `mtp.*` head only.** The target may be a safetensors
  directory or a `.gguf` converted WITH the head (llama.cpp's layer-indexed
  `nextn` block); a GGUF exported `--no-mtp` is refused and says so.
- **On an NVFP4 safetensors target the concurrency-1 identity below is not
  currently reliable.** Measured on the 35B A3B NVFP4 safetensors: its logits
  land on a coarse grid that yields EXACT ties between distinct tokens, and at
  such a tie speculative-on and speculative-off, and even two speculative-off
  runs, can pick differently. The same weights loaded from GGUF, which expands
  to bf16, show no such ties and are token-identical. Open; see
  [docs/STATUS.md](STATUS.md).
- **Concurrency above 1 is not token-stable for the 27B.** Its greedy output is
  not bit-stable across batch shapes even with speculation off (changing the batch
  size flips a few near-tie tokens), so exact token-for-token agreement between
  speculative-on and speculative-off is a concurrency-1 property only. Correctness
  above concurrency 1 rests on the model-independent bit-exact GDN split-merge
  proof and on matching drafter acceptance, not on identical token streams.

## Consuming it programmatically

The flag is a thin wrapper over the library surface:

```cpp
vllm::entrypoints::EngineParams params;
params.speculative_config =
    vllm::ParseSpeculativeConfigJson(R"({"method":"mtp","num_speculative_tokens":1})");
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, params);
```

Leaving `speculative_config` unset (the default) loads the ordinary engine, which
is byte-identical to a build with no speculative code compiled in.
