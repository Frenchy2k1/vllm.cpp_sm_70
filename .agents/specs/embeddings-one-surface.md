# Embeddings/pooling through the ONE surface (`ARCH-ONE-SURFACE` fold ROW 6)

Row: `ARCH-ONE-SURFACE` leaf, branch `row/EMBEDDINGS-ONE-SURFACE` (task #285).
The audit found the engine-side pooler EXISTS (`ENG-POOLER-SEQ` W1/W2 ops +
`ENG-POOLING-RUNNER` W3 `PoolingRunner`) but is never invoked live: every
registered arch has `is_pooling_model=false`, `/v1/embeddings` is an explicit
residual, and the ABI has no embed entry. This row folds it live:
registry arch -> engine-step invocation -> `vllm_embed` (ABI v15) ->
live `/v1/embeddings`.

## Scope

Rows: `ENG-POOLING-RUNNER` (live engine-step invocation),
`SERVE-POOLING-ENDPOINTS` (`/v1/embeddings`),
`MODEL-EMBED-llama-llama-for-causal-lm` (the `LlamaModel` membership),
cross-ref `ENG-POOLER-SEQ` (landed ops, untouched).

- **In.** (W1) Register the smallest upstream-mirrored embedding arch the
  existing code supports: **`LlamaModel`** — upstream `_EMBEDDING_MODELS`
  maps `"LlamaModel": ("llama", "LlamaForCausalLM")`
  (`vllm/model_executor/models/registry.py:230`) and converts via
  `as_embedding_model` (`adapters.py:230`): the CAUSAL backbone forward with
  the lm_head removed + `DispatchPooler.for_embedding` (LAST pooling default,
  `interfaces_base.py:160`; normalize head). NO new model is built — the
  registered forward IS the shared `Qwen3DenseModel` machinery Llama already
  routes (`llama.h:39-40`), minus lm_head, plus the landed `DispatchPooler`.
  (W1) Invoke the landed `PoolingRunner` in the ENGINE STEP for pooling-task
  requests: the runner builds a `PoolingRunner` iff the loaded model is a
  pooling model (mirror `gpu/model_runner.py:368-369`) and `sample_tokens`
  routes to a pooling branch that returns POOLED DATA instead of sampled
  tokens (mirror `gpu/model_runner.py:1586-1607` + `pool/pooling_runner.py:29-42`);
  the scheduler finishes a pooling request as soon as pooled output exists
  (mirror `v1/core/sched/scheduler.py:1718-1721`), the output carries it to the
  frontend (`scheduler.py:1837`), and async scheduling resolves OFF for pooling
  models (mirror `vllm/config/vllm.py:1068-1073`, our
  `SchedulerConfig::ResolveAsyncScheduling` is_pooling_model arm — already
  landed, now WIRED). Text-generation on the pooling arch and embed on a text
  arch refuse cleanly both directions (the #121 refuse-by-task precedent).
  (W2) `vllm_embed` on `include/vllm.h`: engine handle + text(s) in, float
  vector(s) out, explicit free, `vllm_last_error`; `VLLM_ABI_VERSION` 14->15,
  floor pin advanced. (W3) live `/v1/embeddings` (OpenAI shape), registered
  task-conditionally; socket-level 404 pins BOTH directions. (W4) allowlist
  row removed, FEATURES row -> reachable, records.
- **Out (named residuals).** A REAL embedding checkpoint (e5-mistral /
  Llama-embed class) through the fold on real hardware — not fetchable
  CPU-side in this session; the committed synthetic-fixture arm is the gate
  (the #121 precedent). `MistralModel`/`Qwen2Model`/`Gemma2Model` etc.
  aliases (upstream registry.py:215-260) — additive follow-ups. Matryoshka
  `dimensions` + `encoding_format` on the endpoint/ABI; `/pooling`, `/score`,
  `/rerank`, `/classify`; tokwise pooling (W5); classify heads live-wiring.
  `vllm_embed` batches sequentially through the synchronous engine (no
  AsyncLLM fan-out).

## Upstream chain (pinned `${VLLM_SOURCE}` @ 555967922, 0.26.0.dev0)

- Task selection: `LLM.embed` -> `pooling_task="embed"`
  (`vllm/entrypoints/pooling/offline.py:47-119`); `--runner pooling`
  resolution `vllm/config/model.py:1008-1030`; `convert="embed"` default for
  pooling runners `model.py:1058-1060`; arch->pooling class
  `registry.py:215-260` (`"LlamaModel": ("llama", "LlamaForCausalLM")` :230);
  `as_embedding_model` wrap `adapters.py:230-261` (pooler =
  `DispatchPooler.for_embedding`, :257), lm_head replaced by a missing-layer
  stage (`adapters.py:135-151`), checkpoint loadable from BOTH `*ForCausalLM`
  and bare `*Model` prefixes (`adapters.py:178-181` candidate_prefixes
  `["", "model."]`), LAST default (`interfaces_base.py:160`).
- Engine step: `PoolingRunner` built iff pooling model
  (`v1/worker/gpu/model_runner.py:368-369`); pool-instead-of-sample
  (`model_runner.py:1586-1607`); `pool()` = gather at `logits_indices` +
  normalize, `is_valid = seq_lens == prompt_len`
  (`pool/pooling_runner.py:29-42`).
- Scheduler: pooling stops as soon as there is output
  (`v1/core/sched/scheduler.py:1718-1721`), `pooling_output` on the
  EngineCoreOutput (:1837), emit condition includes pooler output (:1792);
  `check_stop` asserts non-pooling (`sched/utils.py:95`).
- Config: async scheduling disabled for pooling (`config/vllm.py:1068-1073`);
  decoder+LAST pooling SUPPORTS chunked prefill and prefix caching
  (`config/model.py:1883-1902,1929-1948`) so those defaults stay untouched.
- Endpoint: `POST /v1/embeddings` (`entrypoints/pooling/embed/api_router.py:28-43`),
  handler absent => "does not support" (:22-25); response shape
  `entrypoints/pooling/embed/protocol.py` (OpenAI list/data/usage).

## Our baseline

- Landed, test-only: `layers/pooler/*` (methods/activations/heads/poolers/
  dispatch_pooler), `v1/worker/gpu/pool/pooling_runner.{h,cpp}`,
  `tests/vllm/v1/worker/gpu/pool/test_pooling_runner.cpp` (STRUCTURAL
  double-precision LAST+normalize cosine gate — NOT recorded-oracle fixtures;
  the qwen36_embed goldens are token-embedding LOOKUP goldens, unrelated).
- `ModelInfo.is_pooling_model` exists (all false);
  `SchedulerConfig::ResolveAsyncScheduling(is_pooling_model)` arm landed but
  passed `false` at the only call site (`model_loader.cpp:552`).
- Engine plumb points already marked: `scheduler.cpp` "DEFERRED: pooling
  stop.", `output_processor.cpp:216,433` pooling-deferred markers,
  `input_batch.h:66`, `llm_engine.h` deferred list.
- Precedents: #121 (Parakeet: refuse-by-task registry + capi guard +
  task-conditional route + socket 404 pins + committed tiny fixtures), #123
  route-table 404s, #136 append-only ABI field growth.

## Port map

| Upstream | Local | Notes |
|---|---|---|
| `adapters.py:230-261` + `registry.py:230` | `src/vllm/model_executor/models/llama_embedding_registry.cpp` (NEW TU) | `REGISTER_VLLM_MODEL(.., "LlamaModel", ..)`, `is_pooling_model=true`, `is_text_generation_model=false`; loader accepts both name prefixes, never loads lm_head; LoadedModel owns `DispatchPooler::ForEmbedding(cfg, kLast)` |
| model returns hidden states (no lm_head) | `Qwen3DenseModel::ForwardHidden` (additive in `qwen3.h`/`qwen3.cpp`) | shared ForwardLayers tail gains a return-hidden arm; text callers byte-identical |
| `VllmModelForPooling.pooler` | `LoadedModel::pooler()` virtual (default nullptr) | additive on the type-erased base |
| `model_runner.py:368-369,1586-1607` | `GPUModelRunner::pooling_runner_` + pool branch in `sample_tokens` (`runner.cpp`) | ADDITIVE + model-task-gated; text path untouched |
| `outputs.py ModelRunnerOutput.pooler_output` | `ModelRunnerOutput::pooler_output` (`v1/engine/types.h`) | `vector<optional<vector<float>>>`, empty on generation steps |
| `request.py pooling_params` | `Request::pooling_params` + `EngineCoreRequest::pooling_params` | `optional<PoolingParams>`, nullopt = generation, byte-identical |
| `scheduler.py:1718-1721,1792,1837` | the marked elif in `Scheduler::update_from_output` | pooling stop + `EngineCoreOutput::pooling_output` |
| `output_processor.py` pooling branch | `RequestState::make_request_output` + `process_outputs` pooling arm | `RequestOutput::pooling_output` (optional vector; deviation: no separate PoolingRequestOutput class, recorded) |
| `offline.py LLM.embed` | `LLMEngine::add_pooling_request` + `embed()` driver | tokens overload + step loop |
| `config/vllm.py:1068-1073` | pass `info.is_pooling_model` at `model_loader.cpp:552` | wires the landed arm |
| no upstream C ABI (llama.h idiom) | `vllm_embed` / `vllm_embedding_result(_free)`, ABI v15 | `src/capi/vllm_c.cpp`; refuse both directions |
| `pooling/embed/api_router.py:28` + protocol | `handle_embeddings` in `api_server.{h,cpp}` + `set_embedding`; server main task dispatch | task-conditional route registration, 404 both ways |

## Tests to port

- `tests/models/language/pooling/test_embedding.py` (real-oracle cosine) ->
  the registry-anchored fold arm below; the REAL-checkpoint leg stays the named
  residual (no cosine-vs-oracle number fabricated).
- `tests/entrypoints/pooling/embed/*` (endpoint shape) ->
  `tests/vllm/entrypoints/openai/test_api_server.cpp` embeddings section.

## Gates

- Fold gate `tests/vllm/models/test_llama_embedding_fold.cpp`: committed tiny
  synthetic `LlamaModel` checkpoint fixture (deterministic generator script,
  #121 precedent) driven through `LoadedEngine::FromModelDir` ->
  `add_pooling_request` -> `step` (the REGISTRY/RUNNER path), asserting
  (a) IDENTICAL vectors vs the direct `ModelRegistry::Forward`+`PoolingRunner`
  path, (b) the pooling lane's double-precision LAST+normalize reference
  (cosine ~= 1, unit L2) — the lane's cosine gate now RUNS THROUGH the
  registry/runner; (c) refuse-both-directions.
- `test_pooling_runner.cpp` continues to pass unchanged (the op-level gate).
- `test_capi`: v15 floor pin (>= 15), `vllm_embed` bad-path contract + the
  REAL fixture-checkpoint smoke through `vllm_engine_load`+`vllm_embed`,
  refuse-both-directions pins.
- `test_openai_api_server`: `/v1/embeddings` handler shape + socket-level 404
  pins BOTH directions (route absent on text servers; generate routes absent
  on embedding servers).
- `vllm_capi_c_check` (c_header_compile.c references the new symbols).
- Full CPU `-Werror` build; `scripts/agent-preflight.sh` EXIT=0;
  mutation-verify every pin RED-first.

## Dependencies

- Landed pooling ops (`ENG-POOLER-SEQ` W1/W2) + `PoolingRunner` (W3); the
  shared dense backbone (`LlamaModel == Qwen3DenseModel`); the registry, capi
  and server seams; the #121/#123/#136 fold precedents.

## Risks / decisions

- Depends only on landed code (pooler ops, runner, registry, capi, server).
- Risk: the pooling branch must not perturb the SACRED text path — every hook
  is task-gated on `is_pooling_model` (registration-time constant) or
  `pooling_params.has_value()`; default nullopt/false everywhere.
- Risk: chunked prefill of pooling prompts — handled by the same
  discard/is_valid predicate the text path uses (`seq_len < num_tokens`),
  mirroring `pooling_runner.py:40-41`; pooled output only when fully
  prefilled.
- Recorded deviation: `vllm_embed`/`/v1/embeddings` drive the SYNCHRONOUS
  LLMEngine (embed is blocking request/response); AsyncLLM stays
  generation-only.

## Work breakdown

1. **W1 — registry + runner (landed on this row):** `LlamaModel` arch TU +
   bare-prefix loader + `ForwardHidden` pooling forward; `PoolingRunner`
   invoked task-gated in the engine step; scheduler pooling stop; refusals
   both directions.
2. **W2 — ABI (landed on this row):** `vllm_embed` + `vllm_embedding_result_free`,
   VLLM_ABI_VERSION 15, floor pin advanced, strict-C references, dlopen symbols.
3. **W3 — server (landed on this row):** task-conditional `/v1/embeddings` +
   both-direction socket 404 pins; server main pooling dispatch.
4. **W4 — guard/records (landed on this row):** allowlist row removed,
   FEATURES/STATUS/BENCHMARKS/matrices/specs updated.
5. **W-next (residuals, unclaimed):** real embedding checkpoint +
   `LLM(task="embed")` oracle cosine; `MistralModel`/other membership aliases;
   `/pooling`+score/rerank/classify; matryoshka `dimensions`/base64/token-array
   inputs; tokwise pooling (W5 of the pooling lane).
