# Pooling task class — embeddings / classify / score / rerank (`CLAIM-POOLING`)

W0 spike for HIGH-priority feature-gap #2
([vllm-feature-gap-analysis.md](vllm-feature-gap-analysis.md)): the entire
NON-GENERATIVE task class. vLLM can return pooled hidden states (an embedding
vector, a classification/score logit) instead of sampled tokens; vllm.cpp today
has **no pooling runner, no pooler op, and no pooling model rows**. This spec
inventories the whole upstream pooling surface, fixes the exact port map, names
the tests, and lays out the W-breakdown. Pinned oracle
`/home/mudler/_git/vllm` @ `555967922` (vLLM 0.26.0.dev0).

## Scope

- **Rows.** New engine op row `ENG-POOLER-SEQ` (the sequence-pooling op:
  CLS/LAST/MEAN + the normalize/classify activation heads); the existing
  `SERVE-POOLING-ENDPOINTS` (embeddings / pooling / classify / score / rerank
  HTTP) moves `INVENTORIED → SPIKE` and points here. No pooling MODEL row is
  created in W1 (registering a BERT/embedding arch is W3+, and would trip the
  model-checklist rollup — deferred until a concrete arch is brought up).
- **In (W1, this change).** The CPU pooler OP: the sequence pooling methods
  (`CLSPool`/`LastPool`/`MeanPool` + `get_seq_pooling_method`) and the activation
  heads (`PoolerIdentity`/`PoolerNormalize`/`PoolerMultiLabelClassify`/
  `PoolerClassify`), over a packed `[num_tokens, hidden]` CPU hidden-state
  buffer keyed by a minimal `PoolingCursor`/`PoolingMetadata`. Unit-gated vs a
  double-precision reference, RED-first.
- **Out (W2+).** The pooler HEADS composite (`SequencePooler`, matryoshka dims,
  per-request activation flags, ST projector), the tokwise methods
  (`AllPool`/`StepPool`), `DispatchPooler` task routing, the pooling RUNNER
  (return pooled data instead of sampled tokens), the `/v1/embeddings` + score /
  rerank / classify endpoints and their protocol, the `PoolingParams` /
  `PoolerConfig` plumbing, and the concrete pooling models. Each is a named
  downstream brick below.

## Upstream chain

### The `Pooler` abstraction (`vllm/model_executor/layers/pooler/`)

- `abstract.py:16` — `Pooler(nn.Module, ABC)`: the interface every pooler
  implements — `get_supported_tasks()`, `get_pooling_updates(task)`,
  `forward(hidden_states, pooling_metadata) -> PoolerOutput`.
- `common.py:17` — `PoolingParamsUpdate(requires_token_ids)`; `ProjectorFn`,
  `ClassifierFn`, `ActivationFn` type aliases.
- `activations.py` — the activation heads: `PoolerActivation` base (`:84`),
  `PoolerIdentity` (`:106`), `PoolerNormalize` (`:111`, `F.normalize(p=2,
  dim=-1)`), `PoolerMultiLabelClassify` (`:116`, sigmoid),
  `PoolerClassify` (`:121`, sigmoid if `num_labels<2` else `softmax(dim=-1)`),
  `LambdaPoolerActivation` (`:147`); the config-driven factories `get_act_fn`
  (`:19`, dispatches on `problem_type` / sentence-transformers config) and
  `resolve_classifier_act_fn` (`:63`).
- `seqwise/methods.py` — the SEQUENCE pooling methods (one pooled vector per
  sequence): `SequencePoolingMethod` base (`:22`), `CLSPool` (`:36`, first
  token, rejects partial prefill), `LastPool` (`:49`, last scheduled token,
  allows partial prefill), `MeanPool` (`:60`, float32-accumulated mean, rejects
  partial prefill, chunked accumulation), `get_seq_pooling_method` (`:109`).
- `seqwise/heads.py` — the sequence pooler HEADS: `SequencePoolerHead` base
  (`:19`), `EmbeddingPoolerHead` (`:32`, dtype cast → ST projector → matryoshka
  dim slice → optional activation), `ClassifierPoolerHead` (`:123`, classifier →
  affine `(logit-mean)/sigma` calibration → activation).
- `seqwise/poolers.py` — `SequencePooler` (`:41`, `pooling` then `head`);
  `pooler_for_embed` (`:99`) / `pooler_for_classify` (`:113`) factories.
- `tokwise/{methods,heads,poolers}.py` — the TOKEN pooling variants
  (`AllPool`/`StepPool`, late-interaction / token-classify).
- `special.py:23` — `DispatchPooler`: routes `forward` to a sub-pooler by task;
  `for_embedding` / `for_seq_cls` classmethods wire the task→pooler map.
- `config/pooler.py:16` — `SequencePoolingType = "CLS"|"LAST"|"MEAN"`;
  `PoolerConfig` (`:30`) with `seq_pooling_type`, `logit_mean`, `logit_sigma`,
  `get_seq_pooling_type()`.

### The pooling runner path (`vllm/v1/worker/gpu/pool/pooling_runner.py`)

- `PoolingRunner` (`:18`) holds the `VllmModelForPooling`. `pool()` (`:29`)
  gathers `hidden_states[logits_indices]` (LAST pooling), L2-normalizes, and
  returns `(pooled, is_valid)` — the runner returns POOLED DATA where the
  generation runner would sample tokens. `get_supported_tasks` (`:22`) currently
  admits only `"embed"` on decoder-only models (upstream `NOTE`: LAST-only for
  now). The pooled result flows back as `PoolerOutput` (`v1/outputs.py`).
- `v1/pool/metadata.py` — `PoolingCursor` (`:13`, first/last token index tensors
  + `prompt_lens_cpu`/`seq_lens_cpu`/`num_scheduled_tokens_cpu`,
  `is_partial_prefill()` `:31`), `PoolingMetadata` (`:47`), `PoolingStates`
  (`:36`, chunked-prefill ALL-pooling hidden-state cache).

### Task taxonomy (`vllm/tasks.py`)

- `GenerationTask = generate|transcription|realtime` (`:7`);
  `PoolingTask = embed|classify|token_embed|token_classify|plugin|
  embed&token_classify` (`:10`). `score` and `encode` are REMOVED aliases
  (`:20`, mapped to `classify` / `token_embed|token_classify`). `SCORE_TYPE_MAP`
  (`:33`) keys embed→bi-encoder, classify→cross-encoder,
  token_embed→late-interaction.

### Endpoints (`vllm/entrypoints/pooling/`)

- `embed/api_router.py:28` — `POST /v1/embeddings` (+ `/v2/embed` Cohere).
- `scoring/api_router.py:37,71` — `POST /score`, `/v1/score`, `/rerank`,
  `/v1/rerank`, `/v2/rerank`.
- `classify/api_router.py:26` — `POST /classify`, `/pooling`.
- Handlers `ServingEmbedding` / `ServingScores` / `ServingClassify` build a
  `PoolingParams(task=...)`, run the pooling runner, and format the pooled
  vectors/logits into the OpenAI embedding / score / rerank response.

### A concrete pooling model (reference for W3)

- `models/bert.py` — `BertEmbeddingModel` / `BertForSequenceClassification`:
  encoder-only, CLS or MEAN pooling + `DispatchPooler.for_embedding` /
  `for_seq_cls`. `models/roberta.py` mirrors it. Qwen/LLaMA embedding variants
  (`*EmbeddingModel`, LAST pooling) reuse the decoder backbone + a
  `DispatchPooler.for_embedding` — the cheapest first bring-up because the
  backbone forward already exists in our tree.

## Our baseline

- **No pooler op, no runner, no endpoints, no pooling model.** `grep -rin
  pooler src/ include/` is empty. The generation runner
  (`src/vllm/v1/worker/gpu/runner.cpp`) only samples tokens.
- **Reusable substrate present:** the `vt::Tensor` CPU runtime, the dense/
  encoder backbones (Qwen3/Llama/BERT-shaped attention exist for decoder
  families), and the `linear.h` GEMM seam for a classifier head.
- Records: `SERVE-POOLING-ENDPOINTS` (engine-matrix, `INVENTORIED`),
  `MODEL-POOLING` (feature-matrix rollup, `INVENTORIED`). No pooling model rows
  in `model-matrix.md`.

## Port map

| Upstream | Local (this change, W1) | Notes |
|---|---|---|
| `pooler/seqwise/methods.py:35-121` | `include/vllm/model_executor/layers/pooler/methods.h` + `src/.../pooler/methods.cpp` | `SequencePoolingMethod`, `CLSPool`/`LastPool`/`MeanPool`, `GetSeqPoolingMethod`; MeanPool uses the packed per-seq contiguous layout (equivalent to upstream `segment_ids` for the non-partial case it requires) |
| `pooler/activations.py:106-158` | `include/vllm/.../pooler/activations.h` + `src/.../pooler/activations.cpp` | `PoolerIdentity`/`PoolerNormalize`/`PoolerMultiLabelClassify`/`PoolerClassify` row-wise, in place |
| `v1/pool/metadata.py:13-71` | `include/vllm/.../pooler/pooling_metadata.h` | `PoolingCursor` (host index/length vectors) + minimal `PoolingMetadata`; heads/tokwise/token-id fields deferred (documented) |
| `config/pooler.py:16` | `SequencePoolingType` enum in `methods.h` | CLS/LAST/MEAN |
| **W2+ (deferred, named)** | | |
| `pooler/seqwise/heads.py`, `poolers.py`, `special.py` | `pooler/heads.{h,cpp}`, `pooler/dispatch_pooler.{h,cpp}` | matryoshka, projector, `SequencePooler`, `DispatchPooler` task routing |
| `pooler/tokwise/*` | `pooler/tokwise.{h,cpp}` | `AllPool`/`StepPool`, needs `prompt_token_ids` in metadata |
| `v1/worker/gpu/pool/pooling_runner.py` | `src/vllm/v1/worker/gpu/pool/pooling_runner.cpp` | return pooled data from the runner |
| `entrypoints/pooling/{embed,scoring,classify}/*` | `src/vllm/entrypoints/pooling/*` | `/v1/embeddings`, `/score`, `/rerank`, `/classify`, `/pooling` + protocol |
| `models/bert.py` (or a `*EmbeddingModel`) | `src/vllm/model_executor/models/bert.cpp` (or reuse a decoder backbone) | first concrete pooling model → `model-matrix.md` row + checklist |

## Tests to port

| Upstream | Local | Status |
|---|---|---|
| `tests/model_executor/layers/test_pooler_methods.py` (`TestCLSPool`, `TestLastPool`, `TestMeanPool`, `TestGetSeqPoolingMethod`) | `tests/vllm/model_executor/layers/pooler/test_pooler.cpp` | **W1 ported + passing** (double-precision references; the `AllPool`/`StepPool` classes deferred with the tokwise brick) |
| `tests/model_executor/layers/test_pooler_activations.py` (Identity/Normalize/MultiLabelClassify/Classify) | same file | **W1 ported + passing**; the `get_act_fn` config-factory cases deferred with the head brick |
| `tests/model_executor/layers/test_pooler_heads.py` (`TestEmbeddingPoolerHead`, `TestClassifierPoolerHead`) | `tests/vllm/model_executor/layers/pooler/test_pooler_heads.cpp` | **W2 ported + passing** (240 asserts, double-precision refs; + SequencePooler/DispatchPooler composite). The tokwise `TestTokenEmbedding/TokenClassifierPoolerHead` classes deferred with the tokwise (W5) brick; the torch `list_input_gets_stacked` / `head_dtype` cases are return-type/dtype nuances not ported (recorded deviation) |
| `tests/models/language/pooling/test_embedding.py` (real-oracle cosine gate) | `tests/vllm/v1/worker/gpu/pool/test_pooling_runner.cpp` | **W3 runner path ported + passing** as a STRUCTURAL cosine gate (14 asserts vs a double-precision LAST+normalize reference, RED-first). The REAL-model `vllm.LLM(task="embed").encode` oracle cosine gate remains SKIPPED-DEFERRED (W3-model): needs a registered concrete embedding model forward — no cosine-vs-oracle number is fabricated |
| `tests/entrypoints/pooling/embed/*` | — | SKIPPED-DEFERRED (W4 endpoint brick): needs the OpenAI serving layer |

## Gates

- **W1 (this change):** `test_pooler` doctest unit gate, CPU `-Wall -Wextra
  -Werror` clean, all cases GREEN, RED-first proven (disabling the MeanPool
  division / PoolerNormalize denominator fails the arithmetic subcases). No
  correctness-vs-oracle or throughput gate is owed by a pooler-op unit brick
  (`benchmark_binding=false`) — the double-precision references ARE the gate.
- **W3/W4 (future):** a concrete pooling model end-to-end vs the pinned vLLM
  oracle (`vllm.LLM(task="embed").encode(...)`), cosine-parity on the embedding
  vector; and the endpoint response-shape parity. Named here, not owed now.

## Dependencies

- W1: none beyond `vt::Tensor` (CPU) — landed.
- Pooler HEADS (W2): `PoolingParams` (matryoshka `dimensions`, `use_activation`)
  — a small params struct, no engine change.
- Pooling RUNNER (W3): the `is_pooling_model` seam + a `PoolerOutput` return path
  from `runner.cpp`; a concrete model backbone.
- Endpoints (W4): the runner + the OpenAI serving layer (reuses
  `serving_utils`), and `RunnerType`/`ConvertType` (`config/model.py`) selection.

## Work breakdown

1. **W1 — pooler OP (this change).** Sequence methods + activation heads +
   minimal metadata, CPU, unit-gated. `ENG-POOLER-SEQ` → `ACTIVE`;
   `SERVE-POOLING-ENDPOINTS` → `SPIKE`.
2. **W2 — pooler HEADS + `SequencePooler` + `DispatchPooler`** (matryoshka /
   projector / classifier calibration / task routing) + `PoolerConfig`/
   `PoolingParams`; port `test_pooler_heads.py`.
3. **W3 — pooling RUNNER (`ENG-POOLING-RUNNER`) + first concrete pooling model**
   (a `*EmbeddingModel` reusing a landed decoder backbone, or
   `BertEmbeddingModel`); `model-matrix.md` row + checklist + rollup; oracle
   cosine-parity gate. **RUNNER PATH LANDED 2026-07-29 (`CLAIM-POOLING`):**
   `PoolingRunner` (`include/vllm/v1/worker/gpu/pool/pooling_runner.h`) applies the
   model's `Pooler` (`DispatchPooler`) to the last hidden state and returns pooled
   data instead of sampled tokens, structurally cosine-gated RED-first
   (`test_pooling_runner.cpp`). RESIDUAL: the concrete embedding MODEL forward +
   the REAL-model `vllm.LLM(task="embed").encode` oracle cosine gate (no such
   model registered yet — structural gate only, no cosine-vs-oracle number
   fabricated).
4. **W4 — endpoints** `/v1/embeddings`, `/score` + `/rerank`, `/classify` +
   `/pooling` + protocol; `SERVE-POOLING-ENDPOINTS` → implementation; port
   `tests/entrypoints/pooling/*`.
5. **W5 — tokwise** (`AllPool`/`StepPool`, late-interaction / token-classify) +
   chunked-prefill ALL pooling (`PoolingStates`).

## Risks / decisions

- **No model row in W1.** Registering a pooling arch bumps the model-matrix
  rollup + checklist; the pooler OP is independently gateable without it, so the
  model lands in W3 with a real oracle gate (keeps the sweep honest).
- **MeanPool layout.** Upstream builds `segment_ids` via `repeat_interleave` and
  `index_add_` over the whole packed buffer, chunked by
  `_MEAN_POOL_ACCUMULATION_CHUNK_BYTES`. Because MeanPool REJECTS partial
  prefill, each sequence's tokens are contiguous in the packed buffer, so
  summing `prompt_lens[i]` rows from `first_token_indices[i]` is
  value-equivalent; the chunking is a memory optimization with no numeric effect
  (float32 accumulation is preserved). Recorded deviation.
- **vLLM-defined behavior is not reopened** — CLS rejects partial prefill, LAST
  allows it, MeanPool upcasts to float32; all mirrored.
