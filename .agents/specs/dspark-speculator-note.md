# DSpark speculator — grounding note (rider on task #287, 2026-08-08)

USER-requested 2026-08-08; INVENTORIED only — full scoping is its own future
spike (`planned: specs/dspark-spec-decode.md`). Grounded at pin `555967922`:

1. **Drafter:** `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:37` —
   `DSparkSpeculator(DFlashSpeculator)` drafts a BLOCK of
   `num_speculative_tokens` in ONE parallel pass (anchor + N−1 noise queries,
   `:5-11,50-52`); `utils.py` sits beside it.
2. **Config:** method `"dspark"` in `vllm/config/speculative.py:62,310`
   (target_model_config required `:706-709`; default draft
   `deepseek-ai/dspark_qwen3_8b_block7` `:875`); **V2-runner-ONLY** — forced at
   `vllm/config/vllm.py:560-568` and listed with DFlash as "parallel drafting
   natively in V2 via their own speculators" `:2173-2177`.
3. **Draft models:** `vllm/model_executor/models/qwen3_dspark.py:36`
   (`DSparkMarkovHead`) + `:95` (`Qwen3DSparkForCausalLM`, extends the DFlash
   Qwen3 draft) and `gemma4_dspark.py:134/:182`; registry entries
   `registry.py:609,611`.
4. **Both target families are OURS:** Qwen3.6
   (`src/vllm/model_executor/models/qwen3_5_moe.cpp:176`,
   `qwen3_5_dense.cpp:199`) and Gemma4 (`gemma4_registry.cpp:215`).
5. **Slots beside our landed speculator lanes** (same directory shape as
   upstream): MTP `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp` and
   DFlash `.../spec_decode/dflash/speculator.cpp` — DSpark extends DFlash
   upstream, so the DFlash lane is the reuse base.
