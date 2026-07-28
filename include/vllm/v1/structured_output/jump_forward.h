// vllm.cpp ORIGINAL component (§9) — the SGLANG-DISTINCT jump-forward decoding
// primitive (sglang-matrix.md `SGLANG-CONSTRAIN-JUMP`, spec
// specs/sglang-radixattention.md §3 / SW3). Grounded in SGLang v0.5.15
// (f63458b) `python/sglang/srt/constrained/outlines_jump_forward.py:146-172`
// (the forced-run chain) + `managers/schedule_batch.py`@935cda944b^ :503-544
// (`jump_forward_and_retokenize`) / :1094-1145 (`check_for_jump_forward`, the
// scheduler splice — SGLang later REMOVED its scheduler wiring in commit
// 935cda944b, "Remove the support of jump forward").
//
// WHAT JUMP-FORWARD IS: when the grammar FSM has a deterministic forced
// continuation, emit it WITHOUT running the model per token (the speed win on
// constrained decoding).
//
// THE CORRECTNESS SUBTLETY (why this is a SUBSET, not the whole feature): the
// forced continuation is a byte STRING. Re-tokenizing `prefix + forced_string`
// can shift token boundaries versus appending token-by-token — SGLang handles
// this by re-tokenizing the whole text and ROLLING BACK the fused boundary
// token (schedule_batch.py `all_ids[prompt_tokens-1] != ...` guard). Emitting
// forced token IDs blindly therefore does NOT reproduce constrained decode.
//
// THE SAFE SUBSET WE LAND: the TOKEN-UNIQUE forced run. We jump ONLY while the
// grammar admits EXACTLY ONE valid token at a non-accepting state
// (StructuredOutputGrammar::forced_token). Because the constrained sampler masks
// all other tokens to -inf, that token is the argmax under ANY sampling params,
// so emitting it is PROVABLY byte-identical to per-token constrained decode and
// needs NO re-tokenization. The general byte-forced-but-multi-tokenizable span
// (which needs SGLang's re-tokenize + rollback) is DELIBERATELY not jumped here
// — forced_token returns nullopt (>=2 valid tokens) and we fall back to normal
// decode. That case is the NAMED RESIDUAL (see the spec).
#ifndef VLLM_V1_STRUCTURED_OUTPUT_JUMP_FORWARD_H_
#define VLLM_V1_STRUCTURED_OUTPUT_JUMP_FORWARD_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/v1/structured_output/backend_types.h"

namespace vllm::v1 {

// Whether jump-forward is enabled. Opt-in, DEFAULT OFF (mirror SGLang gating its
// jump-forward behind a flag; neither mirror engine defaults it on — spec §6.3).
// Reads env VT_ENABLE_JUMP_FORWARD once ("1"/"true" => on). The production
// scheduler splice (which must recompute KV for the jumped tokens) is a named
// residual, so the default stays off until that lands.
bool JumpForwardEnabled();

// Config-aware resolution (ENG-SGLANG-BEHAVIOR-FLAG). Mirrors the house
// AsyncSchedulingEnabled(resolved) convention (config/scheduler.h): the env var
// VT_ENABLE_JUMP_FORWARD, WHEN SET, is the override and decides ("1"/"true"/
// "TRUE"/"on" => on, anything else => off); when the env var is UNSET, the
// caller's `configured` value decides (the C-ABI vllm_model_params.
// enable_jump_forward / the C++ EngineParams.enable_jump_forward /
// --enable-jump-forward server flag), defaulting OFF when nullopt. Resolved ONCE
// at engine construction (LoadedEngine::jump_forward_enabled()), never per step.
// The no-arg JumpForwardEnabled() above is exactly JumpForwardEnabled(nullopt).
bool JumpForwardEnabled(std::optional<bool> configured);

// Drain the token-unique forced run from `grammar` at its CURRENT state,
// advancing the FSM one forced token at a time and appending each emitted id to
// `out`. Returns the number of tokens emitted (0 == inert: no forced span, or
// disabled). Each emitted token would have cost one model step under normal
// per-token constrained decode — that saving is the speed win.
//
// When `enabled` is false the grammar is NOT touched and 0 is returned (the
// default-off / no-grammar path is byte-identical to normal decode). `max_tokens`
// bounds the drain (pass the request's remaining token budget); a value <= 0
// drains until the forced run ends.
int DrainForcedTokens(StructuredOutputGrammar& grammar,
                      const std::string& request_id, std::vector<int32_t>& out,
                      bool enabled, int max_tokens = 0);

}  // namespace vllm::v1

#endif  // VLLM_V1_STRUCTURED_OUTPUT_JUMP_FORWARD_H_
