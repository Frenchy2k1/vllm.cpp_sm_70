// vllm.cpp ORIGINAL component (§9). See jump_forward.h for scope + the SGLang
// grounding and the safe-subset correctness argument.
#include "vllm/v1/structured_output/jump_forward.h"

#include <cstdlib>
#include <optional>
#include <string>

namespace vllm::v1 {

bool JumpForwardEnabled(std::optional<bool> configured) {
  const char* e = std::getenv("VT_ENABLE_JUMP_FORWARD");
  if (e != nullptr) {
    // Env override present: it decides, mirroring AsyncSchedulingEnabled's
    // VT_ASYNC_SCHED convention ("1"/"true"/"TRUE"/"on" => on, else off).
    const std::string v(e);
    return v == "1" || v == "true" || v == "TRUE" || v == "on";
  }
  // No env override: the config/ABI field decides (default OFF when unset).
  return configured.value_or(false);
}

bool JumpForwardEnabled() { return JumpForwardEnabled(std::nullopt); }

int DrainForcedTokens(StructuredOutputGrammar& grammar,
                      const std::string& request_id, std::vector<int32_t>& out,
                      bool enabled, int max_tokens) {
  if (!enabled) return 0;
  int emitted = 0;
  while (max_tokens <= 0 || emitted < max_tokens) {
    // forced_token returns a valid non-stop token ONLY when it is the UNIQUE
    // grammar-valid token at a non-accepting state (see backend_types.h). That
    // makes emitting it byte-identical to per-token constrained decode.
    const std::optional<int32_t> tok = grammar.forced_token();
    if (!tok.has_value()) break;
    // Advance the FSM by the forced token, exactly as the scheduler would after
    // sampling it (SGLang's jump_and_retokenize updates the grammar state). The
    // token is bitmask-allowed, so accept_tokens accepts it (fill_bitmask and
    // accept_tokens are in exact agreement); the guard is purely defensive.
    if (!grammar.accept_tokens(request_id, {*tok})) break;
    out.push_back(*tok);
    ++emitted;
  }
  return emitted;
}

}  // namespace vllm::v1
