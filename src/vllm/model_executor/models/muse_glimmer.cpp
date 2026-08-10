// Muse Glimmer forward — REFUSE-by-name skeleton (W0). The text tower (W1) and
// the perception encoder (W3) are not wired here; both entrypoints VT_CHECK(false)
// so a forward LOUDLY reports the pending brick rather than returning a silent
// wrong answer. Mirrors `kimi_k3.cpp` / `deepseek_v4.cpp`.
//
// Ported from vllm#51655 head `075d645af` (models/muse_glimmer.py:1304-1345,
// :1604-1613), an OPEN upstream PR — NOT the parity pin. See porting-inventory §9
// deviation 16.
#include "vllm/model_executor/models/muse_glimmer.h"

#include <vector>

#include "vt/dtype.h"

namespace vllm {

namespace {
constexpr const char* kPending =
    "MuseGlimmer forward is not yet implemented — this is a REFUSE-by-name W0 "
    "skeleton (config parse + weight name map + registry only). The text tower "
    "(sandwich norms with a baked +1 offset and split pre/post eps, the iRoPE "
    "NoPE/sliding split, weightless pre-RoPE QK-norm, the query pre-scale, and the "
    "attention output gate) is W1; the perception encoder is W3. MuseGlimmer is "
    "also BEYOND the pinned oracle (555967922) — the pinned vLLM cannot load "
    "muse_glimmer at all, so there is no on-box golden and no speed denominator. "
    "See .agents/specs/muse-glimmer.md §0 and §3.";
}  // namespace

std::vector<float> MuseGlimmerModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false, kPending);
  return {};
}

ForwardLogits MuseGlimmerModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false, kPending);
  return {};
}

}  // namespace vllm
