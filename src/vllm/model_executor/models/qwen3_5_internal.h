// Internal Qwen3.5/3.6 model-selection contracts. Not installed.
// Packed-decode selection mirrors vLLM v0.25.0
// qwen_gdn_linear_attn.py:1286-1298 @ 702f4814, with the first local consumer
// deliberately scoped to the 27B CUDA packed-BA/BF16-activation path. The SSM
// cache dtype is independent (FP32 in the gate checkpoint).
#pragma once

#include <cstdint>
#include <vector>

#include "vt/dtype.h"

namespace vt {
struct Queue;
}  // namespace vt

namespace vllm {

struct GdnStateCache;
struct HfConfig;
struct GdnLayerWeights;

namespace v1 {
struct GDNAttentionMetadata;
}  // namespace v1

// Test-only entry point (SPEC-MTP I5a): run one GDN layer's paged forward over a
// batched step (spec or non-spec, per `meta`) and return the [T*H] output on host
// (f32). Drives the real per-step upload (BuildStepDevInputs) + layer assembly
// (GdnBlockPaged), so the synthetic spec-branch test exercises the exact
// production routing. Defined in qwen3_5.cpp.
std::vector<float> GdnBlockPagedForTest(vt::Queue queue, const GdnLayerWeights& w,
                                        const HfConfig& cfg,
                                        const std::vector<float>& h_host,
                                        const v1::GDNAttentionMetadata& meta,
                                        std::vector<float>& ssm_host,
                                        std::vector<float>& conv_host,
                                        int64_t num_slots, int64_t conv_len,
                                        int64_t T);

}  // namespace vllm

namespace vllm::detail {

struct GdnPackedDecodeEligibility {
  bool runtime_enabled = false;
  bool cuda = false;
  bool dense_model = false;
  bool has_packed_ba = false;
  bool merged_ba_enabled = false;
  bool dtype_compatible = false;
  bool has_state_indices = false;
  int64_t num_prefills = 0;
  int64_t num_prefill_tokens = 0;
  int64_t num_decodes = 0;
  int64_t num_decode_tokens = 0;
  int64_t num_spec_decodes = 0;
  int64_t num_spec_decode_tokens = 0;
  int64_t num_actual_tokens = 0;
};

bool ShouldUsePackedGdnDecode(const GdnPackedDecodeEligibility& eligibility);

// Process-level ENV resolution of the packed pure-decode arm for the real 27B
// dense CUDA gate (owner resident, pure non-spec one-token decode assumed).
// Mirrors, field for field, the exact process-cached env couplings the model
// wires into GdnPackedDecodeEligibility (qwen3_5.cpp GdnBlockPaged):
// PackedGdnDecodeRuntimeEnabled (VT_GDN_PACKED_DECODE), MergedGdnBaEnabled
// (master VT_GDN_MERGED_PROJ + leaf VT_GDN_MERGED_BA), and dtype_compatible —
// GdnInDType (VT_GDN_IN_BF16) == BF16, GdnOutDType dense default
// (VT_GDN_OUT_BF16) == BF16, MergedGdnBaOutputDType(packed)
// (VT_GDN_BA_OUT_BF16) == BF16. Splitting BA (master OR leaf) or reverting any
// coupled dtype therefore deselects packed and runs the decomposed recurrence:
// the gate's dispatch-count contract must expect ZERO packed launches on those
// arms, exactly like VT_GDN_PACKED_DECODE=0. Fields carry the raw getenv
// values (nullptr = unset) so the CPU tier can pin the truth table.
struct GdnPackedDecodeEnvConfig {
  const char* packed_decode = nullptr;  // VT_GDN_PACKED_DECODE
  const char* merged_proj = nullptr;    // VT_GDN_MERGED_PROJ (master)
  const char* merged_ba = nullptr;      // VT_GDN_MERGED_BA (leaf)
  const char* in_bf16 = nullptr;        // VT_GDN_IN_BF16
  const char* out_bf16 = nullptr;       // VT_GDN_OUT_BF16
  const char* ba_out_bf16 = nullptr;    // VT_GDN_BA_OUT_BF16
};

bool PackedGdnDecodeEnvSelected(const GdnPackedDecodeEnvConfig& env);

// W2 merged-qkvz dispatch. vLLM always issues one in_proj_qkvz GEMM
// (qwen_gdn_linear_attn.py:923-936 @ 702f4814); locally the single GEMM is
// selected only on CUDA with the packed 27B owner resident, the runtime
// toggles on (VT_GDN_MERGED_PROJ master, VT_GDN_MERGED_QKVZ leaf) and one
// uniform output dtype (mixed_qkv and z leave one GEMM, so GdnInDType must
// equal GdnOutDType — the 27B default is BF16/BF16). Every other combination
// issues the exact two split GEMMs sliced from the same resident owner.
struct GdnMergedQkvzEligibility {
  bool runtime_enabled = false;
  bool cuda = false;
  bool has_packed_qkvz = false;
  bool uniform_dtype = false;
};

bool ShouldUseMergedGdnQkvz(const GdnMergedQkvzEligibility& eligibility);

// Validate the exact prefix that will be uploaded. Negative rows are inert
// padding; every live slot must be unique and in range. This runs on host
// metadata before the device buffer is constructed, keeping CUDA capture free
// of a validation synchronization. Uniqueness is an O(n) seen-set pass (a live
// slot is drawn from a free-list of distinct slots by construction, so a single
// pass fails closed on any duplicate/out-of-range/negative slot); it is bounded
// by state_slots (== max_num_reqs). `force_full_uniqueness` (driven globally by
// VT_GDN_VALIDATE=1) additionally runs the exhaustive O(n^2) pairwise
// cross-verification — a redundant paranoid check, never needed for
// correctness.
void ValidateGdnStateIndices(const std::vector<int32_t>& indices,
                             int64_t required, int64_t state_slots,
                             bool force_full_uniqueness = false);

// Validate the complete eager/graph metadata contract before any state index
// is uploaded or consumed. Prefill-only vectors must be exact suffix/rebased
// views of the full non-spec vectors. Only CUDA-graph padding may contain the
// inert -1 state-slot sentinel.
void ValidateGdnAttentionMetadata(
    const v1::GDNAttentionMetadata& metadata, int64_t state_slots,
    bool allow_inert_padding);

// Row-copy state I/O cannot consume inert padded rows. Exact-size graphs are
// safe under either state-I/O mode; larger padded graph sizes require indexed
// state I/O, whose kernels define the -1 sentinel.
bool CanUseGdnDecodeGraphSize(int64_t real_batch, int64_t capture_batch,
                              bool indexed_state_io);

// Validate that every per-layer GDN cache exposes the same slot domain. Eager
// and graph paths both upload one shared state-index vector, so accepting a
// smaller later-layer cache would turn an index valid for layer zero into an
// out-of-bounds access. Returns the common slot count (zero for no caches).
int64_t ValidateGdnStateCacheLayout(
    const std::vector<GdnStateCache>& state_caches);

// Upstream MambaStateDtypeCalculator::_mamba_state_dtype. The temporal/SSM
// cache dtype is independent from the convolution cache dtype and accepts the
// exact torch dtype aliases used by raw HF configs.
vt::DType ResolveMambaSsmCacheDType(const HfConfig& config,
                                    vt::DType conv_dtype);

// Host preflight run at the entrance of every CUDA-graph Step, before padding,
// refresh, capture, or replay. It validates the real live request prefix
// against both state tensors in every GDN layer; the lower-level upload check
// remains defense in depth for eager/mixed paths.
void ValidateGdnDecodeGraphState(
    const v1::GDNAttentionMetadata& metadata,
    const std::vector<GdnStateCache>& state_caches, int64_t real_batch);

// ─── ENG-ASYNC-SCHED W4: device-resident input ids for the embed ─────────────
//
// `ModelForwardInput::device_token_ids` says "the input ids for this step are
// ALREADY on the device; the host vector is stale for decode rows". Only the
// embed at the very top of the forward cares, and it sits under five layers of
// entry points (eager / gathered / tap / multi-tap / decode-graph replay), each
// of which takes `const std::vector<int32_t>& token_ids` and passes it down.
//
// Rather than add a defaulted pointer parameter to every one of those and to the
// decode-graph class, the two Qwen3.5 registry forwards establish this SCOPED
// override for the duration of one forward and the embed consults it. The
// trade-off is deliberate and bounded: it is thread-local (a forward runs on one
// host thread), strictly RAII so it cannot leak past the call that set it, and
// set ONLY from the registry entry points that receive the ModelForwardInput —
// so its lifetime is exactly the forward's, not process state. It is null on
// every path except the discrete-CUDA async runner.
// The COUNT travels with the pointer so the embed can prove the buffer is the
// one meant for it. A forward can reach a second, unrelated embed over different
// ids (the multimodal generate helper embeds a prompt and then single tokens);
// an override that matched on "non-null" alone would silently feed that embed
// the wrong row count. Length disagreement means "not mine" and falls back to
// the host upload, which is always correct.
struct DeviceTokenIds {
  const int32_t* ids = nullptr;
  int64_t count = 0;
};

DeviceTokenIds& DeviceTokenIdsOverride();

struct DeviceTokenIdsScope {
  DeviceTokenIdsScope(const int32_t* ids, int64_t count)
      : prev(DeviceTokenIdsOverride()) {
    DeviceTokenIdsOverride() = DeviceTokenIds{ids, count};
  }
  ~DeviceTokenIdsScope() { DeviceTokenIdsOverride() = prev; }
  DeviceTokenIdsScope(const DeviceTokenIdsScope&) = delete;
  DeviceTokenIdsScope& operator=(const DeviceTokenIdsScope&) = delete;
  DeviceTokenIds prev;
};

}  // namespace vllm::detail
