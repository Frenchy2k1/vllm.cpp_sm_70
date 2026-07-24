// Ported from: vllm/v1/attention/backends/gdn_attn.py @ e24d1b24
#include "vllm/v1/attention/backends/gdn_attn.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "vllm/v1/attention/registry.h"

namespace vllm::v1 {

// Ported from utils.py::split_decodes_and_prefills @ e24d1b24 (lines 564-633),
// T0 subset: require_uniform=False, treat_short_extends_as_decodes=True.
std::tuple<int, int, int, int> SplitDecodesAndPrefills(
    const CommonAttentionMetadata& m, int decode_threshold) {
  const int max_query_len = m.max_query_len;
  const int num_reqs = m.num_reqs;
  const int num_tokens = m.num_actual_tokens;
  const std::vector<int32_t>& qsl = m.query_start_loc_cpu;

  // treat_short_extends_as_decodes (default): a batch whose longest query is
  // within the threshold is all decodes (utils.py:599-604).
  if (max_query_len <= decode_threshold) {
    return {num_reqs, 0, num_tokens, 0};
  }

  const std::vector<int32_t> query_lens = m.naive_query_lens();
  // First request is not a decode ⇒ no decodes (utils.py:607-609). This is the
  // decode-first ordering contract: a prefill at the front means the batch is
  // all prefills.
  if (!query_lens.empty() && query_lens[0] > decode_threshold) {
    return {0, num_reqs, 0, num_tokens};
  }

  // is_prefill[i] = query_lens[i] > decode_threshold; first_prefill = argmax
  // (first True) (utils.py:619-632).
  int first_prefill = -1;
  for (int i = 0; i < static_cast<int>(query_lens.size()); ++i) {
    if (query_lens[i] > decode_threshold) {
      first_prefill = i;
      break;
    }
  }
  if (first_prefill < 0) {
    // No prefill found ⇒ all decodes (utils.py:625-626).
    return {num_reqs, 0, num_tokens, 0};
  }

  const int num_decodes = first_prefill;
  const int num_prefills = num_reqs - num_decodes;
  const int num_decode_tokens = qsl[first_prefill];
  const int num_prefill_tokens = num_tokens - num_decode_tokens;
  return {num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens};
}

GDNAttentionMetadata GDNAttentionMetadataBuilder::build(
    int common_prefix_len, const CommonAttentionMetadata& m, bool fast_build) {
  // The non-spec entry point IS the spec build with both spec arguments null:
  // spec_sequence_masks then stays nullopt (gdn_attn.py:189-196) and the
  // identical non-spec branch runs. ONE implementation, so the default
  // production path cannot drift from the spec-tested one.
  return build(common_prefix_len, m, /*num_accepted_tokens=*/nullptr,
               /*num_decode_draft_tokens_cpu=*/nullptr, fast_build);
}

GDNAttentionMetadata GDNAttentionMetadataBuilder::build(
    int /*common_prefix_len*/, const CommonAttentionMetadata& m,
    const std::vector<int32_t>* num_accepted_tokens,
    const std::vector<int32_t>* num_decode_draft_tokens_cpu,
    bool /*fast_build*/) {
  GDNAttentionMetadata meta;
  meta.num_actual_tokens = m.num_actual_tokens;

  // context_lens = num_computed_tokens = seq_lens - query_lens
  // (backend.py::compute_num_computed_tokens; gdn_attn.py:180).
  const std::vector<int32_t> query_lens = m.naive_query_lens();
  std::vector<int32_t> context_lens(static_cast<size_t>(m.num_reqs), 0);
  for (int r = 0; r < m.num_reqs; ++r) {
    context_lens[static_cast<size_t>(r)] = m.seq_lens_cpu[static_cast<size_t>(r)] -
                                           query_lens[static_cast<size_t>(r)];
  }

  // Per-request GDN mamba-state block id = block_table column 0
  // (gdn_attn.py:219; mamba_get_block_table_tensor identity for the "all"/"none"
  // cache modes — align-mode gather is deferred, T0 uses column 0 directly).
  const int block_cols = m.block_table_num_cols;
  std::vector<int32_t> state_indices(static_cast<size_t>(m.num_reqs), 0);
  for (int r = 0; r < m.num_reqs; ++r) {
    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(block_cols);
    if (off < m.block_table_tensor.size()) {
      state_indices[static_cast<size_t>(r)] = m.block_table_tensor[off];
    }
  }

  // ── Spec/non-spec row partition (gdn_attn.py:189-206). A row is a SPEC row
  // iff it carries drafts, i.e. num_decode_draft_tokens[r] >= 0 (the -1
  // sentinel marks a non-spec row, mamba_hybrid.py:247-264). When the builder
  // has no speculative config, or the caller passed no draft counts, or every
  // draft count is 0/negative, spec_sequence_masks stays nullopt exactly as
  // upstream and the whole spec machinery below is skipped. ──
  std::optional<std::vector<uint8_t>> spec_sequence_masks;
  int num_spec_decodes = 0;
  if (use_spec_decode() && num_decode_draft_tokens_cpu != nullptr) {
    int32_t draft_sum = 0;
    for (const int32_t d : *num_decode_draft_tokens_cpu) {
      if (d >= 0) draft_sum += d;
    }
    if (draft_sum != 0) {
      std::vector<uint8_t> masks(static_cast<size_t>(m.num_reqs), 0);
      const size_t n = std::min(masks.size(), num_decode_draft_tokens_cpu->size());
      for (size_t r = 0; r < n; ++r) {
        if ((*num_decode_draft_tokens_cpu)[r] >= 0) {
          masks[r] = 1;
          ++num_spec_decodes;
        }
      }
      if (num_spec_decodes > 0) spec_sequence_masks = std::move(masks);
    }
  }

  int num_decodes = 0;
  int num_prefills = 0;
  int num_decode_tokens = 0;
  int num_prefill_tokens = 0;
  int num_spec_decode_tokens = 0;
  // non_spec_query_start_loc doubles as the chunked-conv / prefill cu_seqlens
  // source below; nullopt means "no non-spec rows at all" (pure spec batch).
  std::optional<std::vector<int32_t>> non_spec_query_start_loc;
  std::optional<std::vector<int32_t>> non_spec_state_indices;

  if (!spec_sequence_masks.has_value()) {
    // ── Non-spec path (gdn_attn.py:211-223) — the production default. ──
    const auto [nd, np, ndt, npt] = SplitDecodesAndPrefills(m, /*decode_threshold=*/1);
    num_decodes = nd;
    num_prefills = np;
    num_decode_tokens = ndt;
    num_prefill_tokens = npt;
    non_spec_state_indices = state_indices;
    non_spec_query_start_loc = m.query_start_loc;
  } else {
    // ── Spec path (gdn_attn.py:224-326). All counting is done on HOST arrays,
    // exactly like upstream's CPU tensors, so no device sync is introduced. ──
    const std::vector<uint8_t>& masks = *spec_sequence_masks;
    int32_t total_query_tokens = 0;
    for (int r = 0; r < m.num_reqs; ++r) {
      total_query_tokens += query_lens[static_cast<size_t>(r)];
    }
    int num_zero_len = 0;
    int32_t non_spec_tokens = 0;
    for (int r = 0; r < m.num_reqs; ++r) {
      if (masks[static_cast<size_t>(r)] != 0) continue;
      const int32_t ql = query_lens[static_cast<size_t>(r)];
      non_spec_tokens += ql;
      if (ql == 1) {
        ++num_decodes;
      } else if (ql == 0) {
        ++num_zero_len;  // zero-length padded rows are not prefills (:229-231)
      }
    }
    const int num_non_spec_rows = m.num_reqs - num_spec_decodes;
    num_prefills = num_non_spec_rows - num_decodes - num_zero_len;
    num_decode_tokens = num_decodes;
    num_prefill_tokens = static_cast<int>(non_spec_tokens) - num_decode_tokens;
    num_spec_decode_tokens =
        static_cast<int>(total_query_tokens) - num_prefill_tokens - num_decode_tokens;

    // DECODE→PREFILL RECLASSIFICATION (gdn_attn.py:243-251, upstream issue
    // #34845). num_decodes and num_spec_decodes are mutually exclusive: the
    // recurrent decode kernel would have to run a uniform 1-token batch while
    // the spec kernel runs multi-token rows over the SAME state cache, so
    // upstream instead routes the leftover 1-token rows through the chunked
    // prefill kernel, which handles a length-1 sequence with an initial state
    // exactly (identical result, just a different kernel).
    if (num_decodes > 0 && num_spec_decodes > 0) {
      num_prefills += num_decodes;
      num_prefill_tokens += num_decode_tokens;
      num_decodes = 0;
      num_decode_tokens = 0;
    }

    // Per-request spec state-slot rows: block_table[spec rows, :num_spec+1]
    // (gdn_attn.py:266-269 / :285-287). Row i column t is the slot timestep t
    // writes its post-token snapshot into.
    const int spec_cols = num_spec_ + 1;
    std::vector<int32_t> spec_state_indices;
    spec_state_indices.reserve(static_cast<size_t>(num_spec_decodes) *
                               static_cast<size_t>(spec_cols));
    for (int r = 0; r < m.num_reqs; ++r) {
      if (masks[static_cast<size_t>(r)] == 0) continue;
      for (int c = 0; c < spec_cols; ++c) {
        const size_t off =
            static_cast<size_t>(r) * static_cast<size_t>(block_cols) + static_cast<size_t>(c);
        spec_state_indices.push_back(
            c < block_cols && off < m.block_table_tensor.size()
                ? m.block_table_tensor[off]
                : kNullStateSlot);
      }
    }
    meta.spec_state_indices_tensor = std::move(spec_state_indices);
    meta.spec_state_indices_num_cols = spec_cols;

    if (num_prefills == 0 && num_decodes == 0) {
      // PURE spec batch (gdn_attn.py:252-273): every real token belongs to a
      // spec row, so the gather order is the identity over the leading
      // spec_token_size tokens and there is no non-spec sub-batch at all.
      const int spec_token_size =
          std::min(num_spec_decodes * (num_spec_ + 1), static_cast<int>(total_query_tokens));
      std::vector<int32_t> spec_token_indx(static_cast<size_t>(spec_token_size));
      for (int i = 0; i < spec_token_size; ++i) spec_token_indx[static_cast<size_t>(i)] = i;
      meta.spec_token_indx = std::move(spec_token_indx);
      meta.non_spec_token_indx = std::vector<int32_t>{};
      // Padded rows are always at the BACK, so the batch's own cumulative
      // offsets already are the spec sub-batch's (:270-272).
      meta.spec_query_start_loc = std::vector<int32_t>(
          m.query_start_loc.begin(),
          m.query_start_loc.begin() + static_cast<std::ptrdiff_t>(num_spec_decodes) + 1);
      // non_spec_* stay nullopt.
    } else {
      // MIXED batch (gdn_attn.py:274-323): stable-sort the per-token spec mask
      // so the non-spec tokens keep their relative order in the low half and
      // the spec tokens in the high half.
      std::vector<int32_t> non_spec_token_indx;
      std::vector<int32_t> spec_token_indx;
      non_spec_token_indx.reserve(static_cast<size_t>(num_prefill_tokens + num_decode_tokens));
      spec_token_indx.reserve(static_cast<size_t>(num_spec_decode_tokens));
      for (int r = 0; r < m.num_reqs; ++r) {
        const int32_t lo = m.query_start_loc[static_cast<size_t>(r)];
        const int32_t hi = m.query_start_loc[static_cast<size_t>(r) + 1];
        std::vector<int32_t>& dst =
            masks[static_cast<size_t>(r)] != 0 ? spec_token_indx : non_spec_token_indx;
        for (int32_t t = lo; t < hi; ++t) dst.push_back(t);
      }
      meta.spec_token_indx = std::move(spec_token_indx);
      meta.non_spec_token_indx = std::move(non_spec_token_indx);

      std::vector<int32_t> nssi;
      nssi.reserve(static_cast<size_t>(num_non_spec_rows));
      std::vector<int32_t> sqsl{0};
      sqsl.reserve(static_cast<size_t>(num_spec_decodes) + 1);
      std::vector<int32_t> nsqsl{0};
      nsqsl.reserve(static_cast<size_t>(num_non_spec_rows) + 1);
      for (int r = 0; r < m.num_reqs; ++r) {
        const int32_t ql = query_lens[static_cast<size_t>(r)];
        if (masks[static_cast<size_t>(r)] != 0) {
          sqsl.push_back(sqsl.back() + ql);
        } else {
          nssi.push_back(state_indices[static_cast<size_t>(r)]);
          nsqsl.push_back(nsqsl.back() + ql);
        }
      }
      meta.spec_query_start_loc = std::move(sqsl);
      non_spec_state_indices = std::move(nssi);
      non_spec_query_start_loc = std::move(nsqsl);
    }

    // num_accepted_tokens filtered to the spec rows (gdn_attn.py:324-325).
    // Upstream asserts it is present whenever a spec row exists.
    if (num_accepted_tokens == nullptr) {
      throw std::invalid_argument(
          "GDN build: num_accepted_tokens is required when the batch carries "
          "speculative draft tokens (gdn_attn.py:324)");
    }
    std::vector<int32_t> nat;
    nat.reserve(static_cast<size_t>(num_spec_decodes));
    for (int r = 0; r < m.num_reqs; ++r) {
      if (masks[static_cast<size_t>(r)] == 0) continue;
      const int32_t acc = static_cast<size_t>(r) < num_accepted_tokens->size()
                              ? (*num_accepted_tokens)[static_cast<size_t>(r)]
                              : 1;
      if (acc < 1 || acc > num_spec_ + 1) {
        throw std::invalid_argument(
            "GDN build: num_accepted_tokens must be in [1, num_spec + 1]");
      }
      nat.push_back(acc);
    }
    meta.num_accepted_tokens = std::move(nat);
  }

  meta.num_decodes = num_decodes;
  meta.num_prefills = num_prefills;
  meta.num_decode_tokens = num_decode_tokens;
  meta.num_prefill_tokens = num_prefill_tokens;
  meta.num_spec_decodes = num_spec_decodes;
  meta.num_spec_decode_tokens = num_spec_decode_tokens;
  meta.spec_sequence_masks = spec_sequence_masks;
  meta.non_spec_state_indices_tensor = non_spec_state_indices;
  meta.non_spec_query_start_loc = non_spec_query_start_loc;

  // ── Prefill-kernel inputs + has_initial_state (gdn_attn.py:333-405). Only
  // computed when there is prefill work; decode-only batches leave
  // has_initial_state == None (a decode always continues a sequence). ──
  if (num_prefills > 0) {
    // has_initial_state = context_lens > 0, restricted to the NON-SPEC rows
    // when a spec split happened (gdn_attn.py:389-392) so it lines up with
    // non_spec_query_start_loc / non_spec_state_indices_tensor.
    std::vector<uint8_t> has_initial_state;
    has_initial_state.reserve(static_cast<size_t>(m.num_reqs));
    for (int r = 0; r < m.num_reqs; ++r) {
      if (spec_sequence_masks.has_value() &&
          (*spec_sequence_masks)[static_cast<size_t>(r)] != 0) {
        continue;
      }
      has_initial_state.push_back(context_lens[static_cast<size_t>(r)] > 0 ? 1 : 0);
    }
    meta.has_initial_state = has_initial_state;

    if (!spec_sequence_masks.has_value() && num_decodes > 0) {
      // MIXED non-spec batch: peel the leading decodes off and rebase the
      // prefill-only cu_seqlens / state indices / mask (gdn_attn.py:340-350,
      // :400-401).
      const std::vector<int32_t>& nsqsl = *non_spec_query_start_loc;
      std::vector<int32_t> pqsl;
      pqsl.reserve(nsqsl.size() - static_cast<size_t>(num_decodes));
      for (size_t i = static_cast<size_t>(num_decodes); i < nsqsl.size(); ++i) {
        pqsl.push_back(nsqsl[i] - num_decode_tokens);
      }
      meta.prefill_query_start_loc = std::move(pqsl);

      const std::vector<int32_t>& nssi = *non_spec_state_indices;
      meta.prefill_state_indices = std::vector<int32_t>(
          nssi.begin() + static_cast<std::ptrdiff_t>(num_decodes), nssi.end());

      meta.prefill_has_initial_state = std::vector<uint8_t>(
          has_initial_state.begin() + static_cast<std::ptrdiff_t>(num_decodes),
          has_initial_state.end());
    } else {
      // Prefill-only, or the spec case (where the reclassification already
      // forced num_decodes == 0): the whole non-spec sub-batch IS the prefill
      // sub-batch (gdn_attn.py:352-354, :403).
      meta.prefill_query_start_loc = non_spec_query_start_loc;
      meta.prefill_state_indices = non_spec_state_indices;
      meta.prefill_has_initial_state = has_initial_state;
    }
  }
  // else: has_initial_state / prefill_* stay nullopt (gdn_attn.py:405).

  // Upstream invariant (gdn_attn.py:406-409): the recurrent decode kernel and
  // the spec kernel are mutually exclusive after reclassification.
  if (num_decodes > 0 && num_spec_decodes > 0) {
    throw std::logic_error(
        "GDN build: non-spec decodes and spec decodes must be mutually "
        "exclusive after reclassification");
  }

  // ── FULL-cudagraph padding of the REQUEST-indexed spec metadata
  // (gdn_attn.py:411-462). num_actual_tokens is token-padded for graph replay,
  // but everything below is indexed by REQUEST, so it is padded to m.num_reqs —
  // never to the token count. Padded rows get the NULL state slot (skipped by
  // every kernel) and an accepted count of 1. Only the spec block is ported;
  // the non-spec decode padding (:464-478) stays with our runner, which pads
  // the graph batch itself. ──
  if (use_full_cuda_graph_ && num_prefills == 0 && num_decodes == 0 &&
      num_spec_decodes > 0) {
    const int batch_size = m.num_reqs;
    const int cols = meta.spec_state_indices_num_cols;
    std::vector<int32_t>& ssi = *meta.spec_state_indices_tensor;
    ssi.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(cols), kNullStateSlot);
    std::vector<uint8_t>& ssm = *meta.spec_sequence_masks;
    ssm.resize(static_cast<size_t>(batch_size), 0);
    std::vector<int32_t>& sqsl = *meta.spec_query_start_loc;
    const int32_t spec_num_query_tokens = sqsl.back();
    sqsl.resize(static_cast<size_t>(batch_size) + 1, spec_num_query_tokens);
    std::vector<int32_t>& nat = *meta.num_accepted_tokens;
    nat.resize(static_cast<size_t>(batch_size), 1);
  }

  return meta;
}

std::vector<int64_t> GDNAttentionBackend::get_kv_cache_shape(
    int64_t /*num_blocks*/, int64_t /*block_size*/, int64_t /*num_kv_heads*/,
    int64_t /*head_size*/, const std::string& /*cache_dtype_str*/) const {
  // GDN is a state-space backend: its cache is the mamba state (shape from
  // MambaSpec, not a paged KV cache). Upstream GDNAttentionBackend intentionally
  // does not implement get_kv_cache_shape.
  throw std::logic_error(
      "GDN_ATTN is an SSM backend; the mamba-state shape comes from MambaSpec, "
      "not get_kv_cache_shape.");
}

namespace {
// GDN_ATTN self-registers for discoverability in the attention-backend registry
// (mirrors upstream @register_backend(AttentionBackendEnum.GDN_ATTN),
// registry.py:180). It is an SSM (mamba) backend selected PER LAYER by the model
// architecture (the linear-attention layers), NOT via the paged-attention
// priority walk — so it is intentionally absent from the CUDA/CPU
// get_attn_backend_priority lists. Registration makes it constructible through
// MakeAttentionBackend(kCUDA/kCPU, "GDN_ATTN") without an inline code edit.
AttentionBackendFactory MakeGDNAttentionBackend = []() -> std::unique_ptr<AttentionBackend> {
  return std::make_unique<GDNAttentionBackend>();
};
const AttentionBackendRegistrar kGdnAttnCuda{vt::DeviceType::kCUDA,
                                             GDNAttentionBackend::kName,
                                             MakeGDNAttentionBackend};
const AttentionBackendRegistrar kGdnAttnCpu{vt::DeviceType::kCPU,
                                            GDNAttentionBackend::kName,
                                            MakeGDNAttentionBackend};
}  // namespace

}  // namespace vllm::v1
