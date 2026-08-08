// Tests for the M1.6 Task 4 GDN attention metadata — the prefill/decode
// segmentation GDNAttentionMetadataBuilder emits so the M0.7 GDN ops
// (vt::GdnPrefill / vt::GdnDecode) can be driven from a batched step.
//
// Ported from vllm/v1/attention/backends/gdn_attn.py @ e24d1b24 (the non-spec
// AND the spec build paths) plus
// tests/v1/attention/test_gdn_metadata_builder.py — the full
// GDN_BUILD_TEST_CASES table (:43-119, the upstream #34845 reclassification
// regression), test_has_initial_state_after_reclassification (:187) and
// test_full_cudagraph_spec_metadata_uses_request_count (:201).
//
// The builder ASSUMES a decode-first-reordered batch (the runner reorders on
// reorder_batch_threshold=1); every batch below is provided decode-first, as
// upstream split_decodes_and_prefills requires.
#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::CommonAttentionMetadata;
using vllm::v1::ComputeCausalConv1dMetadata;
using vllm::v1::GDNAttentionBackend;
using vllm::v1::GDNAttentionMetadata;
using vllm::v1::GDNAttentionMetadataBuilder;
using vllm::v1::MakeCommonAttentionMetadata;
using vllm::v1::SplitDecodesAndPrefills;
using vllm::v1::StepInputs;

namespace {

// Build a CommonAttentionMetadata from explicit (seq_len, query_len) pairs, in
// the given order. block_table has one column: block id = 100 + req index, so
// the state index of req r is 100 + 10*r (easy to assert).
CommonAttentionMetadata make_cam(const std::vector<int32_t>& seq_lens,
                                 const std::vector<int32_t>& query_lens,
                                 int block_table_num_cols = 1) {
  StepInputs step;
  step.seq_lens = seq_lens;
  step.query_start_loc = {0};
  int32_t total = 0;
  for (const int32_t q : query_lens) {
    total += q;
    step.query_start_loc.push_back(total);
  }
  step.num_scheduled_tokens = query_lens;
  step.slot_mapping = {std::vector<int64_t>(static_cast<size_t>(total), 0)};

  // Column c of request r holds block id 100 + 10*r + c, so both the non-spec
  // state index (column 0 ⇒ 100 + 10r) and each spec slot column are uniquely
  // identifiable in the assertions below.
  std::vector<int32_t> block_table_flat;
  for (size_t r = 0; r < seq_lens.size(); ++r) {
    for (int c = 0; c < block_table_num_cols; ++c) {
      block_table_flat.push_back(100 + 10 * static_cast<int32_t>(r) + c);
    }
  }
  return MakeCommonAttentionMetadata(step, block_table_flat, block_table_num_cols);
}

// Upstream `_build(builder, batch, num_decode_draft_tokens)`
// (test_gdn_metadata_builder.py:154-169): when draft counts are supplied the
// harness also passes a num_accepted_tokens vector of all ones.
GDNAttentionMetadata build_spec(GDNAttentionMetadataBuilder& builder,
                                const CommonAttentionMetadata& m,
                                const std::vector<int32_t>& num_decode_draft_tokens,
                                std::vector<int32_t> num_accepted = {}) {
  if (num_accepted.empty()) {
    num_accepted.assign(num_decode_draft_tokens.size(), 1);
  }
  return builder.build(0, m, &num_accepted, &num_decode_draft_tokens);
}

}  // namespace

// Core case from the M1.6 Task 4 brief: a mixed batch with 1 decode + 1 prefill,
// provided decode-first (the ordering the builder assumes). The decode req has a
// 6-token context with 5 already computed (query_len 1 ⇒ has_initial_state
// true); the prefill req is a fresh 4-token sequence (num_computed 0 ⇒
// has_initial_state false).
TEST_CASE("GDN build: mixed decode + prefill (decode-first)") {
  // req0: decode  seq_len 6, query_len 1  (computed 5)
  // req1: prefill seq_len 4, query_len 4  (computed 0)
  const CommonAttentionMetadata m = make_cam({6, 4}, {1, 4});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(/*common_prefix_len=*/0, m);

  CHECK(meta.num_decodes == 1);
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_decode_tokens == 1);
  CHECK(meta.num_prefill_tokens == 4);
  CHECK(meta.num_spec_decodes == 0);
  CHECK(meta.num_spec_decode_tokens == 0);
  CHECK(meta.num_actual_tokens == 5);

  // has_initial_state (full batch, decode-first): [decode computed>0=1,
  // prefill computed0=0].
  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{1, 0});

  // Non-spec state indices = block_table col 0 = {100, 110}.
  REQUIRE(meta.non_spec_state_indices_tensor.has_value());
  CHECK(*meta.non_spec_state_indices_tensor == std::vector<int32_t>{100, 110});
  REQUIRE(meta.non_spec_query_start_loc.has_value());
  CHECK(*meta.non_spec_query_start_loc == std::vector<int32_t>{0, 1, 5});

  // Prefill sub-batch = leading decode peeled off + rebased by num_decode_tokens.
  // non_spec_query_start_loc = {0,1,5}; [num_decodes=1:] = {1,5}; -1 = {0,4}.
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 4});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{110});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{0});
}

TEST_CASE("GDN metadata: causal-conv programs enumerate each sequence chunk once") {
  // Unequal lengths distinguish the exact flattened descriptor from a rectangular
  // sequence x chunk grid. BLOCK_M=8 yields 1+2+3 programs, with no padded work.
  const auto chunks = ComputeCausalConv1dMetadata(
      std::vector<int32_t>{0, 1, 10, 27});
  CHECK(chunks.batch_ptr == std::vector<int32_t>{0, 1, 1, 2, 2, 2});
  CHECK(chunks.token_chunk_offset_ptr ==
        std::vector<int32_t>{0, 0, 1, 0, 1, 2});
}

// Decode-only batch: all query_len==1. num_prefills==0 ⇒ has_initial_state and
// all prefill_* fields are None (gdn_attn.py:405) — the decode kernel reads the
// state via state_indices and needs no has_initial_state mask.
TEST_CASE("GDN build: decodes only") {
  // Mirrors upstream test case "pure_regular_decode".
  const CommonAttentionMetadata m = make_cam({40, 30, 20}, {1, 1, 1});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 3);
  CHECK(meta.num_prefills == 0);
  CHECK(meta.num_decode_tokens == 3);
  CHECK(meta.num_prefill_tokens == 0);
  CHECK(meta.num_spec_decodes == 0);

  CHECK_FALSE(meta.has_initial_state.has_value());
  CHECK_FALSE(meta.prefill_query_start_loc.has_value());
  CHECK_FALSE(meta.prefill_state_indices.has_value());
  CHECK_FALSE(meta.prefill_has_initial_state.has_value());

  // Non-spec state indices are still emitted for the decode kernel.
  REQUIRE(meta.non_spec_state_indices_tensor.has_value());
  CHECK(*meta.non_spec_state_indices_tensor == std::vector<int32_t>{100, 110, 120});
}

// Prefills-only batch of two FRESH sequences (num_computed 0 for both). No decode
// to peel: prefill_query_start_loc == non_spec_query_start_loc, and
// has_initial_state is all-false.
TEST_CASE("GDN build: prefills only (fresh, no initial state)") {
  const CommonAttentionMetadata m = make_cam({4, 7}, {4, 7});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_decode_tokens == 0);
  CHECK(meta.num_prefill_tokens == 11);

  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{0, 0});
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 4, 11});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{100, 110});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{0, 0});
}

// Prefills-only batch of two CONTINUING sequences (chunked prefill: num_computed
// > 0 for both). has_initial_state is all-true — the GDN op must start from the
// carried state.
TEST_CASE("GDN build: prefills only (continuing, has initial state)") {
  // req0: seq_len 10, query_len 4 (computed 6); req1: seq_len 20, query_len 7
  // (computed 13).
  const CommonAttentionMetadata m = make_cam({10, 20}, {4, 7});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_decodes == 0);
  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{1, 1});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{1, 1});
}

// Multiple leading decodes + one prefill: all leading query_len==1 requests are
// decodes; the prefill tail is peeled and rebased.
TEST_CASE("GDN build: multiple decodes + one prefill") {
  // req0,1,2: decodes (query_len 1); req3: fresh prefill seq_len 5, query_len 5
  // (computed 0).
  const CommonAttentionMetadata m = make_cam({30, 40, 50, 5}, {1, 1, 1, 5});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 3);
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_decode_tokens == 3);
  CHECK(meta.num_prefill_tokens == 5);

  // has_initial_state: decodes computed>0 (29,39,49) true; prefill fresh (0)
  // false.
  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{1, 1, 1, 0});

  // Prefill sub-batch: non_spec qsl {0,1,2,3,8}; [3:] = {3,8}; - 3 = {0,5}.
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 5});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{130});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{0});
}

// Decode-first ORDERING CONTRACT: split_decodes_and_prefills is order-sensitive.
// A batch whose FIRST request is a prefill (query_len > threshold) is classified
// as all-prefills, even if a query_len==1 request follows (utils.py:607-609).
// This documents why the runner must reorder decodes to the front.
TEST_CASE("GDN build: prefill-first is all-prefills (ordering contract)") {
  // req0: prefill (query_len 4), req1: query_len 1 AFTER it.
  const CommonAttentionMetadata m = make_cam({4, 6}, {4, 1});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_decode_tokens == 0);
  CHECK(meta.num_prefill_tokens == 5);
}

TEST_CASE("SplitDecodesAndPrefills: all-decode fast path") {
  const CommonAttentionMetadata m = make_cam({5, 6}, {1, 1});
  const auto [nd, np, ndt, npt] = SplitDecodesAndPrefills(m, 1);
  CHECK(nd == 2);
  CHECK(np == 0);
  CHECK(ndt == 2);
  CHECK(npt == 0);
}

TEST_CASE("GDNAttentionBackend: name / ssm / no kv-cache shape") {
  const GDNAttentionBackend backend;
  CHECK(backend.get_name() == "GDN_ATTN");
  CHECK(GDNAttentionBackend::is_ssm() == true);
  CHECK_THROWS_AS(backend.get_kv_cache_shape(1, 16, 2, 128), std::logic_error);
}

// ===========================================================================
// SPEC-DECODE cases (SPEC-MTP I4). Ported from
// tests/v1/attention/test_gdn_metadata_builder.py @ e24d1b24 — the whole
// GDN_BUILD_TEST_CASES table (:43-119), which exists upstream because of the
// #34845 crash: a batch mixing a plain 1-token decode with a multi-token spec
// decode used to violate the "decodes and spec decodes are mutually exclusive"
// invariant. The fix reclassifies the plain decodes as prefills.
// ===========================================================================

// GDN_BUILD_TEST_CASES["mixed_decode_and_spec_decode"] — the original #34845
// crash: one non-spec query_len=1 row plus one spec row.
TEST_CASE("GDN build spec: mixed_decode_and_spec_decode (the #34845 case)") {
  const CommonAttentionMetadata m = make_cam({65, 20}, {1, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {-1, 2});

  CHECK(meta.num_decodes == 0);  // reclassified
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_prefill_tokens == 1);
  CHECK(meta.num_spec_decodes == 1);
  CHECK(meta.num_decode_tokens == 0);
  CHECK(meta.num_spec_decode_tokens == 3);

  REQUIRE(meta.spec_sequence_masks.has_value());
  CHECK(*meta.spec_sequence_masks == std::vector<uint8_t>{0, 1});
  // Spec row = request 1 ⇒ block-table row {110, 111, 112}, k+1 = 3 columns.
  CHECK(meta.spec_state_indices_num_cols == 3);
  REQUIRE(meta.spec_state_indices_tensor.has_value());
  CHECK(*meta.spec_state_indices_tensor == std::vector<int32_t>{110, 111, 112});
  // spec_query_start_loc is a fresh cumsum over the spec rows only.
  REQUIRE(meta.spec_query_start_loc.has_value());
  CHECK(*meta.spec_query_start_loc == std::vector<int32_t>{0, 3});
  // Token gather orders: request 0 owns token 0, request 1 owns tokens 1..3.
  REQUIRE(meta.non_spec_token_indx.has_value());
  CHECK(*meta.non_spec_token_indx == std::vector<int32_t>{0});
  REQUIRE(meta.spec_token_indx.has_value());
  CHECK(*meta.spec_token_indx == std::vector<int32_t>{1, 2, 3});
  // Non-spec sub-batch is re-cumsummed over the non-spec rows only.
  REQUIRE(meta.non_spec_query_start_loc.has_value());
  CHECK(*meta.non_spec_query_start_loc == std::vector<int32_t>{0, 1});
  REQUIRE(meta.non_spec_state_indices_tensor.has_value());
  CHECK(*meta.non_spec_state_indices_tensor == std::vector<int32_t>{100});
  // num_accepted_tokens is filtered to the spec rows.
  REQUIRE(meta.num_accepted_tokens.has_value());
  CHECK(*meta.num_accepted_tokens == std::vector<int32_t>{1});
}

// GDN_BUILD_TEST_CASES["pure_spec_decode"] — every row is a spec row, so no
// reclassification is needed and there is no non-spec sub-batch at all.
TEST_CASE("GDN build spec: pure_spec_decode") {
  const CommonAttentionMetadata m = make_cam({50, 30}, {3, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {2, 2});

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 0);
  CHECK(meta.num_prefill_tokens == 0);
  CHECK(meta.num_spec_decodes == 2);
  CHECK(meta.num_spec_decode_tokens == 6);

  REQUIRE(meta.spec_state_indices_tensor.has_value());
  CHECK(*meta.spec_state_indices_tensor ==
        std::vector<int32_t>{100, 101, 102, 110, 111, 112});
  REQUIRE(meta.spec_query_start_loc.has_value());
  CHECK(*meta.spec_query_start_loc == std::vector<int32_t>{0, 3, 6});
  // Identity gather over the leading spec_token_size tokens (gdn_attn.py:258-262).
  REQUIRE(meta.spec_token_indx.has_value());
  CHECK(*meta.spec_token_indx == std::vector<int32_t>{0, 1, 2, 3, 4, 5});
  REQUIRE(meta.non_spec_token_indx.has_value());
  CHECK(meta.non_spec_token_indx->empty());
  // No non-spec sub-batch (gdn_attn.py:270-272).
  CHECK_FALSE(meta.non_spec_query_start_loc.has_value());
  CHECK_FALSE(meta.non_spec_state_indices_tensor.has_value());
  CHECK_FALSE(meta.has_initial_state.has_value());
}

// GDN_BUILD_TEST_CASES["spec_decode_with_real_prefill"] — a genuine multi-token
// prefill alongside a spec row; there is no plain decode to reclassify.
TEST_CASE("GDN build spec: spec_decode_with_real_prefill") {
  const CommonAttentionMetadata m = make_cam({100, 20}, {50, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {-1, 2});

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_prefill_tokens == 50);
  CHECK(meta.num_spec_decodes == 1);
  CHECK(meta.num_spec_decode_tokens == 3);
  REQUIRE(meta.non_spec_token_indx.has_value());
  CHECK(meta.non_spec_token_indx->size() == 50);
  CHECK(meta.non_spec_token_indx->front() == 0);
  CHECK(meta.non_spec_token_indx->back() == 49);
  REQUIRE(meta.spec_token_indx.has_value());
  CHECK(*meta.spec_token_indx == std::vector<int32_t>{50, 51, 52});
}

// GDN_BUILD_TEST_CASES["prefill_decode_and_spec_decode"] — all three row kinds
// in one batch; only the plain decode gets reclassified.
TEST_CASE("GDN build spec: prefill_decode_and_spec_decode") {
  const CommonAttentionMetadata m = make_cam({100, 65, 20}, {50, 1, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {-1, -1, 2});

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_prefill_tokens == 51);
  CHECK(meta.num_spec_decodes == 1);
  CHECK(meta.num_spec_decode_tokens == 3);
  REQUIRE(meta.non_spec_query_start_loc.has_value());
  CHECK(*meta.non_spec_query_start_loc == std::vector<int32_t>{0, 50, 51});
  REQUIRE(meta.non_spec_state_indices_tensor.has_value());
  CHECK(*meta.non_spec_state_indices_tensor == std::vector<int32_t>{100, 110});
  // With spec rows present the prefill sub-batch IS the whole non-spec batch —
  // no decode peel-off (gdn_attn.py:340 keys the peel on spec_sequence_masks
  // being None).
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 50, 51});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{100, 110});
}

// GDN_BUILD_TEST_CASES["multiple_decodes_reclassified"].
TEST_CASE("GDN build spec: multiple_decodes_reclassified") {
  const CommonAttentionMetadata m = make_cam({40, 50, 60, 20}, {1, 1, 1, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {-1, -1, -1, 2});

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 3);
  CHECK(meta.num_prefill_tokens == 3);
  CHECK(meta.num_spec_decodes == 1);
  CHECK(meta.num_spec_decode_tokens == 3);
  REQUIRE(meta.non_spec_query_start_loc.has_value());
  CHECK(*meta.non_spec_query_start_loc == std::vector<int32_t>{0, 1, 2, 3});
  // All three reclassified rows continue an existing sequence.
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{1, 1, 1});
}

// GDN_BUILD_TEST_CASES["zero_length_padding_with_spec"] — a query_len==0 padded
// row counts as neither a decode nor a prefill (gdn_attn.py:229-231).
TEST_CASE("GDN build spec: zero_length_padding_with_spec") {
  const CommonAttentionMetadata m = make_cam({16, 65, 20}, {0, 1, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {-1, -1, 2});

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_prefill_tokens == 1);
  CHECK(meta.num_spec_decodes == 1);
}

// Upstream test_has_initial_state_after_reclassification (:187-198): after the
// reclassification num_prefills > 0, so the prefill path must compute
// has_initial_state, and the reclassified row (context 65-1 = 64 > 0) is True.
TEST_CASE("GDN build spec: has_initial_state after reclassification") {
  const CommonAttentionMetadata m = make_cam({65, 20}, {1, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta = build_spec(builder, m, {-1, 2});

  CHECK(meta.num_prefills > 0);
  REQUIRE(meta.has_initial_state.has_value());
  // Restricted to the NON-SPEC rows, so entry 0 IS the reclassified request.
  CHECK(meta.has_initial_state->size() == 1);
  CHECK((*meta.has_initial_state)[0] == 1);
}

// Upstream test_full_cudagraph_spec_metadata_uses_request_count (:201-231):
// FULL-cudagraph token padding must not pad the REQUEST-indexed metadata — the
// spec tensors are padded to the request count (and only then).
TEST_CASE("GDN build spec: full-cudagraph padding uses the request count") {
  const int num_spec = 3;
  const CommonAttentionMetadata m = make_cam({80, 96}, {4, 4}, /*cols=*/num_spec + 1);
  GDNAttentionMetadataBuilder builder(num_spec, /*use_full_cuda_graph=*/true);
  const GDNAttentionMetadata meta = build_spec(builder, m, {3, 3});

  CHECK(meta.num_spec_decodes == 2);       // == batch size
  CHECK(meta.num_spec_decode_tokens == 8);  // == total tokens
  REQUIRE(meta.spec_state_indices_tensor.has_value());
  CHECK(meta.spec_state_indices_num_cols == num_spec + 1);
  CHECK(meta.spec_state_indices_tensor->size() ==
        static_cast<size_t>(m.num_reqs) * static_cast<size_t>(num_spec + 1));
  REQUIRE(meta.spec_sequence_masks.has_value());
  CHECK(meta.spec_sequence_masks->size() == static_cast<size_t>(m.num_reqs));
  REQUIRE(meta.spec_query_start_loc.has_value());
  CHECK(meta.spec_query_start_loc->size() == static_cast<size_t>(m.num_reqs) + 1);
  REQUIRE(meta.num_accepted_tokens.has_value());
  CHECK(meta.num_accepted_tokens->size() == static_cast<size_t>(m.num_reqs));
}

// The padding fill values matter: a padded row must be SKIPPED by the kernels
// (NULL slot) and must not select an out-of-range accepted column.
TEST_CASE("GDN build spec: cudagraph padding fills NULL slots and accepted=1") {
  const int num_spec = 1;
  // Three request rows, only the first two carrying drafts.
  const CommonAttentionMetadata m = make_cam({80, 96, 16}, {2, 2, 0}, /*cols=*/num_spec + 1);
  GDNAttentionMetadataBuilder builder(num_spec, /*use_full_cuda_graph=*/true);
  const GDNAttentionMetadata meta = build_spec(builder, m, {1, 1, -1});

  CHECK(meta.num_spec_decodes == 2);
  REQUIRE(meta.spec_state_indices_tensor.has_value());
  const std::vector<int32_t>& ssi = *meta.spec_state_indices_tensor;
  REQUIRE(ssi.size() == 6);
  CHECK(ssi[0] == 100);
  CHECK(ssi[1] == 101);
  CHECK(ssi[2] == 110);
  CHECK(ssi[3] == 111);
  CHECK(ssi[4] == vllm::v1::kNullStateSlot);
  CHECK(ssi[5] == vllm::v1::kNullStateSlot);
  REQUIRE(meta.spec_sequence_masks.has_value());
  CHECK(*meta.spec_sequence_masks == std::vector<uint8_t>{1, 1, 0});
  REQUIRE(meta.spec_query_start_loc.has_value());
  CHECK(*meta.spec_query_start_loc == std::vector<int32_t>{0, 2, 4, 4});
  REQUIRE(meta.num_accepted_tokens.has_value());
  CHECK(*meta.num_accepted_tokens == std::vector<int32_t>{1, 1, 1});
}

// The rollback selector reaches the metadata unchanged: num_accepted_tokens is
// filtered by the spec mask, preserving order.
TEST_CASE("GDN build spec: num_accepted_tokens is filtered to spec rows in order") {
  const CommonAttentionMetadata m = make_cam({40, 50, 60}, {1, 3, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const GDNAttentionMetadata meta =
      build_spec(builder, m, {-1, 2, 2}, /*num_accepted=*/{1, 3, 2});

  CHECK(meta.num_spec_decodes == 2);
  REQUIRE(meta.num_accepted_tokens.has_value());
  CHECK(*meta.num_accepted_tokens == std::vector<int32_t>{3, 2});
}

TEST_CASE("GDN build spec: rejects a missing or out-of-range accepted count") {
  const CommonAttentionMetadata m = make_cam({40, 50}, {1, 3}, /*cols=*/3);
  GDNAttentionMetadataBuilder builder(/*num_spec=*/2);
  const std::vector<int32_t> drafts = {-1, 2};
  CHECK_THROWS_AS(builder.build(0, m, nullptr, &drafts), std::invalid_argument);
  const std::vector<int32_t> bad_zero = {1, 0};
  CHECK_THROWS_AS(builder.build(0, m, &bad_zero, &drafts), std::invalid_argument);
  const std::vector<int32_t> bad_big = {1, 4};  // > num_spec + 1
  CHECK_THROWS_AS(builder.build(0, m, &bad_big, &drafts), std::invalid_argument);
}

// ── DEFAULT-OFF proof (metadata half). Three ways of asking for a non-spec
// build must produce IDENTICAL metadata: the 3-argument override, the spec
// overload with null arguments, and a spec-configured builder handed an
// all-non-spec draft vector. The last one is the one that matters in production:
// with a speculator configured but no drafts in flight, the step must still take
// the exact pre-I4 path. ──
namespace {
void CheckMetaEqual(const GDNAttentionMetadata& a, const GDNAttentionMetadata& b) {
  CHECK(a.num_prefills == b.num_prefills);
  CHECK(a.num_prefill_tokens == b.num_prefill_tokens);
  CHECK(a.num_decodes == b.num_decodes);
  CHECK(a.num_decode_tokens == b.num_decode_tokens);
  CHECK(a.num_spec_decodes == b.num_spec_decodes);
  CHECK(a.num_spec_decode_tokens == b.num_spec_decode_tokens);
  CHECK(a.num_actual_tokens == b.num_actual_tokens);
  CHECK(a.has_initial_state == b.has_initial_state);
  CHECK(a.non_spec_state_indices_tensor == b.non_spec_state_indices_tensor);
  CHECK(a.non_spec_query_start_loc == b.non_spec_query_start_loc);
  CHECK(a.prefill_query_start_loc == b.prefill_query_start_loc);
  CHECK(a.prefill_state_indices == b.prefill_state_indices);
  CHECK(a.prefill_has_initial_state == b.prefill_has_initial_state);
  CHECK(a.batch_ptr == b.batch_ptr);
  CHECK(a.token_chunk_offset_ptr == b.token_chunk_offset_ptr);
  CHECK(a.spec_query_start_loc == b.spec_query_start_loc);
  CHECK(a.spec_state_indices_tensor == b.spec_state_indices_tensor);
  CHECK(a.spec_state_indices_num_cols == b.spec_state_indices_num_cols);
  CHECK(a.spec_sequence_masks == b.spec_sequence_masks);
  CHECK(a.spec_token_indx == b.spec_token_indx);
  CHECK(a.non_spec_token_indx == b.non_spec_token_indx);
  CHECK(a.num_accepted_tokens == b.num_accepted_tokens);
}
}  // namespace

TEST_CASE("GDN build: spec-off metadata is identical to the non-spec build") {
  const std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> batches = {
      {{6, 4}, {1, 4}},           // mixed decode + prefill
      {{40, 30, 20}, {1, 1, 1}},  // decodes only
      {{4, 7}, {4, 7}},           // fresh prefills
      {{10, 20}, {4, 7}},         // continuing prefills
      {{30, 40, 50, 5}, {1, 1, 1, 5}},
  };
  for (const auto& [seq_lens, query_lens] : batches) {
    CAPTURE(seq_lens.size());
    const CommonAttentionMetadata m = make_cam(seq_lens, query_lens, /*cols=*/2);
    GDNAttentionMetadataBuilder plain;
    const GDNAttentionMetadata base = plain.build(0, m);

    // Same builder, explicit null spec arguments.
    CheckMetaEqual(plain.build(0, m, nullptr, nullptr), base);

    // Speculator CONFIGURED (num_spec = 1) but no drafts in flight — the
    // production shape of every non-speculative step once MTP is enabled.
    GDNAttentionMetadataBuilder spec(/*num_spec=*/1);
    const std::vector<int32_t> no_drafts(seq_lens.size(), -1);
    const std::vector<int32_t> accepted(seq_lens.size(), 1);
    CheckMetaEqual(spec.build(0, m, &accepted, &no_drafts), base);
    // ... and with drafts declared but all zero-length (upstream's
    // `num_decode_draft_tokens[... >= 0].sum() == 0` early-out, :191-196).
    const std::vector<int32_t> zero_drafts(seq_lens.size(), 0);
    CheckMetaEqual(spec.build(0, m, &accepted, &zero_drafts), base);
  }
}
