// ROCm op-table registration (BACKEND-ROCM).
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt::rocm {

void RmsNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual);
void EmbeddingKernelRocm(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids);
void MulScalarKernelRocm(Queue& q, Tensor& out, const Tensor& x, double scalar);
void SoftCapKernelRocm(Queue& q, Tensor& out, const Tensor& x, double cap);
void AddKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void GeluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void SiluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void MatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void MatmulBTKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void RopeCosSinCacheKernelRocm(Queue& q, Tensor& cos_sin, const Tensor& pos,
                               const RopeArgs& args);
void GreedyArgmaxKernelRocm(Queue& q, Tensor& out, const Tensor& logits);
void CastBf16KernelRocm(Queue& q, Tensor& out, const Tensor& in);
void CastF32KernelRocm(Queue& q, Tensor& out, const Tensor& in);
void QkvSplitKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                        const Tensor& qkv);
void LayerNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
                         const Tensor* bias, const LayerNormArgs& args);
void ReluKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void RopeNeoxKernelRocm(Queue& q, Tensor& qs, Tensor& ks, const Tensor& pos,
                        const RopeArgs& args);
void RopeFromCacheKernelRocm(Queue& q, Tensor& qs, Tensor* ks, const Tensor& positions,
                             const Tensor& cache, const RopeArgs& args);
void ReshapeAndCacheKernelRocm(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                               Tensor& v_cache, const Tensor& slot_mapping);
void PagedAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args);
void GeluTanhKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void GeluErfKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void MoeRouterTopKKernelRocm(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args, const Tensor* bias);
// BACKEND-ROCM-GDN-KERNELS family 1 (rocm_gdn_state.hip): the indexed state I/O
// pair `IndexedGdnOpsNative()` requires (issue #41, spec rocm-gdn-kernels.md).
void GdnStateGatherKernelRocm(Queue& q, Tensor& working, const Tensor& cache,
                              const Tensor& state_idx, const Tensor* has_initial_state);
void GdnStateScatterKernelRocm(Queue& q, Tensor& cache, const Tensor& working,
                               const Tensor& state_idx);
// BACKEND-ROCM-GDN-KERNELS family 2 (rocm_gdn_conv.hip): causal conv1d prefill
// fwd + decode update (gdn-semantics §2/§3).
void CausalConv1dFwdKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                               const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                               const Tensor& his, const CausalConv1dArgs& args);
void CausalConv1dUpdateKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                                  const Tensor* bias, Tensor& conv_state,
                                  const Tensor* conv_state_indices,
                                  const CausalConv1dArgs& args);
// BACKEND-ROCM-GDN-KERNELS family 3 (rocm_gdn_postconv.hip): the fused
// post-conv glue (conv-split + q/k l2norm + g/beta), gdn-semantics §4/§6.
void GdnPostConvKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                           Tensor& g_out, Tensor& beta_out, const Tensor& conv,
                           const Tensor& araw, const Tensor& braw, const Tensor& a_log,
                           const Tensor& dt_bias, const L2NormArgs& args);
// BACKEND-ROCM-GDN-KERNELS family 4 (rocm_gdn_scan.hip): the gated-delta
// recurrence over the portable scan (gdn-semantics §7/§8).
void GdnPrefillKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta,
                          Tensor& state, const Tensor& qsl, const GdnArgs& args);
void GdnDecodeKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor* state_idx, const GdnArgs& args);

namespace {

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernelRocm)));
    RegisterOp(OpId::kEmbedding, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelRocm)));
    RegisterOp(OpId::kMulScalar, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MulScalarFn>(&MulScalarKernelRocm)));
    RegisterOp(OpId::kSoftCap, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SoftCapFn>(&SoftCapKernelRocm)));
    RegisterOp(OpId::kAdd, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernelRocm)));
    RegisterOp(OpId::kGeluAndMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&GeluAndMulKernelRocm)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernelRocm)));
    RegisterOp(OpId::kMatmul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernelRocm)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernelRocm)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernelRocm)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernelRocm)));
    RegisterOp(OpId::kCastBf16, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16KernelRocm)));
    RegisterOp(OpId::kCastF32, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32KernelRocm)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernelRocm)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernelRocm)));
    RegisterOp(OpId::kRelu, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernelRocm)));
    RegisterOp(OpId::kGeluTanh, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluTanhKernelRocm)));
    RegisterOp(OpId::kGeluErf, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluErfKernelRocm)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernelRocm)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RopeFromCacheFn>(&RopeFromCacheKernelRocm)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernelRocm)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<PagedAttentionFn>(&PagedAttentionKernelRocm)));
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernelRocm)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GdnStateGatherFn>(&GdnStateGatherKernelRocm)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GdnStateScatterFn>(&GdnStateScatterKernelRocm)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernelRocm)));
    RegisterOp(OpId::kCausalConv1dUpdate, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernelRocm)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GdnPostConvFn>(&GdnPostConvKernelRocm)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernelRocm)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernelRocm)));
  }
} registrar;

}  // namespace
}  // namespace vt::rocm
