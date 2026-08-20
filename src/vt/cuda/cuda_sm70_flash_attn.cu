// SPDX-License-Identifier: Apache-2.0
// sm70 decode attention — Volta tensor-core fast path (Phase 2 brick F).
//
// NOTE: the D==256 parallel decode (XQA partition + split reduce) and the
// D==256 BM32 split-KV prefill below are ported from 1Cat-vLLM's
// flash-attention-v100 (BSD-3-Clause, c. 2025 D.Skryabin; see
// third_party/flash_attn_v100/COPYING-flash-attn-v100).
//
// One warp-block per (query token, q-head). QK^T is a m16n16k16 fp16 WMMA
// over 4 d-panels (D==64) using the vendored volta fragment library; the flash
// online softmax + weighted-V stays fp32 SIMT and mirrors the reference
// `PagedAttentionKernel` (cuda_paged_attn.cu) token-for-token, so the
// reference launched on identical paged tensors is a directly-comparable
// oracle (differences only from fp16-MMA vs fp32-fma score rounding).
//
// Scope (honest): head_dim==64, fp16 Q/KV, paged KV, causal decode, no window,
// softcap==0. Anything else: leave to the reference. Exposes two extern "C"
// host entries — my kernel and the reference oracle — so the parity test needs
// no torch.

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "vt/cuda/cuda_device_caps.h"

#include "../../../third_party/flash_attn_v100/include/fused_mma.h"
#include "../../../third_party/flash_attn_v100/kernel/flash_v100_traits.cuh"

namespace vt::cuda {

constexpr int kF2Keys = 16;    // 16-key tile (m16n16k16 N)
constexpr int kF2Threads = 32;

// One block per (token, q-head); 32 threads (one warp). Head-dim-TD templated
// (instantiated for 64 and 128) so real Llama/Mistral/Qwen head widths run.
// Decode seq-len routing: d==256 requests with seq_len at/above the parallel
// XQA threshold are handled by the partition/reduce kernels below; the
// sequential 1-warp kernel here only serves the retained short-seq shapes
// (and every d != 256 shape, for which the guard is compile-time dead).
// The host passes the routing threshold in `fa2_seq_route` (0 = never route:
// used under CUDA-graph capture, where the parallel workspace cannot be
// (re)allocated). The kernel body below the guard is byte-identical for all
// shapes that do NOT take the parallel route.
template <int TD>
__global__ void Sm70Fa2DecodeAttn(
    float* out, const __half* query, const __half* k_cache, const __half* v_cache,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* query_start,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, float softcap,
    int64_t fa2_seq_route) {
  if (d != TD) return;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (h >= hq) return;
  const int64_t seqlen = seq_lens[t];
  if (TD == 256 && fa2_seq_route > 0 && seqlen >= fa2_seq_route) return;
  const int64_t g = h / (hq / num_kv);
  const int64_t qoff = (t * hq + h) * d;

  extern __shared__ __half sm[];  // carve in __half
  __half* sq = sm;                // [16][TD] fp16, row0 = the query row
  __half* sk = sq + 16 * TD;      // key tile [16][TD] key-major
  __half* sb = sk + 16 * TD;      // transposed B panel [16][16]
  float* sc = reinterpret_cast<float*>(sb + 16 * 16);  // [16][16] f32 scores
  float* sacc = sc + 16 * 16;     // [TD] running acc

  for (int e = threadIdx.x; e < TD; e += blockDim.x)
    sq[e] = query[qoff + e];
  for (int e = threadIdx.x + TD; e < 16 * TD; e += blockDim.x)
    sq[e] = __ushort_as_half(0);
  for (int e = threadIdx.x; e < TD; e += blockDim.x) sacc[e] = 0.f;
  __syncthreads();

  float m = -CUDART_INF_F, l = 0.f;
  const int64_t jmax = seqlen - 1;  // decode causal: keys 0..seqlen-1

  for (int64_t jb = 0; jb <= jmax; jb += kF2Keys) {
    const int nb = (int)((jmax - jb + 1) < kF2Keys ? (jmax - jb + 1) : kF2Keys);

    // Pass A: zero the B/intermediate tiles, then load valid keys into sk
    // (key-major [key][d], fp16), masked keys left at 0 (their score ignored).
    for (int e = threadIdx.x; e < 16 * 64; e += blockDim.x)
      sk[e] = __ushort_as_half(0);
    __syncthreads();
    for (int e = threadIdx.x; e < nb * d; e += blockDim.x) {
      const int key = jb + (e / d);
      const int cc_ = e % d;
      const int64_t blk = block_table[(int64_t)t * bt_row + (key / block_size) * bt_col];
      const int64_t off = key % block_size;
      sk[(e / d) * d + cc_] = k_cache[blk * kc_blk + off * kc_pg + g * kc_hd + cc_];
    }
    __syncthreads();

    // -- QK^T: accumulate C[16x16] over the 4 d-panels --
    // A-panel i: Q rows (row0 valid) x cols [p*16..p*16+16) -> load row-major
    //            16x16 from sq with ldm 16.
    // B-panel:   B[k][key] = sk[key][p*16 + k] -> build transpose in sb,
    //            load matrix_b row-major ldm 16.
    volta::fragment<volta::accumulator, 16, 16, 16, float> C;
    fill_fragment(C, 0.f);
    for (int p = 0; p < d / 16; ++p) {
      volta::fragment<volta::matrix_a, 16, 16, 16, __half, volta::row_major> A;
      volta::fragment<volta::matrix_b, 16, 16, 16, __half, volta::row_major> B;
      load_matrix_sync(A, sq + p * 16, 16);
      for (int e = threadIdx.x; e < 16 * 16; e += blockDim.x) {
        const int kk = e / 16, key = e % 16;  // B[kk][key]
        sb[e] = sk[key * d + (p * 16 + kk)];
      }
      __syncthreads();
      load_matrix_sync(B, sb, 16);
      mma_sync(C, A, B, C);
      __syncwarp();
    }
    // Store C row-major (row 0 = the query's scores for keys jb..jb+15).
    store_matrix_sync(sc, C, 16, volta::mem_row_major);
    __syncthreads();

    // -- fp32 flash online, mirroring the reference exactly (key-major) --
    for (int e = 0; e < nb; ++e) {
      const float s = sc[e] * scale;          // softcap==0 -> ApplySoftcap passthrough
      const float m0 = m;
      const float m1 = fmaxf(m0, s);
      const float corr = expf(m0 - m1);       // 0 when m0 == -inf
      const float pw = expf(s - m1);
      const int64_t key = jb + e;
      const int64_t blk = block_table[(int64_t)t * bt_row + (key / block_size) * bt_col];
      const int64_t off = key % block_size;
      for (int ee2 = threadIdx.x; ee2 < TD; ee2 += blockDim.x)
        sacc[ee2] = sacc[ee2] * corr +
                    pw * __half2float(v_cache[blk * vc_blk + off * vc_pg + g * vc_hd + ee2]);
      __syncwarp();
      m = m1;
      l = l * corr + pw;  // all lanes: same s, so m/l stay uniform
    }
    __syncthreads();
  }

  const float inv = 1.0f / l;
  for (int e = threadIdx.x; e < TD; e += blockDim.x) out[qoff + e] = sacc[e] * inv;
}


// ====== FA2-V100 DECODE, D==256 PARALLEL (XQA-256-wide + partition + split-reduce) ==
// Ported from 1Cat-vLLM/flash-attention-v100 flash_decode_paged.cu
// (flash_attention_decode_xqa_tc_partition_kernel_256_wide +
//  the reduce_stats/output pair). One 256-thread (8-warp) block per
// (request, kv-head, partition); per kv-head, GROUP = q_per_kv query heads are
// QK^T'd in one 8x32x16 WMMA batch over 64-half QK panels, softmaxed one full
// warp per row, then PV is a SIMT f32 dot over the V cache. Partitions write
// f32 partial sums + per-partition max/sum; one route-aware split reduce
// merges them in ascending partition order (the same key order the sequential
// kernel accumulates) — the f32 order the short path uses.
// Seq-len sawtooth (the Y-sawtooth constants from the reference, folded from
// its env patchwork into compile-time constants):
//   seq < 111104            -> 256-token partitions
//   111104 <= seq < 147841  -> 1024-token partitions
//   147841 <= seq < 258176  -> 256-token partitions
//   seq >= 258176           -> 1024-token partitions
// All routes (256/1024) land in the same front-of-buffer partial slots and the
// merge kernel re-resolves the partition size per request on device, so one
// stats + output pair serves every length. Route is gated to d==256 and
// q_per_kv in {4,6,8}; everything else stays on the sequential kernel.
constexpr int kVtXqWarpSize = 32;
constexpr int kVtXq256WideWarpCount = 8;
constexpr int kVtXq256Threads = kVtXq256WideWarpCount * kVtXqWarpSize;
constexpr int kVtXqBlockN = 128;               // key tokens per QK block
constexpr int kVtXq256WideBlockM = 8;          // query rows staged per block
constexpr int kVtXq256WideThreadsPerRow = kVtXqWarpSize;
constexpr int kVtXqPageIdsCapacity = kVtXqBlockN / 16;
constexpr int kVtXqD = 256;
constexpr int kVtXqQKPanelDim = 64;            // d-panels: 256/64 = 4
constexpr int kVtXqPVPanelDim = 128;           // d-panels: 256/128 = 2
constexpr int kVtXqPaddedQStride = 264;        // padded to dodge smem banking
constexpr int kVtXqPaddedKVStride = 136;
constexpr int kVtXqKVStepUint4 = kVtXqPaddedKVStride / 8;
constexpr int kVtXqQKStepUint4 = 72 / 8;       // QK-pipeline K stride (72)
constexpr int kVtXqQKPipelinePanelDim = 64;
constexpr float kVtXqNegInf = -1.0e30f;

// ---- sawtooth thresholds (flash_decode_paged.cu:247-263 defaults) ----
constexpr int kVtXqSeqRouteBegin = 512;        // below: sequential path
constexpr int kVtXqP1024MidSeqLen = 111104;    // [mid, long) + [final, inf) -> p1024
constexpr int kVtXqP256LongSeqLen = 147841;    // [long, final) -> back to p256
constexpr int kVtXqP1024FinalSeqLen = 258176;  // >= final -> p1024
// Host knob for capture-mode: when the workspace cannot be (re)allocated this
// threshold (in seq-len units) is raised so every d==256 request stays on the
// sequential kernel. Pass kVtXqSeqRouteNever to disable routing entirely.
constexpr int64_t kVtXqSeqRouteNever = int64_t{1} << 40;

// Which XQA partition width serves a seq length (device-side selection, shared
// by the partition kernels and the cross-partition reduce).
__device__ __forceinline__ bool vt_xq_route_p256(const int seq_len) {
  return (seq_len >= kVtXqSeqRouteBegin && seq_len < kVtXqP1024MidSeqLen) ||
         (seq_len >= kVtXqP256LongSeqLen && seq_len < kVtXqP1024FinalSeqLen);
}
__device__ __forceinline__ bool vt_xq_route_p1024(const int seq_len) {
  return (seq_len >= kVtXqP1024MidSeqLen && seq_len < kVtXqP256LongSeqLen) ||
         seq_len >= kVtXqP1024FinalSeqLen;
}
__device__ __forceinline__ int vt_xq_sawtooth_partition_size(const int seq_len) {
  return vt_xq_route_p1024(seq_len) ? 1024 : 256;
}

struct alignas(256) VtXq256SmemLayout {
  alignas(16) __half q[kVtXq256WideBlockM * kVtXqPaddedQStride];
  union {
    alignas(16) __half k[kVtXqBlockN * kVtXqPaddedKVStride];
    alignas(16) __half v[kVtXqBlockN * kVtXqPaddedKVStride];
  } reuse_kv;
  // P kept in fp32 (fp16 KV) living in the score buffer.
  alignas(16) float score[kVtXq256WideBlockM * kVtXqBlockN];
  alignas(16) float row_max[kVtXq256WideBlockM];
  alignas(16) float row_sum[kVtXq256WideBlockM];
  alignas(16) int page_ids[kVtXqPageIdsCapacity];
  __device__ __forceinline__ __half* k_buffer(int) { return reuse_kv.k; }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

struct alignas(256) VtXq256QkPipelineSmemLayout {
  alignas(16) __half q[kVtXq256WideBlockM * kVtXqPaddedQStride];
  union {
    struct {
      alignas(16)
          __half panel[2][kVtXqBlockN * kVtXqQKStepUint4 * 8];  // KQ pipeline
    } qk;
    alignas(16) __half v[kVtXqBlockN * kVtXqPaddedKVStride];
  } reuse_kv;
  alignas(16) float score[kVtXq256WideBlockM * kVtXqBlockN];
  alignas(16) float row_max[kVtXq256WideBlockM];
  alignas(16) float row_sum[kVtXq256WideBlockM];
  alignas(16) int page_ids[kVtXqPageIdsCapacity];
  __device__ __forceinline__ __half* k_buffer(int index) {
    return reuse_kv.qk.panel[index];
  }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <bool PIPELINE>
struct VtXqSmemSelector {
  using Type = VtXq256SmemLayout;
};
template <>
struct VtXqSmemSelector<true> {
  using Type = VtXq256QkPipelineSmemLayout;
};

// Load a 16-byte K/V vector for (page_ids[logical_block], block_offset);
// fp16 KV only. BLOCK_SIZE=0 falls back to the runtime block_size argument.
template <int BLOCK_SIZE>
__device__ __forceinline__ uint4 vt_xq_load_kv_vector(
    const __half* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  int logical_block;
  int block_offset;
  if constexpr (BLOCK_SIZE == 16) {
    logical_block = token_offset >> 4;
    block_offset = token_offset & 15;
  } else if constexpr (BLOCK_SIZE == 784) {
    logical_block = token_offset / 784;
    block_offset = token_offset - logical_block * 784;
  } else {
    logical_block = token_offset / block_size;
    block_offset = token_offset - logical_block * block_size;
  }
  const int physical_block = page_ids[logical_block];
  const int64_t physical_offset =
      static_cast<int64_t>(physical_block) * block_stride +
      static_cast<int64_t>(block_offset) * token_stride +
      static_cast<int64_t>(kv_head_idx) * head_stride + panel_offset;
  const uint4* cache_vec = reinterpret_cast<const uint4*>(kv_cache);
  return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
}

template <int BLOCK_SIZE, int NUM_THREADS>
__device__ __forceinline__ void vt_xq_load_kv_panel(
    __half* __restrict__ shared_kv, const __half* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx = threadIdx.x) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  for (int copy_idx = copy_thread_idx; copy_idx < copy_count;
       copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] =
        vt_xq_load_kv_vector<BLOCK_SIZE>(
            kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
            tile_page_offset, kv_tile_start, block_size, kv_head_idx,
            block_stride, token_stride, head_stride, panel_offset);
  }
}

template <int BLOCK_SIZE, int NUM_THREADS>
__device__ __forceinline__ void vt_xq_load_kv_panel_staged(
    __half* __restrict__ shared_kv, const __half* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx) {
  constexpr int kLoadStages = 4;
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  for (int copy_base = copy_thread_idx; copy_base < copy_count;
       copy_base += NUM_THREADS * kLoadStages) {
    uint4 staged[kLoadStages];
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        staged[stage] = vt_xq_load_kv_vector<BLOCK_SIZE>(
            kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
            tile_page_offset, kv_tile_start, block_size, kv_head_idx,
            block_stride, token_stride, head_stride, panel_offset);
      }
    }
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        const int row = copy_idx / panel_d_stride_uint4;
        const int vec_col = copy_idx % panel_d_stride_uint4;
        shared_vec[row * kv_smem_stride_uint4 + vec_col] = staged[stage];
      }
    }
  }
  for (int copy_idx = copy_thread_idx + copy_count;
       copy_idx < kVtXqBlockN * panel_d_stride_uint4; copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] =
        make_uint4(0, 0, 0, 0);
  }
}

// D=256 XQA partition kernel: one 8-warp block per (batch, kv_head,
// partition_idx). GROUP_SIZE = q_per_kv in {4,6,8}. PADDED_SMEM picks padded
// (264/136) or dense (256/128) shared strides; for GROUP==6 with QK_SW_PIPE the
// QK panel K loads run on producer warps into a double-buffered pipeline and
// the 4 QK consumer warps read panel-by-panel.
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM,
          int NUM_THREADS = kVtXq256Threads, int MIN_BLOCKS_PER_SM = 1,
          int BLOCK_SIZE = 0, bool PIPELINE = false>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    VtXqDecodePartitionKernel(
        const __half* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ tmp_out,
        float* __restrict__ max_logits, float* __restrict__ exp_sums,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int batch_size, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t bt_row,
        const int64_t bt_col, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale) {
  constexpr int kQKPanelDim = PIPELINE ? kVtXqQKPipelinePanelDim : kVtXqPVPanelDim;
  constexpr int kNumQKPanels = kVtXqD / kQKPanelDim;  // panel-dim
  constexpr int kNumPVPanels = kVtXqD / kVtXqPVPanelDim;
  using SmemLayout = typename VtXqSmemSelector<PIPELINE>::Type;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "D=256 XQA supports q_per_kv in {4,6,8}");
  static_assert(!PIPELINE || (GROUP_SIZE == 6 && (NUM_THREADS == 6 * kVtXqWarpSize ||
                                                  NUM_THREADS == 8 * kVtXqWarpSize)),
                "QK SW pipeline requires the six/eight-warp G6 kernel");
  static_assert(!PIPELINE || MIN_BLOCKS_PER_SM == 2, "pipeline dual-CTA carveout");

  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv) return;
  const int seq_len = seq_lens[batch_idx];
  if (seq_len <= 0) return;
  // (compile-time) partition-width route: the launch carries the matching
  // template; the sawtooth overlap between the two launched kernels is
  // disambiguated by this check.
  const bool routed = (PARTITION_SIZE == 1024) ? vt_xq_route_p1024(seq_len)
                                               : vt_xq_route_p256(seq_len);
  if (!routed) return;
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (start_token_idx >= seq_len) return;
  const int seq_num_partitions = (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  if (partition_idx >= seq_num_partitions) return;
  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) return;

  const int tid = threadIdx.x;
  const int warp_id = tid / kVtXqWarpSize;
  const int lane_id = tid % kVtXqWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kVtXqBlockN - 1) / kVtXqBlockN;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sQ = smem.q;
  __half* sK = smem.k_buffer(0);
  __half* sV = smem.v_buffer();
  float* sS = smem.score;
  float row_max_reg = kVtXqNegInf;
  float row_sum_reg = 0.f;
  constexpr int kRowAcc = kVtXqD / kVtXqWarpSize;  // 8 per thread
  float out_acc[kRowAcc];
#pragma unroll
  for (int i = 0; i < kRowAcc; ++i) out_acc[i] = 0.f;

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  constexpr int q_global_stride_uint4 = kVtXqD / 8;
  constexpr int q_smem_stride_uint4 = (PADDED_SMEM ? kVtXqPaddedQStride : kVtXqD) / 8;
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4; idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset = static_cast<int64_t>(batch_idx) * q_stride0 +
                             static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid; idx < (kVtXq256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kVtXqBlockN;
    const int valid_k_rows = min(kVtXqBlockN, part_tokens - block_n * kVtXqBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table[static_cast<int64_t>(batch_idx) * bt_row +
                                                   static_cast<int64_t>(start_page + idx) * bt_col]);
    }
    __syncthreads();

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kVtXqBlockN) {
const int valid_kv_tile_rows =
          min(kVtXqBlockN, valid_k_rows - kv_tile_start);
      volta::fragment<volta::matrix_a, 8, 32, 16, half, volta::row_major> qk_a_frag;
      volta::fragment<volta::matrix_b, 8, 32, 16, half, volta::col_major> qk_b_frag;
      volta::fragment<volta::accumulator, 8, 32, 16, float> qk_acc_frag;
      if (warp_id < kVtXqBlockN / 32) {
        fill_fragment(qk_acc_frag, 0.f);
      }

      if constexpr (PIPELINE) {
        constexpr int kConsumerWarps = kVtXqBlockN / 32;
        constexpr int kProducerThreads =
            NUM_THREADS - kConsumerWarps * kVtXqWarpSize;
        const int producer_tid = tid - kConsumerWarps * kVtXqWarpSize;
        if (producer_tid >= 0) {
          vt_xq_load_kv_panel_staged<BLOCK_SIZE, kProducerThreads>(
              smem.k_buffer(0), k_cache, smem.page_ids, valid_kv_tile_rows,
              kQKPanelDim / 8, kVtXqQKStepUint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, 0, producer_tid);
        }
        __syncthreads();
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          if (producer_tid >= 0 && panel_idx + 1 < kNumQKPanels) {
            vt_xq_load_kv_panel_staged<BLOCK_SIZE, kProducerThreads>(
                smem.k_buffer((panel_idx + 1) & 1), k_cache, smem.page_ids,
                valid_kv_tile_rows, kQKPanelDim / 8, kVtXqQKStepUint4,
                tile_page_offset, kv_tile_start, block_size, kv_head_idx,
                k_block_stride, k_token_stride, k_head_stride,
                panel_offset + kQKPanelDim, producer_tid);
          }
          if (warp_id < kConsumerWarps) {
            const int tile_n = warp_id * 32;
            const __half* sK_panel = smem.k_buffer(panel_idx & 1);
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / 16); ++k_tile) {
              const int k_offset = k_tile * 16;
              load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                               (PADDED_SMEM ? kVtXqPaddedQStride : 256));
              load_matrix_sync(qk_b_frag, sK_panel + tile_n * 72 + k_offset, 72);
              mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      } else {
        if (warp_id < kVtXqBlockN / 32) {
          fill_fragment(qk_acc_frag, 0.f);
        }
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          vt_xq_load_kv_panel<BLOCK_SIZE, NUM_THREADS>(
              sK, k_cache, smem.page_ids, valid_kv_tile_rows, kQKPanelDim / 8,
              kVtXqKVStepUint4, tile_page_offset, kv_tile_start, block_size,
              kv_head_idx, k_block_stride, k_token_stride, k_head_stride,
              panel_offset);
          for (int idx = tid + valid_kv_tile_rows * (kQKPanelDim / 8);
               idx < kVtXqBlockN * (kQKPanelDim / 8); idx += NUM_THREADS) {
            const int row = idx / (kQKPanelDim / 8);
            const int vec_col = idx % (kQKPanelDim / 8);
            reinterpret_cast<uint4*>(sK)[row * kVtXqKVStepUint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();
          if (warp_id < kVtXqBlockN / 32) {
            const int tile_n = warp_id * 32;
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / 16); ++k_tile) {
              const int k_offset = k_tile * 16;
              load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                               (PADDED_SMEM ? kVtXqPaddedQStride : 256));
              load_matrix_sync(qk_b_frag, sK + tile_n * (PADDED_SMEM ? kVtXqPaddedKVStride : 128) + k_offset,
                               (PADDED_SMEM ? kVtXqPaddedKVStride : 128));
              mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      }

      if (warp_id < kVtXqBlockN / 32) {
#pragma unroll
        for (int i = 0; i < 8; ++i) qk_acc_frag.x[i] *= softmax_scale;
        const int tile_n = warp_id * 32;
        store_matrix_sync(sS + kv_tile_start + tile_n, qk_acc_frag,
                          kVtXqBlockN, volta::mem_row_major);
      } else {
        fill_fragment(qk_acc_frag, 0.f);
      }
      __syncthreads();

      // fp32 online softmax per row (whole warp per row), P written fp32 into
      // the score buffer (fp16 KV -> P stays fp32: no half round-trip).
      if (tid < GROUP_SIZE * kVtXq256WideThreadsPerRow) {
        const int row = tid / kVtXq256WideThreadsPerRow;
        const int thread_in_row = tid % kVtXq256WideThreadsPerRow;
        const unsigned mask = 0xffffffffu;
        float* sS_row_f = sS + row * kVtXqBlockN;
        const int vec_cols = valid_k_rows >> 2;
        const int tail_start = vec_cols << 2;
        const int vec_col = thread_in_row;

        float thread_max = kVtXqNegInf;
        if (vec_col < vec_cols) {
          const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
          thread_max = fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
        }
#pragma unroll
        for (int c = tail_start + thread_in_row; c < valid_k_rows;
             c += kVtXq256WideThreadsPerRow) {
          thread_max = fmaxf(thread_max, sS_row_f[c]);
        }
#pragma unroll
        for (int o = kVtXq256WideThreadsPerRow / 2; o > 0; o >>= 1) {
          thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kVtXqWarpSize));
        }
        const float row_max = __shfl_sync(mask, thread_max, 0, kVtXqWarpSize);
        const float old_max = __shfl_sync(mask, row_max_reg, 0, kVtXqWarpSize);
        const float new_max = fmaxf(old_max, row_max);
        const float exp_diff = __expf(old_max - new_max);

        float thread_sum = 0.f;
        if (vec_col < vec_cols) {
          const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
          const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
          const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
          const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
          const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
          thread_sum += (e0 + e1) + (e2 + e3);
          reinterpret_cast<float4*>(sS_row_f)[vec_col] = make_float4(e0, e1, e2, e3);
        }
#pragma unroll
        for (int c = tail_start + thread_in_row; c < kVtXqBlockN;
             c += kVtXq256WideThreadsPerRow) {
          const float v = (c < valid_k_rows) ? sS_row_f[c] : kVtXqNegInf;
          const float e = __expf(fmaxf(v - new_max, -80.0f));
          thread_sum += (c < valid_k_rows) ? e : 0.0f;
          sS_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
        }
#pragma unroll
        for (int o = kVtXq256WideThreadsPerRow / 2; o > 0; o >>= 1) {
          thread_sum += __shfl_down_sync(mask, thread_sum, o, kVtXqWarpSize);
        }
const float row_sum = __shfl_sync(mask, thread_sum, 0, kVtXqWarpSize);
        const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kVtXqWarpSize);
        if (thread_in_row == 0) {
          row_sum_reg = exp_diff * old_sum + row_sum;
          row_max_reg = new_max;
        }
        if (block_n > 0) {
#pragma unroll
          for (int i = 0; i < kRowAcc; ++i) out_acc[i] *= exp_diff;
        }
      }
      __syncthreads();

      for (int panel = 0; panel < kNumPVPanels; ++panel) {
        const int panel_offset = panel * kVtXqPVPanelDim;
        for (int kv_tile = 0; kv_tile < valid_kv_tile_rows; kv_tile += kVtXqBlockN) {
          const int valid_k_v_rows =
              min(kVtXqBlockN, valid_kv_tile_rows - kv_tile);
          vt_xq_load_kv_panel<BLOCK_SIZE, NUM_THREADS>(
              sV, v_cache, smem.page_ids, valid_k_v_rows, kVtXqPVPanelDim / 8,
              kVtXqKVStepUint4, tile_page_offset, kv_tile_start + kv_tile,
              block_size, kv_head_idx, v_block_stride, v_token_stride,
              v_head_stride, panel_offset);
          for (int idx = tid + valid_k_v_rows * (kVtXqPVPanelDim / 8);
               idx < kVtXqBlockN * (kVtXqPVPanelDim / 8); idx += NUM_THREADS) {
            const int row = idx / (kVtXqPVPanelDim / 8);
            const int vec_col = idx % (kVtXqPVPanelDim / 8);
            reinterpret_cast<uint4*>(sV)[row * kVtXqKVStepUint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();
          if (tid < GROUP_SIZE * kVtXq256WideThreadsPerRow) {
            const int row = tid / kVtXq256WideThreadsPerRow;
            const int thread_in_row = tid % kVtXq256WideThreadsPerRow;
            const float* sS_row_f = sS + row * kVtXqBlockN + kv_tile;
#pragma unroll
            for (int token = 0; token < kVtXqBlockN; ++token) {
              if (token >= valid_k_v_rows) break;
              const float prob = sS_row_f[token];  // fp32 P (fp16 KV)
              const __half* sV_row =
                  sV + (kv_tile + token) * (PADDED_SMEM ? kVtXqPaddedKVStride : 128);
#pragma unroll
              for (int d_iter = 0; d_iter < (kVtXqPVPanelDim / 32); ++d_iter) {
                const int local_d = lane_id + d_iter * 32;
                const int acc_idx = panel * (kVtXqPVPanelDim / 32) + d_iter;
                out_acc[acc_idx] =
                    fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
              }
            }
          }
          __syncthreads();
        }
      }
    }
  }

  if (tid < GROUP_SIZE * kVtXq256WideThreadsPerRow) {
    const int row = tid / kVtXq256WideThreadsPerRow;
    const int thread_in_row = tid % kVtXq256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kVtXq256WideThreadsPerRow) {
    const int row = tid / kVtXq256WideThreadsPerRow;
    const int thread_in_row = tid % kVtXq256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    float* tmp_out_ptr = tmp_out +
                         static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                         static_cast<int64_t>(head_idx) * tmp_out_stride1 +
                         static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    for (int d = thread_in_row; d < kVtXqD; d += kVtXq256WideThreadsPerRow) {
      tmp_out_ptr[d] = out_acc[d / kVtXqWarpSize] * inv_row_sum;
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}

// Cross-partition stats: block reduce max/sum over the request's partitions,
// then materialize the per-partition weights into max_logits (overwritten)
// and the global sum into exp_sums[stats_base]. Per-request partition width
// resolved on device via the sawtooth so one stats+output pair serves both the
// p256 and p1024 legs.
__global__ void VtXqDecodeReduceStatsKernel(
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t stats_stride0, const int64_t stats_stride1) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (batch_idx >= batch_size || head_idx >= num_heads_q) return;
  const int seq_len = seq_lens[batch_idx];
  const int partition_size = vt_xq_sawtooth_partition_size(seq_len);
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len < kVtXqSeqRouteBegin || num_partitions <= 0) return;

  extern __shared__ float max_shared[];
  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float m = max_logits[stats_base + i];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  // 8-warp block reduce max.
  __shared__ float warp_scratch[8];
  {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float val = local_max;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, o));
    if (lane == 0) warp_scratch[warp] = val;
    __syncthreads();
    val = (threadIdx.x < 8) ? warp_scratch[lane] : -1.0e20f;
    if (warp == 0) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, o));
      if (lane == 0) warp_scratch[0] = val;
    }
    __syncthreads();
  }
  const float global_max = warp_scratch[0];

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float weight =
        exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
    max_logits[stats_base + i] = weight;
    local_sum += weight;
  }
  {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float val = local_sum;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_down_sync(0xffffffffu, val, o);
    if (lane == 0) warp_scratch[warp] = val;
    __syncthreads();
    val = (threadIdx.x < 8) ? warp_scratch[lane] : 0.f;
    if (warp == 0) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, o);
      if (lane == 0) warp_scratch[0] = val;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) exp_sums[stats_base] = warp_scratch[0];
}

// Cross-partition output: one 8-thread block per (request, head, 32-d tile),
// each thread accumulates one d in ascending partition order (f32 FMA, the
// short path's order), then writes the f32 output element.
template <int D_TILE>
__global__ void VtXqDecodeReduceOutputKernel(
    const float* __restrict__ tmp_out, const float* __restrict__ weights,
    const float* __restrict__ global_sums, const int* __restrict__ seq_lens,
    float* __restrict__ out, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1) {
  static_assert(D_TILE > 0 && D_TILE <= 32, "dim tile must fit one warp");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = blockIdx.z * D_TILE + threadIdx.x;
  if (batch_idx >= batch_size || head_idx >= num_heads_q || d >= 256) return;
  const int seq_len = seq_lens[batch_idx];
  const int partition_size = vt_xq_sawtooth_partition_size(seq_len);
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || seq_len < kVtXqSeqRouteBegin || num_partitions <= 0) {
    out[static_cast<int64_t>(batch_idx) * out_stride0 +
        static_cast<int64_t>(head_idx) * out_stride1 + d] = 0.f;
    return;
  }

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  const int64_t tmp_out_base = static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                               static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const float global_sum = global_sums[stats_base];
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  float acc = 0.f;
  for (int i = 0; i < num_partitions; ++i) {
    acc = fmaf(weights[stats_base + i],
               tmp_out[tmp_out_base + static_cast<int64_t>(i) * tmp_out_stride2 + d],
               acc);
  }
  out[static_cast<int64_t>(batch_idx) * out_stride0 +
      static_cast<int64_t>(head_idx) * out_stride1 + d] = acc * inv_global_sum;
}

// ====== FA2-V100 PREFILL (Phase 2 brick G) : WMMA fp16, D==64 ====
// One (request, q-head) block, 32 threads. Queries are 16-row slices, keys
// stream in 16-key tiles; QK^T per (q-slice x 16 keys) is a m16n16k16 WMMA over
// the 4 d-panels (D==64); the per-row flash (online max/sum + weighted-V) is
// fp32 SIMT with causal masking (key j valid iff j <= context + row). Same
// paged strides/contract as PagedAttentionKernel.
// Honest scope: fp16 Q/KV, D==64, causal, request q_len <= 64 (else decline).
constexpr int kPFMaxQ = 64;

template <int TD>
__global__ void Sm70FaPrefillAttn(
    float* out, const __half* query, const __half* k_cache, const __half* v_cache,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* q_start,
    int64_t num_reqs, int64_t hq, int64_t nkv, int64_t d,
    int64_t block_size, int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
    int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale,
    int64_t fa2_bm_route) {
  if (d != TD) return;
  const int64_t r = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (r >= num_reqs || h >= hq) return;
  const int qlen = (int)(q_start[r + 1] - q_start[r]);
  const int seqlen = (int)seq_lens[r];
  // d==256 requests served by the parallel BM32 split-KV prefill below are
  // skipped here (the guard is compile-time dead for every retained shape).
  if (TD == 256 && fa2_bm_route > 0 && qlen >= fa2_bm_route) return;
  if (qlen <= 0 || qlen > kPFMaxQ || seqlen <= 0 || h >= hq) return;
  const int ctxt = seqlen - qlen;
  const int64_t g = h / (hq / nkv);
  const int64_t qbase = q_start[r];   // global query-token row of req r

  extern __shared__ __half ps[];
  __half* sq = ps;                 // [16][TD] q-slice
  __half* sk = sq + 16 * TD;      // [16][TD] key tile
  __half* sv = sk + 16 * TD;      // [16][TD] v tile
  __half* sb = sv + 16 * TD;      // [16][16] B transpose
  __half* sa16 = sb + 16 * 16;      // [16][16] A tile (16-wide rows, for wmma)
  float* ss = reinterpret_cast<float*>(sa16 + 16 * 16);  // [16][16] scores
  float* acc = ss + 16 * 16;       // [16][64] running acc (current slice)
  float* mrow = acc + 16 * (TD);   // [16]
  float* lrow = mrow + 16;         // [16]

  for (int qs = 0; qs < qlen; qs += 16) {
    const int nrows = (qlen - qs) < 16 ? (qlen - qs) : 16;
    // load q-slice A (rows 0..nrows-1 valid, rest zero)
    for (int e = threadIdx.x; e < 16 * TD; e += blockDim.x) sq[e] = __ushort_as_half(0);
    for (int e = threadIdx.x; e < nrows * d; e += blockDim.x) {
      const int rr = e / (int)d, cc = e % (int)d;
      sq[rr * TD + cc] = query[((qbase + qs + rr) * hq + h) * d + cc];
    }
    for (int e = threadIdx.x; e < 16 * TD; e += blockDim.x) acc[e] = 0.f;
    for (int e = threadIdx.x; e < 16; e += blockDim.x) { mrow[e] = -CUDART_INF_F; lrow[e] = 0.f; }
    __syncthreads();
    // -- key stream over all keys [0, seqlen) --
    for (int64_t jb = 0; jb < (int64_t)seqlen; jb += kF2Keys) {
      const int nkeys = (int)(((int)seqlen - (int)jb) < kF2Keys ? ((int)seqlen - (int)jb) : kF2Keys);
      for (int e = threadIdx.x; e < 16 * TD; e += blockDim.x) sk[e] = __ushort_as_half(0);
      for (int e = threadIdx.x; e < 16 * TD; e += blockDim.x) sv[e] = __ushort_as_half(0);
      for (int e = threadIdx.x; e < nkeys * d; e += blockDim.x) {
        const int key = (int)jb + (e / d);
        const int cc = e % d;
        const int64_t blk = block_table[r * bt_row + (key / block_size) * bt_col];
        const int64_t off = key % block_size;
        sk[(e / d) * d + cc] = k_cache[blk * kc_blk + off * kc_pg + g * kc_hd + cc];
        sv[(e / d) * d + cc] = v_cache[blk * vc_blk + off * vc_pg + g * vc_hd + cc];
      }
      __syncthreads();
      // QK^T for this (slice, keytile): 4 d-panels -> S[16][16]
      volta::fragment<volta::accumulator, 16, 16, 16, float> C;
      fill_fragment(C, 0.f);
      for (int p = 0; p < d / 16; ++p) {
        // A16: row-major 16x16 with stride 16: [r][k] = query row r col p*16+k.
        for (int e = threadIdx.x; e < 16 * 16; e += blockDim.x) {
          const int rr = e / 16, kk = e % 16;
          sa16[e] = (rr < nrows) ? sq[rr * TD + (p * 16 + kk)] : __ushort_as_half(0);
        }
        // B transpose: [k][key] = sk[key][p*16 + k].
        for (int e = threadIdx.x; e < 16 * 16; e += blockDim.x) {
          const int kk = e / 16, key = e % 16;
          sb[e] = sk[key * d + (p * 16 + kk)];
        }
        __syncthreads();
        volta::fragment<volta::matrix_a, 16, 16, 16, __half, volta::row_major> A;
        volta::fragment<volta::matrix_b, 16, 16, 16, __half, volta::row_major> B;
        load_matrix_sync(A, sa16, 16);
        load_matrix_sync(B, sb, 16);
        mma_sync(C, A, B, C);
        __syncwarp();
      }
      store_matrix_sync(ss, C, 16, volta::mem_row_major);
      __syncthreads();
      // fp32 flash per row (batch of 16 keys per tile)
      for (int rr = 0; rr < nrows; ++rr) {
        const int absq = ctxt + qs + rr;      // absolute position of this query
        float mx = mrow[rr], ls = lrow[rr];
        float tm = -CUDART_INF_F;
        for (int ccelf = 0; ccelf < nkeys; ++ccelf) {
          const int kk = (int)jb + ccelf;
          if (kk <= absq) {
            const float s = ss[rr * 16 + ccelf] * scale;
            tm = fmaxf(tm, s);
          }
        }
        float mnew = fmaxf(mx, tm);
        float corr = expf(mx - mnew);
        for (int e = threadIdx.x; e < TD; e += blockDim.x) acc[rr * TD + e] *= corr;
        ls *= corr;
        for (int ccelf = 0; ccelf < 16; ++ccelf) {
          const int kf = (int)jb + ccelf;
          if (kf <= absq) {
            float w = expf(ss[rr * 16 + ccelf] * scale - mnew);
            ls += w;
            for (int e = threadIdx.x; e < TD; e += blockDim.x)
              acc[rr * TD + e] += w * __half2float(sv[ccelf * d + e]);
          }
        }
        __syncwarp();
        if (threadIdx.x == 0) { mrow[rr] = mnew; lrow[rr] = ls; }
        __syncthreads();
      }
      __syncthreads();
    }
    // write the slice out
    for (int e = threadIdx.x; e < nrows * d; e += blockDim.x) {
      const int rr = e / d, cc = e % d;
      const float inv = 1.0f / lrow[rr];
      out[((int64_t)(qbase + qs + rr) * hq + h) * d + cc] = acc[rr * d + cc] * inv;
    }
    __syncthreads();
  }
}

// ====== FA2-V100 PREFILL, D==256 PARALLEL (512-thread BM32 + splitkv3) ====
// Ported from 1Cat-vLLM/flash-attention-v100 fused_mha_forward_paged.cu
// (flash_attention_forward_paged_d256_bm32_splitkv3_*). Each 512-thread block
// processes a 32-query-row tile (M=32, N=128 keys, WMMA m16n16k16); the KV
// range is split 3 ways (splitkv3) with per-part f32 partial O + per-part
// row max/sum, merged by the 256-thread 8-threads-per-row kernel. All partial
// and merged accumulators stay f32 (the same f32 din accumulate the sequential
// prefill uses); only the final store writes the f32 output. The tile can
// straddle paged-KV blocks; pages here use the harness block_size (must be a
// multiple of 16) instead of the reference's fixed 784-token pages.
constexpr int kVtBmBlockM = 32;
constexpr int kVtBmBlockN = 128;
constexpr int kVtBmPanelM = 16;
constexpr int kVtBmSoftmaxN = 32;
constexpr int kVtBmD = 256;
constexpr int kVtBmThreads = 512;
constexpr int kVtBmSplitParts = 3;
constexpr int kVtBmPageSlots = kVtBmBlockN / 16;
constexpr int kVtBmPanels = kVtBmBlockN / kVtBmSoftmaxN;  // 4
constexpr int kVtBmProbElems = kVtBmPanelM * kVtBmSoftmaxN;
constexpr float kVtBmNegInf = -1.0e30f;
// Host gating for the prefill split: only whole-batch d==256 prefills with at
// least this many query rows (and pages divisible by 16) take the BM32 path.
constexpr int64_t kVtBmMinRows = 32;

template <int PROBABILITY_PANELS>
struct alignas(16) VtBmPhaseSharedStorage {
  __half query[kVtBmBlockM * kVtBmD];
  float score[kVtBmBlockM * kVtBmBlockN];
  __half probability_top[PROBABILITY_PANELS * kVtBmProbElems];
  __half probability_bottom[PROBABILITY_PANELS * kVtBmProbElems];
  float row_max[kVtBmBlockM];
  float row_sum[kVtBmBlockM];
  float row_exp_diff[PROBABILITY_PANELS * kVtBmBlockM];
  int page_idx[kVtBmPageSlots];
  int page_offset[kVtBmPageSlots];
  uint64_t k_tile_ptr[kVtBmPageSlots];
  uint64_t v_tile_ptr[kVtBmPageSlots];
  int block_index;
  int split_end_block;
  int batch_id;
  int kv_head_id;
  int q_len;
  int seq_len;
};

__device__ __forceinline__ int vt_bm_swizzled_row_slot(int row) {
  return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}
__device__ __forceinline__ int vt_bm_matrix_a_offset(int row, int column) {
  const int tile = column / 16;
  const int within_tile = column - tile * 16;
  const int plane = within_tile >> 3;
  const int inner = within_tile & 7;
  return tile * kVtBmPanelM * 16 + plane * kVtBmPanelM * 8 +
         vt_bm_swizzled_row_slot(row) * 8 + inner;
}
// Stage 32 query rows (global, contig per request) into the swizzled A layout.
__device__ __forceinline__ void vt_bm_stage_query(const __half* __restrict__ source,
                                                  __half* __restrict__ destination) {
  constexpr int kHALF_PER_UINT4 = 8;
  constexpr int kVectorsPerRow = kVtBmD / kHALF_PER_UINT4;
  constexpr int kVectorsPerPanel = kVtBmPanelM * kVectorsPerRow;
  constexpr int kQueryVectors = kVtBmBlockM * kVectorsPerRow;
  const uint4* src = reinterpret_cast<const uint4*>(source);
  uint4* dst = reinterpret_cast<uint4*>(destination);
#pragma unroll
  for (int index = threadIdx.x; index < kQueryVectors; index += kVtBmThreads) {
    const int row = index / kVectorsPerRow;
    const int vector_column = index % kVectorsPerRow;
    const int panel = row / kVtBmPanelM;
    const int panel_row = row - panel * kVtBmPanelM;
    const int k_tile = vector_column >> 1;
    const int plane = vector_column & 1;
    const int slot = vt_bm_swizzled_row_slot(panel_row);
    dst[panel * kVectorsPerPanel + k_tile * (2 * kVtBmPanelM) + plane * kVtBmPanelM + slot] =
        __ldg(src + index);
  }
}
__device__ __forceinline__ void vt_bm_stage_query_partial(
    const __half* __restrict__ source, __half* __restrict__ destination,
    int valid_rows) {
  constexpr int kHALF_PER_UINT4 = 8;
  constexpr int kVectorsPerRow = kVtBmD / kHALF_PER_UINT4;
  constexpr int kVectorsPerPanel = kVtBmPanelM * kVectorsPerRow;
  constexpr int kQueryVectors = kVtBmBlockM * kVectorsPerRow;
  const uint4* src = reinterpret_cast<const uint4*>(source);
  uint4* dst = reinterpret_cast<uint4*>(destination);
#pragma unroll
  for (int index = threadIdx.x; index < kQueryVectors; index += kVtBmThreads) {
    const int row = index / kVectorsPerRow;
    const int vector_column = index % kVectorsPerRow;
    const int panel = row / kVtBmPanelM;
    const int panel_row = row - panel * kVtBmPanelM;
    const int k_tile = vector_column >> 1;
    const int plane = vector_column & 1;
    const int slot = vt_bm_swizzled_row_slot(panel_row);
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row < valid_rows) value = __ldg(src + index);
    dst[panel * kVectorsPerPanel + k_tile * (2 * kVtBmPanelM) + plane * kVtBmPanelM + slot] =
        value;
  }
}
__device__ __forceinline__ void vt_bm_load_matrix_a(
    volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>& fragment,
    const __half* __restrict__ matrix, int k_offset) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
  const int slot = vt_bm_swizzled_row_slot(row);
  const int tile_offset = (k_offset / 16) * 16 * 16;
  uint32_t address = static_cast<uint32_t>(
      __cvta_generic_to_shared(matrix + tile_offset + slot * 8));
  uint32_t* words = reinterpret_cast<uint32_t*>(&fragment);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
               : "r"(address)
               : "memory");
  address += 16 * 8 * sizeof(__half);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7])
               : "r"(address)
               : "memory");
}
__device__ __forceinline__ int vt_bm_accumulator_row(int lane, int element) {
  const int row_base =
      (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  return row_base + ((element >> 1) & 1) * 2;
}
__device__ __forceinline__ int vt_bm_accumulator_column(int lane, int element) {
  const int column_base = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
  return column_base + (element & 1) + ((element >> 2) & 1) * 4;
}
__device__ __forceinline__ void vt_bm_spill_pair_scratch(
    float* __restrict__ score,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_top,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_bottom) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_pair = warp >> 1;
  const int warp_in_pair = warp & 1;
  const int pair_column = warp_pair * 32 + (lane & 15) * 2;
  const int lane_row = lane >> 4;
#pragma unroll
  for (int element_pair = 0; element_pair < 4; ++element_pair) {
    const int element = element_pair * 2;
    const int offset =
        (warp_in_pair * 16 + element_pair * 2 + lane_row) * kVtBmBlockN +
        pair_column;
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(score + offset));
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(address),
                 "r"(__float_as_uint(acc_top.x[element])),
                 "r"(__float_as_uint(acc_top.x[element + 1]))
                 : "memory");
    asm volatile("st.shared.v2.u32 [%0+4096], {%1, %2};" ::"r"(address),
                 "r"(__float_as_uint(acc_bottom.x[element])),
                 "r"(__float_as_uint(acc_bottom.x[element + 1]))
                 : "memory");
  }
}
__device__ __forceinline__ void vt_bm_reload_pair_scratch(
    const float* __restrict__ score,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_top,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_bottom) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_pair = warp >> 1;
  const int warp_in_pair = warp & 1;
  const int pair_column = warp_pair * 32 + (lane & 15) * 2;
  const int lane_row = lane >> 4;
#pragma unroll
  for (int element_pair = 0; element_pair < 4; ++element_pair) {
    const int element = element_pair * 2;
    const int offset =
        (warp_in_pair * 16 + element_pair * 2 + lane_row) * kVtBmBlockN +
        pair_column;
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(score + offset));
    uint32_t words[2];
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(words[0]), "=r"(words[1])
                 : "r"(address)
                 : "memory");
    acc_top.x[element] = __uint_as_float(words[0]);
    acc_top.x[element + 1] = __uint_as_float(words[1]);
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2+4096];"
                 : "=r"(words[0]), "=r"(words[1])
                 : "r"(address)
                 : "memory");
    acc_bottom.x[element] = __uint_as_float(words[0]);
    acc_bottom.x[element + 1] = __uint_as_float(words[1]);
  }
}
__device__ __forceinline__ void vt_bm_sync_warp_pair(int warp_pair) {
  const int barrier_id = warp_pair + 1;
  asm volatile("bar.sync %0, 64;" ::"r"(barrier_id) : "memory");
}
__device__ __forceinline__ float vt_bm_make_probability_row(
    const float* __restrict__ score_row, __half* __restrict__ probability,
    int probability_row, float* __restrict__ row_max, float* __restrict__ row_sum,
    int state_row, int panel, float neg_inf) {
  const int lane = threadIdx.x & 31;
  const float* panel_score = score_row + panel * kVtBmSoftmaxN;
  float thread_max = neg_inf;
  if (lane < kVtBmSoftmaxN / 4) {
    const float4 values = reinterpret_cast<const float4*>(panel_score)[lane];
    thread_max = fmaxf(fmaxf(values.x, values.y), fmaxf(values.z, values.w));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffffu, thread_max, offset));
  }
  const float panel_max = __shfl_sync(0xffffffffu, thread_max, 0);
  const float old_max = row_max[state_row];
  const float new_max = fmaxf(old_max, panel_max);
  const float exp_diff = __expf(old_max - new_max);
  float thread_sum = 0.0f;
  if (lane < kVtBmSoftmaxN / 4) {
    float4 values = reinterpret_cast<const float4*>(panel_score)[lane];
    values.x = __expf(fmaxf(values.x - new_max, -80.0f));
    values.y = __expf(fmaxf(values.y - new_max, -80.0f));
    values.z = __expf(fmaxf(values.z - new_max, -80.0f));
    values.w = __expf(fmaxf(values.w - new_max, -80.0f));
    thread_sum = (values.x + values.y) + (values.z + values.w);
    const int column = lane * 4;
    __half2* first_pair = reinterpret_cast<__half2*>(
        probability + vt_bm_matrix_a_offset(probability_row, column));
    __half2* second_pair = reinterpret_cast<__half2*>(
        probability + vt_bm_matrix_a_offset(probability_row, column + 2));
    *first_pair = __float22half2_rn(make_float2(values.x, values.y));
    *second_pair = __float22half2_rn(make_float2(values.z, values.w));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(0xffffffffu, thread_sum, offset);
  }
  const float panel_sum = __shfl_sync(0xffffffffu, thread_sum, 0);
  if (lane == 0) {
    row_sum[state_row] = exp_diff * row_sum[state_row] + panel_sum;
    row_max[state_row] = new_max;
  }
  return exp_diff;
}
__device__ __forceinline__ float vt_bm_make_split_probability_row(
    const float* __restrict__ score_row, __half* __restrict__ probability,
    int probability_row, float* __restrict__ row_max, float* __restrict__ row_sum,
    int state_row, int panel, float neg_inf, bool has_visible_value) {
  if (!has_visible_value) {
    const int lane = threadIdx.x & 31;
    if (lane < kVtBmSoftmaxN / 4) {
      const int column = lane * 4;
      __half2* first_pair = reinterpret_cast<__half2*>(
          probability + vt_bm_matrix_a_offset(probability_row, column));
      __half2* second_pair = reinterpret_cast<__half2*>(
          probability + vt_bm_matrix_a_offset(probability_row, column + 2));
      const __half2 zero = __float2half2_rn(0.0f);
      *first_pair = zero;
      *second_pair = zero;
    }
    return 1.0f;
  }
  return vt_bm_make_probability_row(score_row, probability, probability_row, row_max,
                                    row_sum, state_row, panel, neg_inf);
}
__device__ __forceinline__ void vt_bm_scale_accumulator_rows(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& accumulator,
    float first_row_scale, float second_row_scale) {
  accumulator.x[0] *= first_row_scale;
  accumulator.x[1] *= first_row_scale;
  accumulator.x[2] *= second_row_scale;
  accumulator.x[3] *= second_row_scale;
  accumulator.x[4] *= first_row_scale;
  accumulator.x[5] *= first_row_scale;
  accumulator.x[6] *= second_row_scale;
  accumulator.x[7] *= second_row_scale;
}
__device__ __forceinline__ void vt_bm_scale_accumulators(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_top,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_bottom,
    const float* __restrict__ row_exp_diff) {
  const int row = vt_bm_accumulator_row(threadIdx.x & 31, 0);
  vt_bm_scale_accumulator_rows(acc_top, row_exp_diff[row], row_exp_diff[row + 2]);
  vt_bm_scale_accumulator_rows(acc_bottom, row_exp_diff[kVtBmPanelM + row],
                               row_exp_diff[kVtBmPanelM + row + 2]);
}
__device__ __forceinline__ void vt_bm_update_pv_panel(
    const __half* __restrict__ probability_top,
    const __half* __restrict__ probability_bottom,
    const __half* __restrict__ value_k0, const __half* __restrict__ value_k16,
    int64_t value_token_stride, int d_offset, int valid_columns,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_top,
    volta::fragment<volta::accumulator, 16, 16, 16, float>& acc_bottom) {
  volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major> a_fragment;
  volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major> b_fragment;
  load_matrix_sync(b_fragment, value_k0 + d_offset, value_token_stride);
  vt_bm_load_matrix_a(a_fragment, probability_top, 0);
  mma_sync(acc_top, a_fragment, b_fragment, acc_top);
  vt_bm_load_matrix_a(a_fragment, probability_bottom, 0);
  mma_sync(acc_bottom, a_fragment, b_fragment, acc_bottom);
  if (valid_columns > 16) {
    load_matrix_sync(b_fragment, value_k16 + d_offset, value_token_stride);
    vt_bm_load_matrix_a(a_fragment, probability_top, 16);
    mma_sync(acc_top, a_fragment, b_fragment, acc_top);
    vt_bm_load_matrix_a(a_fragment, probability_bottom, 16);
    mma_sync(acc_bottom, a_fragment, b_fragment, acc_bottom);
  }
}
__device__ __forceinline__ void vt_bm_store_unnormalized_partial(
    const volta::fragment<volta::accumulator, 16, 16, 16, float>& accumulator,
    float* __restrict__ output, int row_offset, int d_offset, int valid_rows) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    const int row =
        row_offset + vt_bm_accumulator_row(lane, element);
    if (row < valid_rows) {
      const int column = d_offset + vt_bm_accumulator_column(lane, element);
      output[row * kVtBmD + column] = accumulator.x[element];
    }
  }
}

// 512-thread BM32 phase body: causal paged flash over the request's own
// seq_len (shared.actual_n = seq_lens[batch_id]); SPLIT_KV slices the 128-key
// block range into kVtBmSplitParts contiguous bands via blockIdx.y.
template <bool IS_CAUSAL, bool ALL_P, bool PAIR_SCRATCH, bool SPLIT_KV,
          bool CHECK_SPLIT_EMPTY>
__device__ __forceinline__ void vt_bm_phase_body(
    const __half* __restrict__ Q, const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, const int* __restrict__ q_start,
    int H, int M_total, int num_kv_heads, int64_t bt_row,
    int64_t bt_col, int block_size, int64_t k_block_stride,
    int64_t k_token_stride, int64_t k_head_stride, int64_t v_block_stride,
    int64_t v_token_stride, int64_t v_head_stride, float softmax_scale,
    float* __restrict__ split_tmp_out, float* __restrict__ split_tmp_row_max,
    float* __restrict__ split_tmp_row_sum) {
  static_assert(SPLIT_KV && ALL_P && PAIR_SCRATCH,
                "BM32 split-KV requires the all-P pair-scratch body");
  __shared__ __align__(16) VtBmPhaseSharedStorage<kVtBmPanels> shared;

  const int start_row = blockIdx.x * kVtBmBlockM;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  if (tid == 0) {
    shared.batch_id = blockIdx.z / H;
    shared.kv_head_id = (blockIdx.z % H) / (H / num_kv_heads);
    shared.q_len = q_start[shared.batch_id + 1] - q_start[shared.batch_id];
    shared.seq_len = seq_lens[shared.batch_id];
    shared.block_index = 0;
    shared.split_end_block = 0;
  }
  __syncthreads();
  const int M_local = shared.q_len;
  if (start_row >= M_local) return;
  const int valid_q_rows = min(kVtBmBlockM, M_local - start_row);
  const int causal_q_offset = max(shared.seq_len - M_local, 0);

  {
    const __half* q_ptr =
        Q + (static_cast<size_t>(q_start[blockIdx.z / H]) + start_row) * H * kVtBmD +
        static_cast<size_t>(blockIdx.z % H) * kVtBmD;
    if (valid_q_rows == kVtBmBlockM) {
      vt_bm_stage_query(q_ptr, shared.query);
    } else {
      vt_bm_stage_query_partial(q_ptr, shared.query, valid_q_rows);
    }
  }
  if (tid < kVtBmBlockM) {
    shared.row_max[tid] = kVtBmNegInf;
    shared.row_sum[tid] = 0.0f;
  }
  __syncthreads();

  const int total_n_blocks = (shared.seq_len + kVtBmBlockN - 1) / kVtBmBlockN;
  if constexpr (SPLIT_KV) {
    if (tid == 0) {
      shared.block_index =
          static_cast<int>(blockIdx.y) * total_n_blocks / kVtBmSplitParts;
      shared.split_end_block =
          (static_cast<int>(blockIdx.y) + 1) * total_n_blocks / kVtBmSplitParts;
    }
    __syncthreads();
  }

  volta::fragment<volta::accumulator, 16, 16, 16, float> acc_top;
  volta::fragment<volta::accumulator, 16, 16, 16, float> acc_bottom;
  fill_fragment(acc_top, 0.0f);
  fill_fragment(acc_bottom, 0.0f);

  for (;;) {
    if (shared.block_index >= shared.split_end_block) break;
    const int start_col = shared.block_index * kVtBmBlockN;
    const int valid_k_rows = min(kVtBmBlockN, shared.seq_len - start_col);

    if (tid < kVtBmPageSlots) {
      const int token_offset = tid * 16;
      if (token_offset < valid_k_rows) {
        const int global_token_idx = start_col + token_offset;
        const int virtual_block_idx = global_token_idx / block_size;
        const int physical_block =
            block_table[static_cast<int64_t>(shared.batch_id) * bt_row +
                        virtual_block_idx * bt_col];
        const int page_offset = global_token_idx - virtual_block_idx * block_size;
        shared.page_idx[tid] = physical_block;
        shared.page_offset[tid] = page_offset;
        shared.k_tile_ptr[tid] = reinterpret_cast<uint64_t>(
            K_cache + (int64_t)physical_block * k_block_stride +
            (int64_t)page_offset * k_token_stride +
            (int64_t)shared.kv_head_id * k_head_stride);
        shared.v_tile_ptr[tid] = reinterpret_cast<uint64_t>(
            V_cache + (int64_t)physical_block * v_block_stride +
            (int64_t)page_offset * v_token_stride +
            (int64_t)shared.kv_head_id * v_head_stride);
      } else {
        shared.page_idx[tid] = -1;
        shared.page_offset[tid] = 0;
        shared.k_tile_ptr[tid] = 0;
        shared.v_tile_ptr[tid] = 0;
      }
    }
    __syncthreads();

    if (warp_id < kVtBmBlockN / 16 && warp_id * 16 < valid_k_rows) {
      const int tile_n = warp_id * 16;
      const __half* k_tile =
          reinterpret_cast<const __half*>(shared.k_tile_ptr[warp_id]);
      if constexpr (PAIR_SCRATCH) {
        vt_bm_spill_pair_scratch(shared.score, acc_top, acc_bottom);
      } else {
        store_matrix_sync(shared.score + tile_n, acc_top,
                          kVtBmBlockN, volta::mem_row_major);
        store_matrix_sync(
            shared.score + kVtBmPanelM * kVtBmBlockN + tile_n, acc_bottom,
            kVtBmBlockN, volta::mem_row_major);
      }
      asm volatile("" ::: "memory");

      volta::fragment<volta::accumulator, 16, 16, 16, float> qk_top;
      volta::fragment<volta::accumulator, 16, 16, 16, float> qk_bottom;
      {
        volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major> a_fragment;
        volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major> b_fragment;
        fill_fragment(qk_top, 0.0f);
        fill_fragment(qk_bottom, 0.0f);
#pragma unroll
        for (int k_offset = 0; k_offset < kVtBmD; k_offset += 16) {
          load_matrix_sync(b_fragment, k_tile + k_offset, k_token_stride);
          vt_bm_load_matrix_a(a_fragment, shared.query, k_offset);
          mma_sync(qk_top, a_fragment, b_fragment, qk_top);
          vt_bm_load_matrix_a(a_fragment,
                              shared.query + kVtBmPanelM * kVtBmD, k_offset);
          mma_sync(qk_bottom, a_fragment, b_fragment, qk_bottom);
        }
      }
      asm volatile("" ::: "memory");

      if constexpr (PAIR_SCRATCH) {
        vt_bm_reload_pair_scratch(shared.score, acc_top, acc_bottom);
        const int partner_warp = warp_id ^ 1;
        if (partner_warp * 16 < valid_k_rows) {
          vt_bm_sync_warp_pair(warp_id >> 1);
        }
      } else {
        load_matrix_sync(acc_top, shared.score + tile_n,
                         kVtBmBlockN, volta::mem_row_major);
        load_matrix_sync(acc_bottom,
                         shared.score + kVtBmPanelM * kVtBmBlockN + tile_n,
                         kVtBmBlockN, volta::mem_row_major);
      }

      store_matrix_sync(shared.score + tile_n, qk_top,
                        kVtBmBlockN, volta::mem_row_major);
      store_matrix_sync(
          shared.score + kVtBmPanelM * kVtBmBlockN + tile_n, qk_bottom,
          kVtBmBlockN, volta::mem_row_major);
    }
    __syncthreads();

    // mask + softmax scale
    for (int index = tid;
         index < kVtBmBlockM * kVtBmBlockN;
         index += kVtBmThreads) {
      const int row = index / kVtBmBlockN;
      const int column = index - row * kVtBmBlockN;
      const int global_m = start_row + row;
      const int global_n = start_col + column;
      const bool is_valid =
          global_m < M_local && global_n < start_col + valid_k_rows;
      if (is_valid && global_n <= global_m + causal_q_offset) {
        shared.score[index] *= softmax_scale;
      } else {
        shared.score[index] = kVtBmNegInf;
      }
    }
    __syncthreads();

#pragma unroll
    for (int panel = 0; panel < kVtBmPanels; ++panel) {
      const int panel_start = panel * kVtBmSoftmaxN;
      const int valid_panel_columns = min(kVtBmSoftmaxN, valid_k_rows - panel_start);
      if (valid_panel_columns > 0) {
        __half* probability_top =
            shared.probability_top + panel * kVtBmProbElems;
        __half* probability_bottom =
            shared.probability_bottom + panel * kVtBmProbElems;
        float* row_exp_diff = shared.row_exp_diff + panel * kVtBmBlockM;
        const int first_global_n = start_col + panel_start;
        const bool top_has_visible_value =
            warp_id < valid_q_rows &&
            (!IS_CAUSAL || first_global_n <= start_row + warp_id + causal_q_offset);
        const bool bottom_has_visible_value =
            kVtBmPanelM + warp_id < valid_q_rows &&
            (!IS_CAUSAL ||
             first_global_n <= start_row + kVtBmPanelM + warp_id + causal_q_offset);
        float top_exp_diff;
        float bottom_exp_diff;
        if constexpr (CHECK_SPLIT_EMPTY) {
          top_exp_diff =
              vt_bm_make_split_probability_row(
                  shared.score + warp_id * kVtBmBlockN, probability_top,
                  warp_id, shared.row_max, shared.row_sum, warp_id, panel,
                  kVtBmNegInf, top_has_visible_value);
          bottom_exp_diff = vt_bm_make_split_probability_row(
              shared.score + (kVtBmPanelM + warp_id) * kVtBmBlockN,
              probability_bottom, warp_id, shared.row_max, shared.row_sum,
              kVtBmPanelM + warp_id, panel, kVtBmNegInf,
              bottom_has_visible_value);
        }
        if (lane_id == 0) {
          row_exp_diff[warp_id] = top_exp_diff;
          row_exp_diff[kVtBmPanelM + warp_id] = bottom_exp_diff;
        }
      }
      __syncwarp();
    }
    __syncthreads();

#pragma unroll
    for (int panel = 0; panel < kVtBmPanels; ++panel) {
      const int panel_start = panel * kVtBmSoftmaxN;
      const int valid_panel_columns = min(kVtBmSoftmaxN, valid_k_rows - panel_start);
      if (valid_panel_columns > 0) {
        const __half* probability_top =
            shared.probability_top + panel * kVtBmProbElems;
        const __half* probability_bottom =
            shared.probability_bottom + panel * kVtBmProbElems;
        const float* row_exp_diff = shared.row_exp_diff + panel * kVtBmBlockM;
        vt_bm_scale_accumulators(acc_top, acc_bottom, row_exp_diff);
        const int page_slot_k0 = panel_start >> 4;
        const __half* value_k0 = reinterpret_cast<const __half*>(
            shared.v_tile_ptr[page_slot_k0]);
        const __half* value_k16 = nullptr;
        if (valid_panel_columns > 16) {
          const int page_slot_k16 = (panel_start + 16) >> 4;
          value_k16 = reinterpret_cast<const __half*>(
              shared.v_tile_ptr[page_slot_k16]);
        }
        vt_bm_update_pv_panel(probability_top, probability_bottom, value_k0,
                              value_k16, v_token_stride, warp_id * 16,
                              valid_panel_columns, acc_top, acc_bottom);
      }
    }
    __syncthreads();

    if (tid == 0) ++shared.block_index;
    __syncthreads();
  }

  {
    const size_t q_head_linear = static_cast<size_t>(blockIdx.z);
    const size_t row_base =
        static_cast<size_t>(q_start[shared.batch_id]) + start_row;
    const size_t partial_row_base =
        (q_head_linear * kVtBmSplitParts + static_cast<size_t>(blockIdx.y)) *
            static_cast<size_t>(M_total) +
        row_base;
    if (tid < kVtBmBlockM) {
      split_tmp_row_max[partial_row_base + tid] = shared.row_max[tid];
      split_tmp_row_sum[partial_row_base + tid] = shared.row_sum[tid];
    }
    float* partial_out = split_tmp_out + partial_row_base * kVtBmD;
    vt_bm_store_unnormalized_partial(acc_top, partial_out, 0, warp_id * 16,
                                     valid_q_rows);
    vt_bm_store_unnormalized_partial(acc_bottom, partial_out, kVtBmPanelM,
                                     warp_id * 16, valid_q_rows);
  }
}

// Partial kernel: 3-way KV split, unnormalized f32 partials + per-part stats.
template <bool CHECK_SPLIT_EMPTY>
__global__ void __launch_bounds__(kVtBmThreads, 2)
    VtBmPrefillSplitkv3PartialKernel(
        const __half* __restrict__ Q, const __half* __restrict__ K_cache,
        const __half* __restrict__ V_cache, const int* __restrict__ block_table,
        const int* __restrict__ seq_lens, const int* __restrict__ q_start,
        int H, int M_total, int num_kv_heads, int block_size,
        int64_t bt_row, int64_t bt_col, int64_t k_block_stride,
        int64_t k_token_stride, int64_t k_head_stride, int64_t v_block_stride,
        int64_t v_token_stride, int64_t v_head_stride, float softmax_scale,
        float* __restrict__ split_tmp_out, float* __restrict__ split_tmp_row_max,
        float* __restrict__ split_tmp_row_sum) {
  vt_bm_phase_body<true, true, true, true, CHECK_SPLIT_EMPTY>(
      Q, K_cache, V_cache, block_table, seq_lens, q_start,
      H, M_total, num_kv_heads, bt_row, bt_col, block_size,
      k_block_stride, k_token_stride, k_head_stride, v_block_stride,
      v_token_stride, v_head_stride, softmax_scale, split_tmp_out,
      split_tmp_row_max, split_tmp_row_sum);
}

// Merge the 3 split parts: one 256-thread block per 32 rows, 8 threads per
// output row (threadIdx>>3 = row, &7 = lane-in-row across d). The per-part f32
// partials combine weight_0*O0 + weight_1*O1 + weight_2*O2 (each part's own
// max/sum renormalized to the merged max), then normalize by the denominator —
// the same f32 order as the sequential per-row flash, so long/short prefill
// agree up to the seq boundary.
__global__ void __launch_bounds__(256) VtBmPrefillMergeKernel(
    const float* __restrict__ split_tmp_out,
    const float* __restrict__ split_tmp_row_max,
    const float* __restrict__ split_tmp_row_sum,
    float* __restrict__ Out, const int* __restrict__ q_start,
    int H, int num_reqs, int M_total) {
  const int start_row_g = blockIdx.x * kVtBmBlockM;
  const int row_local = threadIdx.x >> 3;
  const int lane_in_row = threadIdx.x & 7;
  const int row_g = start_row_g + row_local;
  const int batch_id = blockIdx.z / H;
  const int head_id = blockIdx.z % H;
  if (batch_id >= num_reqs) return;
  const int q_start_b = q_start[batch_id];
  const int q_len = q_start[batch_id + 1] - q_start[batch_id];
  const int row = q_start_b + row_g;
  if (row_g >= q_len || row >= M_total) return;

  const size_t q_head_linear = static_cast<size_t>(blockIdx.z);
  const size_t first_state =
      (q_head_linear * kVtBmSplitParts) * M_total + row;
  const float max_0 = split_tmp_row_max[first_state];
  const float max_1 = split_tmp_row_max[first_state + M_total];
  const float max_2 = split_tmp_row_max[first_state + 2 * M_total];
  const float sum_0 = split_tmp_row_sum[first_state];
  const float sum_1 = split_tmp_row_sum[first_state + M_total];
  const float sum_2 = split_tmp_row_sum[first_state + 2 * M_total];
  const float merged_max = fmaxf(max_0, fmaxf(max_1, max_2));
  const float weight_0 =
      sum_0 > 0.0f ? __expf(fmaxf(max_0 - merged_max, -80.0f)) : 0.0f;
  const float weight_1 =
      sum_1 > 0.0f ? __expf(fmaxf(max_1 - merged_max, -80.0f)) : 0.0f;
  const float weight_2 =
      sum_2 > 0.0f ? __expf(fmaxf(max_2 - merged_max, -80.0f)) : 0.0f;
  const float denominator =
      weight_0 * sum_0 + weight_1 * sum_1 + weight_2 * sum_2;
  const float inverse_denominator = 1.0f / fmaxf(denominator, 1e-24f);
  const size_t output_base =
      (static_cast<size_t>(row) * H + head_id) * kVtBmD;
  const size_t partial_base_0 = first_state * kVtBmD;
  const size_t partial_base_1 = (first_state + M_total) * kVtBmD;
  const size_t partial_base_2 = (first_state + 2 * M_total) * kVtBmD;
#pragma unroll
  for (int column = lane_in_row; column < kVtBmD; column += 8) {
    const float numerator =
        weight_0 * split_tmp_out[partial_base_0 + column] +
        weight_1 * split_tmp_out[partial_base_1 + column] +
        weight_2 * split_tmp_out[partial_base_2 + column];
    Out[output_base + column] = numerator * inverse_denominator;
  }
}

extern "C" void vt_sm70_fa2_prefill(
    float* out, const void* query, const void* k, const void* v,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* q_start,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d,
    int64_t block_size, int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
    int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale,
    cudaStream_t stream) {
  const int td = (int)d;
  if (td != 64 && td != 128 && td != 192 && td != 256) return;
  int64_t bm_route = 0;
  // D==256 whole-batch split: BM32 splitkv3 when there are enough query rows
  // overall and the paged-KV block size is a multiple of the 16-token slot.
  if (td == 256 && num_reqs > 0 && bt_col > 0 && (block_size % 16) == 0) {
    int32_t mtot = 0;
    if (cudaMemcpy(&mtot, q_start + num_reqs, sizeof(int32_t),
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
        mtot >= kVtBmMinRows) {
      bm_route = 1;
    }
  }
  const bool bm = bm_route != 0;
  if (bm) {
    // Lazy per-process BM workspace: split partials/stats [B, H, 3, M, D].
    struct BmWorkspace {
      float* partial = nullptr;   // [B*H*3*M_total*256]
      float* max = nullptr;       // [B*H*3*M_total]
      float* sum = nullptr;       // [B*H*3*M_total]
      size_t bytes = 0;
      ~BmWorkspace() {
        cudaFree(partial);
        cudaFree(max);
        cudaFree(sum);
      }
    };
    static BmWorkspace bws;
    const int32_t mtot = []() {
      int32_t m = 0;
      cudaMemcpy(&m, q_start + num_reqs, sizeof(int32_t), cudaMemcpyDeviceToHost);
      return m;
    }();
    const size_t want =
        (size_t)num_reqs * (size_t)hq * 3u * (size_t)mtot * (256 + 2) * sizeof(float);
    if (bws.bytes < want) {
      cudaFree(bws.partial);
      cudaFree(bws.max);
      cudaFree(bws.sum);
      bws.partial = nullptr;
      bws.max = nullptr;
      bws.sum = nullptr;
      const size_t pc = (size_t)num_reqs * (size_t)hq * 3u * (size_t)mtot * 256 * sizeof(float);
      const size_t sc = (size_t)num_reqs * (size_t)hq * 3u * (size_t)mtot * sizeof(float);
      cudaError_t e = cudaMalloc(&bws.partial, pc);
      if (e == cudaSuccess) e = cudaMalloc(&bws.max, sc);
      if (e == cudaSuccess) e = cudaMalloc(&bws.sum, sc);
      if (e == cudaSuccess) bws.bytes = want;
    }
    if (bws.bytes >= want) {
      const dim3 pg((unsigned)((mtot + kVtBmBlockM - 1) / kVtBmBlockM), 3u,
                    (unsigned)(num_reqs * hq));
      const dim3 mg((unsigned)((mtot + kVtBmBlockM - 1) / kVtBmBlockM), 1u,
                    (unsigned)(num_reqs * hq));
      const __half* qq = static_cast<const __half*>(query);
      const __half* kk = static_cast<const __half*>(k);
      const __half* vv = static_cast<const __half*>(v);
      VtBmPrefillSplitkv3PartialKernel<true><<<pg, kVtBmThreads, 0, stream>>>(
          qq, kk, vv, block_table, seq_lens, q_start, (int)hq, mtot,
          (int)num_kv, (int)block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,
          vc_blk, vc_pg, vc_hd, scale, bws.partial, bws.max, bws.sum);
      VtBmPrefillMergeKernel<<<mg, 256, 0, stream>>>(
          bws.partial, bws.max, bws.sum, out, q_start, (int)hq, (int)num_reqs,
          mtot);
      return;
    }
    // Allocation failed: fall through to the sequential kernel for every
    // request (bn stays routed off).
    bm_route = 0;
  }
  const dim3 grid((unsigned)num_reqs, (unsigned)hq);
#define VT_PREFILL_LAUNCH(TD)                                                          \
  do {                                                                                 \
    const int smem = (3 * 16 * (TD) /*sq,sk,sv*/ + 2 * 16 * 16 /*sb,sa16*/) * 2 +      \
                     (16 * 16 + 16 * (TD) /*ss,acc*/ + 32) * 4;                        \
    Sm70FaPrefillAttn<(TD)><<<grid, kF2Threads, smem, stream>>>(                       \
        out, static_cast<const __half*>(query), static_cast<const __half*>(k),         \
        static_cast<const __half*>(v), block_table, seq_lens, q_start,               \
        num_reqs, hq, num_kv, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,      \
        vc_blk, vc_pg, vc_hd, scale, bm_route);                                          \
  } while (0)
  if (td == 64) VT_PREFILL_LAUNCH(64);
  else if (td == 128) VT_PREFILL_LAUNCH(128);
  else if (td == 192) VT_PREFILL_LAUNCH(192);
  else                VT_PREFILL_LAUNCH(256);
#undef VT_PREFILL_LAUNCH
}

// ---- host entries (extern "C", raw pointers, no torch) ----

// Decode D==256 parallel workspace: lazily grown per-process cache of the
// partition partials + stats. Sized from host-known shape bounds only
// (bt_row/bt_col*block_size caps the per-request seq length), so no device
// reads ever happen in the fast path. If a (re)allocation would occur while
// the launch stream is capturing (cudaMalloc is illegal under capture), the
// call degrades to the sequential kernel for the whole batch — capture-safe.
namespace {
struct DecodeXqWorkspace {
  float* tmp_out = nullptr;    // [B, hq, Pmax, 256]
  float* max_logits = nullptr;  // [B, hq, Pmax]
  float* exp_sums = nullptr;    // [B, hq, Pmax]
  size_t bytes = 0;
  ~DecodeXqWorkspace() {
    cudaFree(tmp);
    cudaFree(max_logits);
    cudaFree(exp_sums);
  }
};
DecodeXqWorkspace& DecodeXqBuf() {
  static DecodeXqWorkspace b;
  return b;
}

// p256 and p1024 partition kernels for one q_per_kv group; G6 additionally
// gets the QK software-pipeline variant (dual-CTA carveout).
template <int GROUP>
struct VtXqGroupDispatcher {
  static void Launch(const dim3& pgrid, cudaStream_t stream, const __half* q,
                     const __half* k, const __half* v, float* tmp,
                     float* mlog, float* esum, const int32_t* bt,
                     const int32_t* sl, int64_t num_reqs, int np,
                     int hq, int nkv, int bs, int64_t bt_row, int64_t bt_col,
                     int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
                     int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale) {
    const int64_t ts0 = hq * 256, ts1 = 256;
    const int64_t ss0 = (int64_t)hq * np, ss1 = np;
    if constexpr (GROUP == 6) {
      const size_t pipe_smem = sizeof(VtXq256QkPipelineSmemLayout);
      auto pk = (void*)VtXqDecodePartitionKernel<256, 6, true, kVtXq256Threads,
                                                 2, 0, true>;
      cudaFuncSetAttribute(pk, cudaFuncAttributeMaxDynamicSharedMemorySize, pipe_smem);
      VtXqDecodePartitionKernel<256, 6, true, kVtXq256Threads, 2, 0, true>
          <<<pgrid, kVtXq256Threads, pipe_smem, stream>>>(
              q, k, v, tmp, mlog, esum, bt, sl, (int)num_reqs, np, hq, nkv, bs,
              hq * 256, 256, ts0, ts1, 256, ss0, ss1, bt_row, bt_col, kc_blk,
              kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, scale);
      VtXqDecodePartitionKernel<1024, 6, true, kVtXq256Threads, 2, 0, true>
          <<<pgrid, kVtXq256Threads, pipe_smem, stream>>>(
              q, k, v, tmp, mlog, esum, bt, sl, (int)num_reqs, np, hq, nkv, bs,
              hq * 256, 256, ts0, ts1, 256, ss0, ss1, bt_row, bt_col, kc_blk,
              kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, scale);
    } else {
      const size_t smem = sizeof(VtXq256SmemLayout);
      auto p256k = (void*)VtXqDecodePartitionKernel<256, GROUP, true, kVtXq256Threads,
                                                    1, 0, false>;
      cudaFuncSetAttribute(p256k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      VtXqDecodePartitionKernel<256, GROUP, true, kVtXq256Threads, 1, 0, false>
          <<<pgrid, kVtXq256Threads, smem, stream>>>(
              q, k, v, tmp, mlog, esum, bt, sl, (int)num_reqs, np, hq, nkv, bs,
              hq * 256, 256, ts0, ts1, 256, ss0, ss1, bt_row, bt_col, kc_blk,
              kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, scale);
      VtXqDecodePartitionKernel<1024, GROUP, true, kVtXq256Threads, 1, 0, false>
          <<<pgrid, kVtXq256Threads, smem, stream>>>(
              q, k, v, tmp, mlog, esum, bt, sl, (int)num_reqs, np, hq, nkv, bs,
              hq * 256, 256, ts0, ts1, 256, ss0, ss1, bt_row, bt_col, kc_blk,
              kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, scale);
    }
  }
};
}  // namespace

extern "C" void vt_sm70_fa2_decode(
    float* out, const void* query, const void* k, const void* v,
    const int32_t* block_table, const int32_t* seq_lens,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, cudaStream_t stream) {
  const int td = (int)d;
  if (td != 64 && td != 128 && td != 192 && td != 256) return;  // caller-side gate
  bool parallel = false;
  int group = 0;
  if (td == 256 && num_kv > 0 && hq % num_kv == 0 && bt_col > 0 &&
      (block_size % 16) == 0) {
    const int qpk = (int)(hq / num_kv);
    if (qpk == 4 || qpk == 6 || qpk == 8) {
      const int64_t max_seq = (bt_row / bt_col) * block_size;
      const int64_t p = (max_seq + 255) / 256;
      const size_t bytes =
          (size_t)num_reqs * (size_t)hq * (size_t)p * (256 + 2) * sizeof(float);
      cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
      const bool capturing =
          cudaStreamIsCapturing(stream, &cap) == cudaSuccess &&
          cap != cudaStreamCaptureStatusNone;
      DecodeXqWorkspace& ws = DecodeXqBuf();
      const bool ready = [&]() -> bool {
        if (capturing) return ws.bytes >= bytes;
        if (ws.bytes >= bytes) return true;
        cudaFree(ws.tmp);
        cudaFree(ws.max_logits);
        cudaFree(ws.exp_sums);
        ws.tmp = nullptr;
        ws.max_logits = nullptr;
        ws.exp_sums = nullptr;
        const size_t tb =
            (size_t)num_reqs * (size_t)hq * (size_t)p * 256 * sizeof(float);
        const size_t sb =
            (size_t)num_reqs * (size_t)hq * (size_t)p * sizeof(float);
        cudaError_t e = cudaMalloc(&ws.tmp, tb);
        if (e == cudaSuccess) e = cudaMalloc(&ws.max_logits, sb);
        if (e == cudaSuccess) e = cudaMalloc(&ws.exp_sums, sb);
        if (e != cudaSuccess) {
          cudaFree(ws.tmp);
          ws.tmp = nullptr;
          cudaFree(ws.max_logits);
          ws.max_logits = nullptr;
          cudaFree(ws.exp_sums);
          ws.exp_sums = nullptr;
          return false;
        }
        ws.bytes = bytes;
        return true;
      }();
      if (ready) {
        parallel = true;
        group = qpk;
      }
    }
  }
  if (parallel) {
    // Route split: the sequential 1-warp kernel keeps every request with
    // seq_len < kVtXqSeqRouteBegin (byte-identical to the unported path for
    // those shapes); the XQA partition pair + split reduce covers the rest.
    {
      const dim3 grid((unsigned)num_reqs, (unsigned)hq);
      const int smem = (2 * 16 * 256 + 16 * 16) * 2 + (16 * 16 + 256 + 32) * 4;
      Sm70Fa2DecodeAttn<256><<<grid, kF2Threads, smem, stream>>>(
          out, static_cast<const __half*>(query), static_cast<const __half*>(k),
          static_cast<const __half*>(v), block_table, seq_lens, nullptr,
          num_reqs, hq, num_kv, 256, block_size, bt_row, bt_col, kc_blk, kc_pg,
          kc_hd, vc_blk, vc_pg, vc_hd, scale, 0.f, kVtXqSeqRouteBegin);
    }
    const int np = (int)(((bt_row / bt_col) * block_size + 255) / 256);
    const dim3 pgrid((unsigned)num_reqs, (unsigned)num_kv, (unsigned)np);
    const dim3 sgrid((unsigned)num_reqs, (unsigned)hq);
    const dim3 ogrid((unsigned)num_reqs, (unsigned)hq, 256 / 8);
    DecodeXqWorkspace& ws = DecodeXqBuf();
    const __half* q = static_cast<const __half*>(query);
    const __half* kc = static_cast<const __half*>(k);
    const __half* vc = static_cast<const __half*>(v);
    const int64_t ts0 = hq * 256, ts1 = 256;
    const int64_t ss0 = (int64_t)hq * np, ss1 = np;
    if (group == 4)
      VtXqGroupDispatcher<4>::Launch(
          pgrid, stream, q, kc, vc, ws.tmp, ws.max_logits, ws.exp_sums,
          block_table, seq_lens, num_reqs, np, (int)hq, (int)num_kv,
          (int)block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg,
          vc_hd, scale);
    else if (group == 8)
      VtXqGroupDispatcher<8>::Launch(
          pgrid, stream, q, kc, vc, ws.tmp, ws.max_logits, ws.exp_sums,
          block_table, seq_lens, num_reqs, np, (int)hq, (int)num_kv,
          (int)block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg,
          vc_hd, scale);
    else
      VtXqGroupDispatcher<6>::Launch(
          pgrid, stream, q, kc, vc, ws.tmp, ws.max_logits, ws.exp_sums,
          block_table, seq_lens, num_reqs, np, (int)hq, (int)num_kv,
          (int)block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg,
          vc_hd, scale);
    VtXqDecodeReduceStatsKernel<<<sgrid, 256, np * sizeof(float), stream>>>(
        ws.max_logits, ws.exp_sums, seq_lens, (int)num_reqs, np, (int)hq, ss0,
        ss1);
    VtXqDecodeReduceOutputKernel<8><<<ogrid, 8, 0, stream>>>(
        ws.tmp, ws.max_logits, ws.exp_sums, seq_lens, out, (int)num_reqs, np,
        (int)hq, ts0, ts1, 256, ss0, ss1, hq * 256, 256);
    return;
  }
  // Sequential (retained shapes + non-eligible d==256): body byte-identical
  // for every request the guard does not route away (d != 256 always passes).
  const dim3 grid((unsigned)num_reqs, (unsigned)hq);
#define VT_DECODE_LAUNCH(TD, ROUTE)                                                  \
  do {                                                                              \
    const int smem = (2 * 16 * (TD) /*sq,sk*/ + 16 * 16 /*sb*/) * 2 +              \
                     (16 * 16 + TD + 32) /*sc,sacc,pad*/ * 4;                     \
    Sm70Fa2DecodeAttn<(TD)><<<grid, kF2Threads, smem, stream>>>(                    \
        out, static_cast<const __half*>(query), static_cast<const __half*>(k),      \
        static_cast<const __half*>(v), block_table, seq_lens, nullptr,              \
        num_reqs, hq, num_kv, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,  \
        vc_blk, vc_pg, vc_hd, scale, 0.f, ROUTE);                                    \
  } while (0)
  if (td == 64) VT_DECODE_LAUNCH(64, kVtXqSeqRouteNever);
  else if (td == 128) VT_DECODE_LAUNCH(128, kVtXqSeqRouteNever);
  else if (td == 192) VT_DECODE_LAUNCH(192, kVtXqSeqRouteNever);
  else VT_DECODE_LAUNCH(256, kVtXqSeqRouteNever);
#undef VT_DECODE_LAUNCH
}

// Oracle (defined in cuda_paged_attn.cu as extern "C"): reference decode path
// over the SAME paged tensors.
extern "C" void vt_sm70_fa2_ref_decode(
    float* out, const void* q, const void* k, const void* v,
    const int32_t* block_table, const int32_t* seq_lens,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, cudaStream_t stream);

// Device-side numeric self-check: run my kernel and the reference
// PagedAttentionKernel (via the extern-C oracle) on the SAME paged tensors,
// diff in this nvcc TU's host code. Returns 0 on parity within tolerance, 1 on
// materialized mismatch, -1 on launch/alloc error. Deterministic inputs.
extern "C" int vt_sm70_fa2_self_check(float tol_rel, int verbose) {
  // Two configs: a single query head (hq=1, kv=1, g=0 only) and a GQA config
  // (hq=8, num_kv=2 -> the group index g = h/(hq/num_kv) takes 0 and 1, so the
  // per-group k/v offsets g*kc/vc_hd get exercised on-device against the
  // oracle). Device-synced before the host diff so a kernel-mode fault reports
  // as a run error, not a stale-buffer comparison.
  struct Cfg { int64_t nr, hq, nkv, d; int64_t seqs[8]; };
  const Cfg cfgs[5] = {
      {3, 1, 1, 64, {20, 17, 33, 0, 0, 0, 0, 0}},
      {2, 8, 2, 64, {9, 25, 0, 0, 0, 0, 0, 0}},
      {2, 4, 1, 128, {25, 17, 0, 0, 0, 0, 0, 0}},  // D=128 (Llama/Mistral width)
      {2, 2, 1, 192, {19, 23, 0, 0, 0, 0, 0, 0}},  // D=192 (wider head)
      {3, 1, 1, 256, {21, 13, 29, 0, 0, 0, 0, 0}},  // D=256 (wider head)
  };
  double worst = 0.0;
  const int ncfg = 5;
  for (int ci = 0; ci < ncfg; ++ci) {
    const Cfg& c = cfgs[ci];
    const int64_t d = c.d, bn = 16;
    int64_t maxblocks = 1;
    for (int64_t r = 0; r < c.nr; ++r) {
      const int64_t nb = (c.seqs[r] + bn - 1) / bn;
      if (nb > maxblocks) maxblocks = nb;
    }
    const int64_t pages = c.nr * maxblocks;
    const int64_t kc_hd = d, kc_pg = c.nkv * d, kc_blk = bn * c.nkv * d;
    const int64_t vc_hd = d, vc_pg = c.nkv * d, vc_blk = bn * c.nkv * d;
    const float scale = 0.125f;  // ~ 1/sqrt(64)

    const size_t kv_elems = (size_t)pages * kc_blk;
    std::vector<unsigned short> hk16(kv_elems), hv16(kv_elems);
    std::vector<unsigned short> hq16((size_t)c.nr * c.hq * d);
    // Fractional, non-half-representable values so the fp16-WMMA vs fp32-fma
    // rounding is actually exercised (exact small-half inputs made the two
    // paths bit-identical, hiding a tolerance regression behind 0.000e+00).
    auto f2h = [](float f) -> unsigned short { return __half_as_ushort(__float2half_rn(f)); };
    for (size_t i = 0; i < kv_elems; ++i) {
      hk16[i] = f2h(0.37f * (float)((i * 13 + 7) % 19) + 0.11f);   // non-exact
      hv16[i] = f2h(-0.23f * (float)((i * 7 + 3) % 13) - 0.07f);
    }
    for (size_t i = 0; i < hq16.size(); ++i)
      hq16[i] = f2h(0.61f * (float)((i * 3 + 1) % 11) + 0.31f);

    __half *q_d = nullptr, *k_d = nullptr, *v_d = nullptr;
    int32_t *bt_d = nullptr, *sl_d = nullptr;
    float *mine = nullptr, *ref = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&q_d, (size_t)c.nr * c.hq * d * sizeof(__half));
    err = (err ?: cudaMalloc((void**)&k_d, kv_elems * sizeof(__half)));
    err = (err ?: cudaMalloc((void**)&v_d, kv_elems * sizeof(__half)));
    err = (err ?: cudaMalloc((void**)&bt_d, (size_t)c.nr * maxblocks * sizeof(int32_t)));
    err = (err ?: cudaMalloc((void**)&sl_d, (size_t)c.nr * sizeof(int32_t)));
    err = (err ?: cudaMalloc((void**)&mine, (size_t)c.nr * c.hq * d * sizeof(float)));
    err = (err ?: cudaMalloc((void**)&ref, (size_t)c.nr * c.hq * d * sizeof(float)));
    if (err != cudaSuccess) {
      fprintf(stderr, "sm70 fa2 self-check cfg%d: alloc: %s\n", ci, cudaGetErrorString(err));
      cudaFree((void*)q_d); cudaFree((void*)k_d); cudaFree((void*)v_d);
      cudaFree((void*)bt_d); cudaFree((void*)sl_d); cudaFree((void*)mine); cudaFree((void*)ref);
      return -1;
    }
    cudaMemcpy(q_d, hq16.data(), (size_t)c.nr * c.hq * d * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, hk16.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, hv16.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);

    std::vector<int32_t> hbt((size_t)c.nr * maxblocks);
    for (int64_t r = 0; r < c.nr; ++r)
      for (int64_t b = 0; b < maxblocks; ++b)
        hbt[(size_t)r * maxblocks + b] = (int32_t)(r * maxblocks + b);
    std::vector<int32_t> hsl(c.seqs, c.seqs + c.nr);
    cudaMemcpy(bt_d, hbt.data(), (size_t)c.nr * maxblocks * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(sl_d, hsl.data(), (size_t)c.nr * sizeof(int32_t), cudaMemcpyHostToDevice);

    const int32_t bt_col = 1;
    cudaStream_t st = 0;
    vt_sm70_fa2_decode(mine, q_d, k_d, v_d, bt_d, sl_d, c.nr, c.hq, c.nkv, d, bn,
                       maxblocks, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd,
                       scale, st);
    vt_sm70_fa2_ref_decode(ref, q_d, k_d, v_d, bt_d, sl_d, c.nr, c.hq, c.nkv, d, bn,
                           maxblocks, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg,
                           vc_hd, scale, st);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "sm70 fa2 self-check cfg%d: launch/sync: %s\n", ci, cudaGetErrorString(err));
      cudaFree((void*)q_d); cudaFree((void*)k_d); cudaFree((void*)v_d);
      cudaFree((void*)bt_d); cudaFree((void*)sl_d); cudaFree((void*)mine); cudaFree((void*)ref);
      return -1;
    }

    std::vector<float> hmine((size_t)c.nr * c.hq * d), href((size_t)c.nr * c.hq * d);
    cudaMemcpy(hmine.data(), mine, (size_t)c.nr * c.hq * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(href.data(), ref, (size_t)c.nr * c.hq * d * sizeof(float), cudaMemcpyDeviceToHost);

    double max_reldev = 0.0;
    for (size_t i = 0; i < hmine.size(); ++i) {
      const double denom = href[i] > 1.0 ? href[i] : 1.0;
      const double rel = std::fabs((double)hmine[i] - href[i]) / denom;
      if (rel > max_reldev) max_reldev = rel;
    }
    if (worst < max_reldev) worst = max_reldev;
    if (verbose)
      fprintf(stdout, "  cfg%d (reqs=%lld hq=%lld kv=%lld): max reldev %.3e\n",
              ci, (long long)c.nr, (long long)c.hq, (long long)c.nkv, max_reldev);

    cudaFree((void*)q_d); cudaFree((void*)k_d); cudaFree((void*)v_d);
    cudaFree((void*)bt_d); cudaFree((void*)sl_d); cudaFree((void*)mine); cudaFree((void*)ref);
  }
  if (verbose)
    fprintf(stdout, "sm70 fa2 self-check: worst max rel dev = %.3e (tol %.2e)\n",
            worst, (double)tol_rel);
  return (worst <= (double)tol_rel) ? 0 : 1;
}


// ---- prefill CPU-oracle parity driver (brick G) ----
// Builds a small 2-request paged prefill, runs vt_sm70_fa2_prefill, and
// compares every query row/head against a plain fp32 CPU flash (online
// softmax over the causal key set). Returns 0 ok / 1 mismatch / 2 not sm_70.
extern "C" int vt_sm70_fa2_prefill_self_check(float tol_rel, int verbose) {
  const DeviceCaps& caps = GetDeviceCaps();
  if (!caps.valid || caps.sm_major != 7 || caps.sm_minor != 0) return 2;
  const int64_t num_reqs = 2, hq = 4, nkv = 2, d = 64, bs = 16, maxblocks = 4;
  const int q1[2] = {40, 12}, s1[2] = {56, 20};  // GQA + multi-slice + tails
  const int64_t pages = num_reqs * maxblocks;
  const size_t kv = (size_t)pages * (bs * nkv * d);
  const int64_t kc_hd = d, kc_pg = nkv * d, kc_blk = bs * nkv * d;
  const int64_t vc_hd = d, vc_pg = nkv * d, vc_blk = bs * nkv * d;
  const size_t qn = (size_t)(q1[0] + q1[1]);

  auto f2h = [](float f) -> __half { return __float2half_rn(f); };
  std::vector<__half> hq_(qn * hq * d), hk(kv), hv(kv);
  for (size_t i = 0; i < hq_.size(); ++i) hq_[i] = f2h(0.61f * (float)((i * 3 + 1) % 11) + 0.31f);
  for (size_t i = 0; i < kv; ++i) {
    hk[i] = f2h(0.37f * (float)((i * 13 + 7) % 19) + 0.11f);
    hv[i] = f2h(-0.23f * (float)((i * 7 + 3) % 13) - 0.07f);
  }
  std::vector<int32_t> hbt((size_t)num_reqs * maxblocks), hsl{56, 20}, hqs{0, 40, 52};
  for (int64_t r = 0; r < num_reqs; ++r)
    for (int64_t b = 0; b < maxblocks; ++b) hbt[(size_t)r * maxblocks + b] = (int32_t)(r * maxblocks + b);

  void* dq; void* dk; void* dv; void* dbt; void* dsl; void* dqs; void* do_=nullptr;
  cudaMalloc(&dq, hq_.size() * sizeof(__half));
  cudaMalloc(&dk, kv * sizeof(__half));
  cudaMalloc(&dv, kv * sizeof(__half));
  cudaMalloc(&dbt, hbt.size() * sizeof(int32_t));
  cudaMalloc(&dsl, 2 * sizeof(int32_t));
  cudaMalloc(&dqs, 3 * sizeof(int32_t));
  if (cudaMalloc(&do_, qn * hq * d * sizeof(float)) != cudaSuccess) return 1;
  cudaMemcpy(dq, hq_.data(), hq_.size() * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk.data(), kv * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), kv * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, hbt.data(), hbt.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dsl, hsl.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dqs, hqs.data(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice);

  cudaStream_t st = 0;
  vt_sm70_fa2_prefill(static_cast<float*>(do_), dq, dk, dv,
                      reinterpret_cast<const int32_t*>(dbt),
                      reinterpret_cast<const int32_t*>(dsl),
                      reinterpret_cast<const int32_t*>(dqs),
                      num_reqs, hq, nkv, d, bs, maxblocks, /*bt_col=*/1,
                      kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, 0.125f, st);
  cudaDeviceSynchronize();

  std::vector<float> got(qn * hq * d);
  cudaMemcpy(got.data(), do_, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

  auto cpu_kv = [&](std::vector<__half>& src, int64_t page, int64_t off, int64_t gh, int c) {
    return __half2float(src[((size_t)page * bs + (size_t)off) * nkv * d + (size_t)gh * d + (size_t)c]);
  };
  auto qv = [&](int row, int64_t h, int c) -> float { return __half2float(hq_[(size_t)row * hq * d + (size_t)h * d + (size_t)c]); };

  int bad = 0;
  for (int r = 0; r < 2 && !bad; ++r) {
    const int qlen = q1[r], seqlen = s1[r], ctxt = seqlen - qlen;
    for (int row = 0; row < qlen && !bad; ++row) {
      const int64_t absq = ctxt + row;
      const int gt = hqs[r] + row;
      for (int h = 0; h < (int)hq && !bad; ++h) {
        const int64_t g = h / (hq / nkv);
        std::vector<float> sc((size_t)absq + 1);
        for (int key = 0; key <= absq; ++key) {
          float s = 0.f;
          for (int c = 0; c < d; ++c) s += qv(gt, h, c) * cpu_kv(hk, r * maxblocks + key / bs, key % bs, g, c);
          sc[(size_t)key] = s * 0.125f;
        }
        float m = -1e30f;
        for (auto& x : sc) m = std::fmax(m, x);
        std::vector<double> w((size_t)absq + 1);
        double denom = 0.0;
        for (int i = 0; i <= absq; ++i) { w[(size_t)i] = std::exp((double)sc[(size_t)i] - m); denom += w[(size_t)i]; }
        for (int c2 = 0; c2 < (int)d; ++c2) {
          double num = 0.0;
          for (int i = 0; i <= absq; ++i)
            num += w[(size_t)i] * (double)cpu_kv(hv, r * maxblocks + i / bs, i % bs, g, c2);
          const float ref = (float)(num / denom);
          const float gv = got[((size_t)gt * hq + (size_t)h) * d + (size_t)c2];
          if (std::fabs((double)gv - (double)ref) >
              (double)tol_rel * std::max(1.0, std::fabs((double)ref))) { bad = 1; break; }
        }
      }
    }
  }
if (verbose) fprintf(stdout, "  prefill self-check: %s\\n", bad ? "MISMATCH" : "parity OK");
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dbt); cudaFree(dsl); cudaFree(dqs); cudaFree(do_);
  return bad;
}

// ---------------------------------------------------------------------------
// y-dg-sync coverage: the sm70 decode consumer must stay CUDA-graph-capture
// safe. The runner replays the steady-state decode inside a captured graph; any
// host-side synchronization (cudaStreamSynchronize/cudaDeviceSynchronize/cudaMalloc)
// issued on the capture stream would trip cudaErrorStreamCaptureImplicit and
// abort replay. This gate replays a decode through
//   cudaStreamBeginCapture -> graph launder -> cudaStreamEndCapture -> cudaGraphLaunch
// and requires replay == eager (same kernel, same inputs). A sync leak inside
// the decode chain fails the capture and this returns 1 (the FUTURE regression
// that the gate exists to catch). Returns 2 when not on sm_70.
extern "C" int vt_sm70_fa2_graph_replay_parity(float tol_rel, int verbose) {
  const DeviceCaps caps = GetDeviceCaps();
  if (!caps.valid || caps.sm_major != 7 || caps.sm_minor != 0) return 2;

  // All locals declared up front so the error paths cannot jump across an
  // initialized declaration (-Werror: goto-crosses-init).
  cudaError_t err = cudaSuccess;
  cudaStream_t st = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  __half *q_d = nullptr, *k_d = nullptr, *v_d = nullptr;
  int32_t *bt_d = nullptr, *sl_d = nullptr;
  float *eager = nullptr, *replay = nullptr;
  std::vector<float> he, hr;
  double max_rel = 0.0;
  int bad = 0;

  const int64_t nr = 3, hq = 1, nkv = 1, d = 64, bn = 16;
  const int64_t seqs[3] = {20, 33, 17};
  int64_t maxblocks = 1;
  for (int64_t r = 0; r < nr; ++r) {
    const int64_t nb = (seqs[r] + bn - 1) / bn;
    if (nb > maxblocks) maxblocks = nb;
  }
  const int64_t pages = nr * maxblocks;
  const int64_t kc_hd = d, kc_pg = nkv * d, kc_blk = bn * nkv * d;
  const int64_t vc_hd = d, vc_pg = nkv * d, vc_blk = bn * nkv * d;
  const float scale = 0.125f;

  const size_t kv_elems = (size_t)pages * (size_t)kc_blk;
  const size_t out_bytes = (size_t)nr * hq * d * sizeof(float);
  std::vector<unsigned short> hk16(kv_elems), hv16(kv_elems);
  std::vector<unsigned short> hq16((size_t)nr * hq * d);
  for (size_t i = 0; i < kv_elems; ++i) {
    hk16[i] = __half_as_ushort(__float2half_rn(0.37f * (float)((i * 13 + 7) % 19) + 0.11f));
    hv16[i] = __half_as_ushort(__float2half_rn(-0.23f * (float)((i * 7 + 3) % 13) - 0.07f));
  }
  for (size_t i = 0; i < hq16.size(); ++i)
    hq16[i] = __half_as_ushort(__float2half_rn(0.61f * (float)((i * 3 + 1) % 11) + 0.31f));

  std::vector<int32_t> hbt((size_t)nr * maxblocks);
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t b = 0; b < maxblocks; ++b)
      hbt[(size_t)r * maxblocks + b] = (int32_t)(r * maxblocks + b);
  std::vector<int32_t> hsl(seqs, seqs + nr);

  err = cudaMalloc((void**)&q_d, (size_t)nr * hq * d * sizeof(__half));
  err = (err ?: cudaMalloc((void**)&k_d, kv_elems * sizeof(__half)));
  err = (err ?: cudaMalloc((void**)&v_d, kv_elems * sizeof(__half)));
  err = (err ?: cudaMalloc((void**)&bt_d, (size_t)nr * maxblocks * sizeof(int32_t)));
  err = (err ?: cudaMalloc((void**)&sl_d, (size_t)nr * sizeof(int32_t)));
  err = (err ?: cudaMalloc((void**)&eager, out_bytes));
  err = (err ?: cudaMalloc((void**)&replay, out_bytes));
  if (err != cudaSuccess) { fprintf(stderr, "graph-replay: alloc: %s\n", cudaGetErrorString(err)); goto cleanup; }
  cudaMemcpy(q_d, hq16.data(), (size_t)nr * hq * d * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(k_d, hk16.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, hv16.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(bt_d, hbt.data(), (size_t)nr * maxblocks * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(sl_d, hsl.data(), (size_t)nr * sizeof(int32_t), cudaMemcpyHostToDevice);

  err = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  if (err != cudaSuccess) { fprintf(stderr, "graph-graph: stream create: %s\n", cudaGetErrorString(err)); goto cleanup; }

  // Eager decode on the dedicated (non-blocking) stream.
  vt_sm70_fa2_decode(eager, q_d, k_d, v_d, bt_d, sl_d, nr, hq, nkv, d, bn,
                     maxblocks, /*bt_col=*/1, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg,
                     vc_hd, scale, st);
  if (cudaStreamSynchronize(st) != cudaSuccess) { fprintf(stderr, "graph-graph: eager sync failed\n"); goto cleanup; }

  // Captured decode: the record path begins on the CUDA graph. Any host-sync /
  // malloc the decode launcher would inject into the capture stream fails the
  // capture with cudaErrorStreamCaptureImplicit -> rc==1 (the regression gate).
  err = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    fprintf(stderr, "y-dg-sync: begin-capture failed: %s (host-sync leak in decode)\n", cudaGetErrorString(err));
    goto cleanup;
  }
  vt_sm70_fa2_decode(replay, q_d, k_d, v_d, bt_d, sl_d, nr, hq, nkv, d, bn,
                     maxblocks, 1, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd,
                     scale, st);
  err = cudaStreamEndCapture(st, &graph);
  if (err != cudaSuccess) { fprintf(stderr, "y-dg-sync: end-capture failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
  exec = nullptr;
  err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
  if (err != cudaSuccess) {
    fprintf(stderr, "graph dg-sync: instantiate: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaGraphLaunch(exec, st);
  if (err == cudaSuccess) err = cudaStreamSynchronize(st);
  if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
  if (err != cudaSuccess) { fprintf(stderr, "graph dg-sync: replay launch/sync: %s\n", cudaGetErrorString(err)); goto cleanup; }

  he.assign(out_bytes / 4, 0.f);
  hr.assign(out_bytes / 4, 0.f);
  cudaMemcpy(he.data(), eager, out_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), replay, out_bytes, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < he.size(); ++i) {
    const double denom = he[i] > 1.0 ? he[i] : 1.0;
    max_rel = std::fmax(max_rel, std::fabs((double)he[i] - (double)hr[i]) / denom);
  }
  bad = (max_rel > (double)tol_rel) ? 1 : 0;

 cleanup:
  if (st) cudaStreamDestroy(st);
  cudaFree(q_d); cudaFree(k_d); cudaFree(v_d); cudaFree(bt_d); cudaFree(sl_d);
  cudaFree(eager); cudaFree(replay);
  if (verbose) fprintf(stdout, "  graph-replay parity: max relative dev %.3e (tol %.1e)\n", max_rel, (double)tol_rel);
  if (err == cudaErrorStreamCaptureImplicit) {
    fprintf(stderr, "y-dg-sync: decode captured a host-sync violation\n");
    return 1;
  }
  if (err != cudaSuccess) return 1;
  return bad;
}

} // namespace vt::cuda