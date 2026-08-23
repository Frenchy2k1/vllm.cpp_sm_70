// SPDX-License-Identifier: Apache-2.0
// sm70 decode attention — Volta tensor-core fast path (Phase 2 brick F).
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

constexpr int kF2D = 64;       // head dim served
constexpr int kF2Keys = 16;    // 16-key tile (m16n16k16 N)
constexpr int kF2Threads = 32;

// One block per (token, q-head); 32 threads (one warp). Head-dim-TD templated
// (instantiated for 64 and 128) so real Llama/Mistral/Qwen head widths run.
template <int TD>
__global__ void Sm70Fa2DecodeAttn(
    float* out, const __half* query, const __half* k_cache, const __half* v_cache,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* query_start,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, float softcap) {
  if (d != TD) return;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (h >= hq) return;
  const int64_t seqlen = seq_lens[t];
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


// ====== FA2-V100 PREFILL (Phase 2 brick G) : WMMA fp16, D==64 =============
// One (request, q-head) block, 32 threads. Queries are 16-row slices, keys
// stream in 16-key tiles; QK^T per (q-slice x 16 keys) is a m16n16k16 WMMA over
// the 4 d-panels (D==64); the per-row flash (online max/sum + weighted-V) is
// fp32 SIMT with causal masking (key j valid iff j <= context + row). Same
// paged strides/contract as PagedAttentionKernel.
// Honest scope: fp16 Q/KV, D==64, causal, request q_len <= 64 (else decline).
constexpr int kPFMaxQ = 64;

__global__ void Sm70FaPrefillAttn(
    float* out, const __half* query, const __half* k_cache, const __half* v_cache,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* q_lens,
    const int32_t* q_start, int64_t num_reqs, int64_t hq, int64_t nkv, int64_t d,
    int64_t block_size, int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
    int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale) {
  if (d != kF2D) return;
  const int64_t r = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (r >= num_reqs || h >= hq) return;
  const int qlen = (int)q_lens[r];
  const int seqlen = (int)seq_lens[r];
  if (qlen <= 0 || qlen > kPFMaxQ || seqlen <= 0 || h >= hq) return;
  const int ctxt = seqlen - qlen;
  const int64_t g = h / (hq / nkv);
  const int64_t qbase = q_start[r];   // global query-token row of req r

  extern __shared__ __half ps[];
  __half* sq = ps;                 // [16][64] q-slice
  __half* sk = sq + 16 * kF2D;     // [16][64] key tile
  __half* sv = sk + 16 * kF2D;     // [16][64] v tile
  __half* sb = sv + 16 * kF2D;     // [16][16] B transpose
  __half* sa16 = sb + 16 * 16;      // [16][16] A tile (16-wide rows, for wmma)
  float* ss = reinterpret_cast<float*>(sa16 + 16 * 16);  // [16][16] scores
  float* acc = ss + 16 * 16;       // [16][64] running acc (current slice)
  float* mrow = acc + 16 * kF2D;   // [16]
  float* lrow = mrow + 16;         // [16]

  for (int qs = 0; qs < qlen; qs += 16) {
    const int nrows = (qlen - qs) < 16 ? (qlen - qs) : 16;
    // load q-slice A (rows 0..nrows-1 valid, rest zero)
    for (int e = threadIdx.x; e < 16 * kF2D; e += blockDim.x) sq[e] = __ushort_as_half(0);
    for (int e = threadIdx.x; e < nrows * d; e += blockDim.x) {
      const int rr = e / (int)d, cc = e % (int)d;
      sq[rr * kF2D + cc] = query[((qbase + qs + rr) * hq + h) * d + cc];
    }
    for (int e = threadIdx.x; e < 16 * kF2D; e += blockDim.x) acc[e] = 0.f;
    for (int e = threadIdx.x; e < 16; e += blockDim.x) { mrow[e] = -CUDART_INF_F; lrow[e] = 0.f; }
    __syncthreads();
    // -- key stream over all keys [0, seqlen) --
    for (int64_t jb = 0; jb < (int64_t)seqlen; jb += kF2Keys) {
      const int nkeys = (int)(((int)seqlen - (int)jb) < kF2Keys ? ((int)seqlen - (int)jb) : kF2Keys);
      for (int e = threadIdx.x; e < 16 * kF2D; e += blockDim.x) sk[e] = __ushort_as_half(0);
      for (int e = threadIdx.x; e < 16 * kF2D; e += blockDim.x) sv[e] = __ushort_as_half(0);
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
          sa16[e] = (rr < nrows) ? sq[rr * kF2D + (p * 16 + kk)] : __ushort_as_half(0);
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
        for (int e = threadIdx.x; e < kF2D; e += blockDim.x) acc[rr * kF2D + e] *= corr;
        ls *= corr;
        for (int ccelf = 0; ccelf < 16; ++ccelf) {
          const int kf = (int)jb + ccelf;
          if (kf <= absq) {
            float w = expf(ss[rr * 16 + ccelf] * scale - mnew);
            ls += w;
            for (int e = threadIdx.x; e < kF2D; e += blockDim.x)
              acc[rr * kF2D + e] += w * __half2float(sv[ccelf * d + e]);
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

extern "C" void vt_sm70_fa2_prefill(
    float* out, const void* query, const void* k, const void* v,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* q_lens,
    const int32_t* q_start, int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d,
    int64_t block_size, int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
    int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale,
    cudaStream_t stream) {
  const int smem = (4 * 16 * kF2D + 16 * 16) /*sq,sk,sv,sb,sa16 halves*/ * 2 +
                   (16 * 16 + 16 * kF2D + 32) /*ss,acc,mrow,lrow*/ * 4;
  const dim3 grid((unsigned)num_reqs, (unsigned)hq);
  Sm70FaPrefillAttn<<<grid, kF2Threads, smem, stream>>>(
      out, static_cast<const __half*>(query), static_cast<const __half*>(k),
      static_cast<const __half*>(v), block_table, seq_lens, q_lens, q_start,
      num_reqs, hq, num_kv, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,
      vc_blk, vc_pg, vc_hd, scale);
}

// ---- host entries (extern "C", raw pointers, no torch) ----
extern "C" void vt_sm70_fa2_decode(
    float* out, const void* query, const void* k, const void* v,
    const int32_t* block_table, const int32_t* seq_lens,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, cudaStream_t stream) {
  const int td = (int)d;
  if (td != 64 && td != 128) return;  // caller-side gate; do not launch garbage
  const dim3 grid((unsigned)num_reqs, (unsigned)hq);
#define VT_DECODE_LAUNCH(TD)                                                        \
  do {                                                                               \
    const int smem = (2 * 16 * (TD) /*sq,sk*/ + 16 * 16 /*sb*/) * 2 +               \
                     (16 * 16 + (TD) + 32) /*sc,sacc,pad*/ * 4;                     \
    Sm70Fa2DecodeAttn<(TD)><<<grid, kF2Threads, smem, stream>>>(                     \
        out, static_cast<const __half*>(query), static_cast<const __half*>(k),       \
        static_cast<const __half*>(v), block_table, seq_lens, nullptr,               \
        num_reqs, hq, num_kv, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,    \
        vc_blk, vc_pg, vc_hd, scale, 0.f);                                             \
  } while (0)
  if (td == 64) VT_DECODE_LAUNCH(64);
  else          VT_DECODE_LAUNCH(128);
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
  const Cfg cfgs[3] = {
      {3, 1, 1, 64, {20, 17, 33, 0, 0, 0, 0, 0}},
      {2, 8, 2, 64, {9, 25, 0, 0, 0, 0, 0, 0}},
      {2, 4, 1, 128, {25, 17, 0, 0, 0, 0, 0, 0}},  // D=128 (Llama/Mistral width)
  };
  double worst = 0.0;
  const int ncfg = 3;
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
  std::vector<int32_t> hbt((size_t)num_reqs * maxblocks), hsl{56, 20}, hql{40, 12}, hqs{0, 40};
  for (int64_t r = 0; r < num_reqs; ++r)
    for (int64_t b = 0; b < maxblocks; ++b) hbt[(size_t)r * maxblocks + b] = (int32_t)(r * maxblocks + b);

  void* dq; void* dk; void* dv; void* dbt; void* dsl; void* dql; void* dqs; void* do_=nullptr;
  cudaMalloc(&dq, hq_.size() * sizeof(__half));
  cudaMalloc(&dk, kv * sizeof(__half));
  cudaMalloc(&dv, kv * sizeof(__half));
  cudaMalloc(&dbt, hbt.size() * sizeof(int32_t));
  cudaMalloc(&dsl, 2 * sizeof(int32_t));
  cudaMalloc(&dql, 2 * sizeof(int32_t));
  cudaMalloc(&dqs, 2 * sizeof(int32_t));
  if (cudaMalloc(&do_, qn * hq * d * sizeof(float)) != cudaSuccess) return 1;
  cudaMemcpy(dq, hq_.data(), hq_.size() * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk.data(), kv * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), kv * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, hbt.data(), hbt.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dsl, hsl.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dql, hql.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dqs, hqs.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

  cudaStream_t st = 0;
  vt_sm70_fa2_prefill(static_cast<float*>(do_), dq, dk, dv,
                      reinterpret_cast<const int32_t*>(dbt),
                      reinterpret_cast<const int32_t*>(dsl),
                      reinterpret_cast<const int32_t*>(dql),
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
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dbt); cudaFree(dsl); cudaFree(dql); cudaFree(dqs); cudaFree(do_);
  return bad;
}

} // namespace vt::cuda