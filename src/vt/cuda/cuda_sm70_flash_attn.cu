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

#include "../../../third_party/flash_attn_v100/include/fused_mma.h"
#include "../../../third_party/flash_attn_v100/kernel/flash_v100_traits.cuh"

namespace vt::cuda {

constexpr int kF2D = 64;       // head dim served
constexpr int kF2Keys = 16;    // 16-key tile (m16n16k16 N)
constexpr int kF2Threads = 32;

// One block per (token, q-head); 32 threads (one warp).
__global__ void Sm70Fa2DecodeAttn(
    float* out, const __half* query, const __half* k_cache, const __half* v_cache,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* query_start,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, float softcap) {
  if (d != kF2D) return;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (h >= hq) return;
  const int64_t seqlen = seq_lens[t];
  const int64_t g = h / (hq / num_kv);
  const int64_t qoff = (t * hq + h) * d;

  extern __shared__ __half sm[];  // carve in __half
  __half* sq = sm;                // [16][64] fp16, row0 = 64-elem query
  __half* sk = sq + 16 * 64;      // key tile [16][64] key-major
  __half* sb = sk + 16 * 64;      // transposed B panel [16][16]
  float* sc = reinterpret_cast<float*>(sb + 16 * 16);  // [16][16] f32 scores
  float* sacc = sc + 16 * 16;     // [64] running acc

  // Load query dp0 row 0; zero rows 1..15; zero acc.
  for (int e = threadIdx.x; e < kF2D; e += blockDim.x)
    sq[e] = query[qoff + e];
  for (int e = threadIdx.x + kF2D; e < 16 * kF2D; e += blockDim.x)
    sq[e] = __ushort_as_half(0);
  for (int e = threadIdx.x; e < kF2D; e += blockDim.x) sacc[e] = 0.f;
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
      for (int ee2 = threadIdx.x; ee2 < kF2D; ee2 += blockDim.x)
        sacc[ee2] = sacc[ee2] * corr +
                    pw * __half2float(v_cache[blk * vc_blk + off * vc_pg + g * vc_hd + ee2]);
      __syncwarp();
      m = m1;
      l = l * corr + pw;  // all lanes: same s, so m/l stay uniform
    }
    __syncthreads();
  }

  const float inv = 1.0f / l;
  for (int e = threadIdx.x; e < kF2D; e += blockDim.x) out[qoff + e] = sacc[e] * inv;
}

// ---- host entries (extern "C", raw pointers, no torch) ----
extern "C" void vt_sm70_fa2_decode(
    float* out, const void* query, const void* k, const void* v,
    const int32_t* block_table, const int32_t* seq_lens,
    int64_t num_reqs, int64_t hq, int64_t num_kv, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, cudaStream_t stream) {
  const int smem =
      (16*64 + 16*64 + 16*16) /*halves*/ * 2 + (16*16 + 64 + 32) /*floats*/ * 4;
  const dim3 grid((unsigned)num_reqs, (unsigned)hq);  // x=token, y=head (ref parity)
  Sm70Fa2DecodeAttn<<<grid, kF2Threads, smem, stream>>>(
      out, static_cast<const __half*>(query), static_cast<const __half*>(k),
      static_cast<const __half*>(v), block_table, seq_lens, nullptr,
      num_reqs, hq, num_kv, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,
      vc_blk, vc_pg, vc_hd, scale, 0.f);
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
  const int64_t num_reqs = 3, hq = 1, num_kv = 1, d = 64, bn = 16;
  const int64_t maxblocks = 3;  // >= ceil(seq/bn) for every request
  const int64_t pages = num_reqs * maxblocks;
  const int64_t kc_hd = d, kc_pg = num_kv * d, kc_blk = bn * num_kv * d;
  const int64_t vc_hd = d, vc_pg = num_kv * d, vc_blk = bn * num_kv * d;
  const float scale = 0.125f;  // ~ 1/sqrt(64)

  // ---- fill host buffers deterministically ----
  const size_t kv_elems = (size_t)pages * kc_blk;
  std::vector<unsigned short> hk16(kv_elems), hv16(kv_elems);
  std::vector<unsigned short> hq16((size_t)num_reqs * hq * d);
  for (size_t i = 0; i < kv_elems; ++i) {
    hk16[i] = (unsigned short)(((i * 13 + 7) % 19) + 1);           // 1..19 bits
    hv16[i] = (unsigned short)(((i * 7 + 3) % 13) + 2);
  }
  for (size_t i = 0; i < hq * (size_t)num_reqs * d; ++i)
    hq16[i] = (unsigned short)(((i * 3 + 1) % 11) + 1);

  // device alloc
  __half *q_d, *k_d, *v_d;
  int32_t *bt_d, *sl_d;
  float *mine, *ref;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void**)&q_d, (size_t)num_reqs * hq * d * sizeof(__half));
  err = (err ?: cudaMalloc((void**)&k_d, kv_elems * sizeof(__half)));
  err = (err ?: cudaMalloc((void**)&v_d, kv_elems * sizeof(__half)));
  err = (err ?: cudaMalloc((void**)&bt_d, (size_t)num_reqs * maxblocks * sizeof(int32_t)));
  err = (err ?: cudaMalloc((void**)&sl_d, (size_t)num_reqs * sizeof(int32_t)));
  err = (err ?: cudaMalloc((void**)&mine, (size_t)num_reqs * hq * d * sizeof(float)));
  err = (err ?: cudaMalloc((void**)&ref, (size_t)num_reqs * hq * d * sizeof(float)));
  if (err != cudaSuccess) {
    fprintf(stderr, "sm70 fa2 self-check: alloc: %s\n", cudaGetErrorString(err));
    cudaFree((void*)q_d); cudaFree((void*)k_d); cudaFree((void*)v_d);
    cudaFree((void*)bt_d); cudaFree((void*)sl_d); cudaFree((void*)mine); cudaFree((void*)ref);
    return -1;
  }
  cudaMemcpy(q_d, hq16.data(), (size_t)num_reqs * hq * d * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_d, hk16.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, hv16.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);

  // block table: token r -> page id r*maxblocks + block
  std::vector<int32_t> hbt((size_t)num_reqs * maxblocks);
  for (int64_t r = 0; r < num_reqs; ++r)
    for (int64_t b = 0; b < maxblocks; ++b) hbt[(size_t)r * maxblocks + b] = (int32_t)(r * maxblocks + b);
  std::vector<int32_t> hsl{20, 17, 33};
  cudaMemcpy(bt_d, hbt.data(), (size_t)num_reqs * maxblocks * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(sl_d, hsl.data(), (size_t)num_reqs * sizeof(int32_t), cudaMemcpyHostToDevice);

  const int32_t bt_col = 1;
  cudaStream_t st = 0;
  vt_sm70_fa2_decode(mine, q_d, k_d, v_d, bt_d, sl_d, num_reqs, hq, num_kv, d, bn,
                     maxblocks, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd,
                     scale, st);
  vt_sm70_fa2_ref_decode(ref, q_d, k_d, v_d, bt_d, sl_d, num_reqs, hq, num_kv, d, bn,
                          maxblocks, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd,
                          scale, st);
  err = cudaGetLastError();

  std::vector<float> hmine((size_t)num_reqs * hq * d), href((size_t)num_reqs * hq * d);
  if (err == cudaSuccess) {
    cudaMemcpy(hmine.data(), mine, (size_t)num_reqs * hq * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(href.data(), ref, (size_t)num_reqs * hq * d * sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    fprintf(stderr, "sm70 fa2 self-check: run: %s\n", cudaGetErrorString(err));
    cudaFree((void*)q_d); cudaFree((void*)k_d); cudaFree((void*)v_d);
    cudaFree((void*)bt_d); cudaFree((void*)sl_d); cudaFree((void*)mine); cudaFree((void*)ref);
    return -1;
  }

  double max_reldev = 0.0;
  for (size_t i = 0; i < hmine.size(); ++i) {
    const double denom = href[i] > 1.0 ? href[i] : 1.0;
    const double rel = std::fabs((double)hmine[i] - href[i]) / denom;
    if (rel > max_reldev) max_reldev = rel;
  }
  cudaFree((void*)q_d); cudaFree((void*)k_d); cudaFree((void*)v_d);
  cudaFree((void*)bt_d); cudaFree((void*)sl_d); cudaFree((void*)mine); cudaFree((void*)ref);

  if (verbose) fprintf(stdout, "sm70 fa2 self-check: max rel dev vs reference = %.3e (tol %.2e)\n",
                       max_reldev, (double)tol_rel);
  return (max_reldev <= (double)tol_rel) ? 0 : 1;
}

} // namespace vt::cuda