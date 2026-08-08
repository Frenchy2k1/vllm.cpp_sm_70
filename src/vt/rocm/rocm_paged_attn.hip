// ROCm paged attention — port of cpu_paged_attn.cpp algebra + CUDA device-side
// request lookup (cuda_paged_attn.cu). Online softmax; NHD KV via strides.
// Supports bf16/f32 query x bf16/f32 KV (mixed allowed).
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/backend.h"

#include <cstdlib>
#include <vector>

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

// Optional correctness path: run the CPU paged-attn oracle on host copies.
// VT_ROCM_ATTN_CPU_REF=1 — isolates whether custom HIP attn is the decode bug.
bool CpuRefEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_ATTN_CPU_REF");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

void PagedAttnCpuRef(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                     const Tensor& v_cache, const Tensor& block_table,
                     const Tensor& seq_lens, const Tensor& query_start_loc,
                     const PagedAttentionArgs& args) {
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  Check(hipStreamSynchronize(st), "cpu_ref sync in");

  // Contiguous host packs (engine KV is a strided unbind view — cannot Bytes()-memcpy).
  auto pack4 = [&](const Tensor& t) {
    VT_CHECK(t.rank == 4, "cpu_ref pack4");
    const int64_t n0 = t.shape[0], n1 = t.shape[1], n2 = t.shape[2], n3 = t.shape[3];
    const size_t es = vt::SizeOf(t.dtype);
    std::vector<uint8_t> buf(static_cast<size_t>(n0 * n1 * n2 * n3) * es);
    // element-wise D2H via one big buffer read of parent is hard; use hipMemcpy2D-ish loop
    // Simple: copy full bounding box if contiguous innermost
    VT_CHECK(t.stride[3] == 1, "cpu_ref need unit inner stride");
    for (int64_t i0 = 0; i0 < n0; ++i0)
      for (int64_t i1 = 0; i1 < n1; ++i1)
        for (int64_t i2 = 0; i2 < n2; ++i2) {
          const int64_t src_off = i0 * t.stride[0] + i1 * t.stride[1] + i2 * t.stride[2];
          const int64_t dst_off = ((i0 * n1 + i1) * n2 + i2) * n3;
          Check(hipMemcpy(buf.data() + static_cast<size_t>(dst_off) * es,
                          static_cast<const char*>(t.data) + static_cast<size_t>(src_off) * es,
                          static_cast<size_t>(n3) * es, hipMemcpyDeviceToHost),
                "pack4 d2h");
        }
    return buf;
  };
  auto pack_contig = [&](const Tensor& t) {
    std::vector<uint8_t> buf(t.Bytes());
    Check(hipMemcpy(buf.data(), t.data, t.Bytes(), hipMemcpyDeviceToHost), "d2h");
    return buf;
  };

  auto h_q = pack_contig(query);
  auto h_k = pack4(k_cache);
  auto h_v = pack4(v_cache);
  auto h_bt = pack_contig(block_table);
  auto h_sl = pack_contig(seq_lens);
  auto h_qsl = pack_contig(query_start_loc);
  std::vector<uint8_t> h_out(out.Bytes(), 0);

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue cq = cpu.CreateQueue();
  const vt::Device cd{vt::DeviceType::kCPU, 0};

  Tensor tq = Tensor::Contiguous(h_q.data(), query.dtype, cd,
                                 {query.shape[0], query.shape[1], query.shape[2]});
  Tensor tk = Tensor::Contiguous(h_k.data(), k_cache.dtype, cd,
                                 {k_cache.shape[0], k_cache.shape[1], k_cache.shape[2],
                                  k_cache.shape[3]});
  Tensor tv = Tensor::Contiguous(h_v.data(), v_cache.dtype, cd,
                                 {v_cache.shape[0], v_cache.shape[1], v_cache.shape[2],
                                  v_cache.shape[3]});
  Tensor tbt = Tensor::Contiguous(h_bt.data(), block_table.dtype, cd,
                                  {block_table.shape[0], block_table.shape[1]});
  Tensor tsl = Tensor::Contiguous(h_sl.data(), seq_lens.dtype, cd, {seq_lens.shape[0]});
  Tensor tqsl =
      Tensor::Contiguous(h_qsl.data(), query_start_loc.dtype, cd, {query_start_loc.shape[0]});
  Tensor to = Tensor::Contiguous(h_out.data(), out.dtype, cd,
                                 {out.shape[0], out.shape[1], out.shape[2]});
  vt::PagedAttention(cq, to, tq, tk, tv, tbt, tsl, tqsl, args);
  cpu.DestroyQueue(cq);

  Check(hipMemcpy(out.data, h_out.data(), out.Bytes(), hipMemcpyHostToDevice), "h2d out");
  Check(hipStreamSynchronize(st), "cpu_ref sync out");
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

__device__ inline float Softcap(float s, float cap) {
  return cap > 0.f ? cap * tanhf(s / cap) : s;
}

// One block per (query_token, q_head). Online softmax matches CUDA.
template <typename TQ, typename TKV, typename TO>
__global__ void PagedAttnOnline(TO* out, const TQ* query, const TKV* k_cache,
                                const TKV* v_cache, const int32_t* block_table,
                                const int32_t* seq_lens, const int32_t* query_start_loc,
                                int64_t num_reqs, int64_t hq, int64_t num_kv_heads, int64_t d,
                                int64_t block_size, int64_t bt_row, int64_t bt_col,
                                int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                int64_t vc_pg, int64_t vc_hd, float scale, float softcap,
                                bool causal, int window_left, int window_right) {
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (h >= hq) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  for (int64_t rr = 0; rr < num_reqs; ++rr) {
    const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
    if (t >= a && t < b) {
      r = rr;
      q0 = a;
      q1 = b;
      break;
    }
  }
  if (r < 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);

  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;
  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;

  // shared: acc[d] + red[blockDim]
  extern __shared__ float smem[];
  float* acc = smem;
  float* red = smem + d;
  __shared__ float s_score, s_m, s_l;

  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.f;
  if (threadIdx.x == 0) {
    s_m = -INFINITY;
    s_l = 0.f;
  }
  __syncthreads();

  if (jmax >= jmin) {
    for (int64_t j = jmin; j <= jmax; ++j) {
      const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
      const int64_t off = j % block_size;
      const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
      float part = 0.f;
      for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
        part += Ld(query, qoff + e) * Ld(k_cache, kbase + e);
      red[threadIdx.x] = part;
      __syncthreads();
      for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
      }
      if (threadIdx.x == 0) s_score = Softcap(red[0] * scale, softcap);
      __syncthreads();

      const float s = s_score;
      const float m_new = fmaxf(s_m, s);
      const float corr = expf(s_m - m_new);
      const float pw = expf(s - m_new);
      const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
      for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
        acc[e] = acc[e] * corr + pw * Ld(v_cache, vbase + e);
      __syncthreads();
      if (threadIdx.x == 0) {
        s_l = s_l * corr + pw;
        s_m = m_new;
      }
      __syncthreads();
    }
  }

  const float inv = (s_l > 0.f) ? (1.f / s_l) : 0.f;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) St(out, qoff + e, acc[e] * inv);
}

}  // namespace

void PagedAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args) {
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto,
           "rocm paged_attention: fp8 KV not implemented");
  VT_CHECK(args.scale > 0.f, "rocm paged_attention: scale must be > 0");
  if (CpuRefEnabled()) {
    PagedAttnCpuRef(q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc,
                    args);
    return;
  }
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  VT_CHECK(hq % num_kv_heads == 0, "rocm paged_attention: GQA ratio");
  if (total_q == 0) return;

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  const int window_left =
      args.window_size.has_value() ? static_cast<int>(args.window_size->left) : -1;
  const int window_right =
      args.window_size.has_value() ? static_cast<int>(args.window_size->right) : -1;
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  // threads: reduce over head_dim; need power-of-2 <= 256 and <= d*something
  unsigned threads = 64;
  while (threads < 256u && threads < static_cast<unsigned>(d)) threads *= 2;
  if (threads > 256u) threads = 256;
  const size_t smem = sizeof(float) * (static_cast<size_t>(d) + threads);
  dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));

  auto launch = [&](auto q_tag, auto kv_tag, auto o_tag) {
    using TQ = decltype(q_tag);
    using TKV = decltype(kv_tag);
    using TO = decltype(o_tag);
    PagedAttnOnline<TQ, TKV, TO><<<grid, threads, smem, st>>>(
        out.Ptr<TO>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
        block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(),
        num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,
        vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal, window_left,
        window_right);
  };

  if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
      out.dtype == DType::kBF16) {
    launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
  } else if (query.dtype == DType::kF32 && k_cache.dtype == DType::kF32 &&
             out.dtype == DType::kF32) {
    launch(float{}, float{}, float{});
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kF32) {
    launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kF32 &&
             out.dtype == DType::kBF16) {
    launch(__hip_bfloat16{}, float{}, __hip_bfloat16{});
  } else if (query.dtype == DType::kF32 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kF32) {
    launch(float{}, __hip_bfloat16{}, float{});
  } else {
    VT_CHECK(false, "rocm paged_attention dtype");
  }
  Check(hipGetLastError(), "paged_attention launch");
}

}  // namespace vt::rocm
