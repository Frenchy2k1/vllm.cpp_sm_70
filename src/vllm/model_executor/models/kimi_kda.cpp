// Kimi Delta Attention (KDA) net-new primitives — host reference impls.
// See kimi_kda.h for the full port map (file:line on both sides, @ 555967922).
#include "vllm/model_executor/models/kimi_kda.h"

#include <cmath>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm::kimi_kda {

namespace {
// softplus with a beta/threshold linearisation, exactly as the KDA gate kernels
// (kda.py:1595-1597, :1240-1244): switch to the linear branch when beta*x
// overflows exp. Computed in double for the portable reference.
inline double SoftplusBeta(double x, double beta, double threshold) {
  const double scaled = x * beta;
  if (scaled > threshold) return x;  // linear regime
  return (1.0 / beta) * std::log1p(std::exp(scaled));
}

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

std::vector<float> KdaLowRankDecay(const std::vector<float>& x,
                                   const std::vector<float>& f_a,
                                   const std::vector<float>& f_b,
                                   int64_t num_tokens, int64_t hidden_size,
                                   int64_t num_heads, int64_t head_dim) {
  const int64_t hd = num_heads * head_dim;
  VT_CHECK(hidden_size > 0 && head_dim > 0 && num_heads > 0, "bad kda dims");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * hidden_size,
           "x size mismatch");
  VT_CHECK(static_cast<int64_t>(f_a.size()) == head_dim * hidden_size,
           "f_a size mismatch");
  VT_CHECK(static_cast<int64_t>(f_b.size()) == hd * head_dim, "f_b size mismatch");

  std::vector<float> g1(static_cast<size_t>(num_tokens) * hd, 0.0f);
  std::vector<float> r(static_cast<size_t>(head_dim));  // f_a(x) per token
  for (int64_t t = 0; t < num_tokens; ++t) {
    const float* x_t = &x[t * hidden_size];
    // r = f_a @ x_t    ([head_dim x hidden] @ [hidden])
    for (int64_t o = 0; o < head_dim; ++o) {
      float acc = 0.0f;
      const float* fa_o = &f_a[o * hidden_size];
      for (int64_t i = 0; i < hidden_size; ++i) acc += fa_o[i] * x_t[i];
      r[static_cast<size_t>(o)] = acc;
    }
    // g1_t = f_b @ r    ([H*D x head_dim] @ [head_dim])
    float* g_t = &g1[t * hd];
    for (int64_t o = 0; o < hd; ++o) {
      float acc = 0.0f;
      const float* fb_o = &f_b[o * head_dim];
      for (int64_t i = 0; i < head_dim; ++i) acc += fb_o[i] * r[static_cast<size_t>(i)];
      g_t[o] = acc;
    }
  }
  return g1;
}

std::vector<float> KdaDecayGate(const std::vector<float>& g1,
                                const std::vector<float>& a_log,
                                const std::vector<float>& dt_bias,
                                int64_t num_tokens, int64_t num_heads,
                                int64_t head_dim, float beta, float threshold) {
  const int64_t hd = num_heads * head_dim;
  VT_CHECK(static_cast<int64_t>(g1.size()) == num_tokens * hd, "g1 size mismatch");
  VT_CHECK(static_cast<int64_t>(a_log.size()) == num_heads, "a_log size mismatch");
  const bool has_bias = !dt_bias.empty();
  VT_CHECK(!has_bias || static_cast<int64_t>(dt_bias.size()) == hd,
           "dt_bias size mismatch");

  std::vector<float> y(static_cast<size_t>(num_tokens) * hd, 0.0f);
  for (int64_t h = 0; h < num_heads; ++h) {
    const double b_a = -std::exp(static_cast<double>(a_log[h]));  // -exp(A_log[h])
    for (int64_t t = 0; t < num_tokens; ++t) {
      const float* g_t = &g1[t * hd + h * head_dim];
      float* y_t = &y[t * hd + h * head_dim];
      for (int64_t d = 0; d < head_dim; ++d) {
        double b_g = g_t[d];
        if (has_bias) b_g += dt_bias[h * head_dim + d];
        const double sp = SoftplusBeta(b_g, beta, threshold);
        y_t[d] = static_cast<float>(b_a * sp);
      }
    }
  }
  return y;
}

std::vector<float> KdaDecayGateChunkCumsum(const std::vector<float>& g1,
                                           const std::vector<float>& a_log,
                                           const std::vector<float>& dt_bias,
                                           int64_t num_tokens, int64_t num_heads,
                                           int64_t head_dim, int64_t chunk_size,
                                           bool log2_domain, float beta,
                                           float threshold) {
  VT_CHECK(chunk_size > 0, "chunk_size must be positive");
  const int64_t hd = num_heads * head_dim;
  // Per-position gate first (reuse the exact KdaDecayGate numerics).
  const std::vector<float> gate =
      KdaDecayGate(g1, a_log, dt_bias, num_tokens, num_heads, head_dim, beta,
                   threshold);
  const double scale = log2_domain ? kRcpLn2 : 1.0;

  std::vector<float> y(static_cast<size_t>(num_tokens) * hd, 0.0f);
  // Chunk-local cumulative sum along time, resetting at each chunk boundary
  // (kda.py:1252-1253 tril-ones matmul is a per-chunk prefix sum). Accumulate in
  // double for a stable reference.
  for (int64_t h = 0; h < num_heads; ++h) {
    for (int64_t d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (int64_t t = 0; t < num_tokens; ++t) {
        if (t % chunk_size == 0) acc = 0.0;  // chunk boundary reset
        acc += static_cast<double>(gate[t * hd + h * head_dim + d]);
        y[t * hd + h * head_dim + d] = static_cast<float>(acc * scale);
      }
    }
  }
  return y;
}

std::vector<float> FusedRMSNormGated(const std::vector<float>& x,
                                     const std::vector<float>& g,
                                     const std::vector<float>& weight,
                                     int64_t num_tokens, int64_t num_heads,
                                     int64_t head_dim,
                                     GatedNormActivation activation, float eps) {
  const int64_t hd = num_heads * head_dim;
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * hd, "x size mismatch");
  VT_CHECK(static_cast<int64_t>(g.size()) == num_tokens * hd, "g size mismatch");
  const bool affine = !weight.empty();
  VT_CHECK(!affine || static_cast<int64_t>(weight.size()) == head_dim,
           "weight size mismatch");

  std::vector<float> out(static_cast<size_t>(num_tokens) * hd, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t h = 0; h < num_heads; ++h) {
      const float* x_r = &x[t * hd + h * head_dim];
      const float* g_r = &g[t * hd + h * head_dim];
      float* o_r = &out[t * hd + h * head_dim];
      // variance = mean(x^2) over head_dim; rstd = rsqrt(variance + eps).
      double var = 0.0;
      for (int64_t d = 0; d < head_dim; ++d)
        var += static_cast<double>(x_r[d]) * x_r[d];
      var /= static_cast<double>(head_dim);
      const double rstd = 1.0 / std::sqrt(var + eps);
      for (int64_t d = 0; d < head_dim; ++d) {
        double normed = static_cast<double>(x_r[d]) * rstd;
        if (affine) normed *= weight[d];
        const double gd = g_r[d];
        double res;
        if (activation == GatedNormActivation::kSwish)
          res = normed * gd * Sigmoid(gd);  // swish/silu gate
        else
          res = normed * Sigmoid(gd);       // sigmoid gate (KDA)
        o_r[d] = static_cast<float>(res);
      }
    }
  }
  return out;
}

std::vector<float> KdaShortConv(const std::vector<float>& x,
                                const std::vector<float>& weight,
                                const std::vector<float>& bias,
                                int64_t num_tokens, int64_t channels,
                                int64_t kernel_size) {
  VT_CHECK(kernel_size > 0 && channels > 0, "bad conv dims");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * channels,
           "x size mismatch");
  VT_CHECK(static_cast<int64_t>(weight.size()) == channels * kernel_size,
           "conv weight size mismatch");
  const bool has_bias = !bias.empty();
  VT_CHECK(!has_bias || static_cast<int64_t>(bias.size()) == channels,
           "conv bias size mismatch");

  std::vector<float> y(static_cast<size_t>(num_tokens) * channels, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t c = 0; c < channels; ++c) {
      double acc = has_bias ? static_cast<double>(bias[c]) : 0.0;
      const float* w_c = &weight[c * kernel_size];
      // Causal, zero initial state: tap j aligns to input position
      // t - (K-1) + j; negative positions read 0.
      for (int64_t j = 0; j < kernel_size; ++j) {
        const int64_t pos = t - (kernel_size - 1) + j;
        if (pos < 0) continue;
        acc += static_cast<double>(w_c[j]) * x[pos * channels + c];
      }
      y[t * channels + c] = static_cast<float>(acc * Sigmoid(acc));  // silu
    }
  }
  return y;
}

std::vector<float> L2NormRows(const std::vector<float>& x, int64_t num_rows,
                              int64_t dim, float eps) {
  VT_CHECK(dim > 0, "bad l2norm dim");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_rows * dim, "x size mismatch");
  std::vector<float> y(static_cast<size_t>(num_rows) * dim, 0.0f);
  for (int64_t t = 0; t < num_rows; ++t) {
    const float* x_r = &x[t * dim];
    double ss = 0.0;
    for (int64_t d = 0; d < dim; ++d) ss += static_cast<double>(x_r[d]) * x_r[d];
    const double rstd = 1.0 / std::sqrt(ss + eps);  // sum, not mean
    for (int64_t d = 0; d < dim; ++d)
      y[t * dim + d] = static_cast<float>(static_cast<double>(x_r[d]) * rstd);
  }
  return y;
}

}  // namespace vllm::kimi_kda
