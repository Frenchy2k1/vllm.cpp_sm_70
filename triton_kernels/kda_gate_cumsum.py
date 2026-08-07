# triton_kernels/kda_gate_cumsum.py  (KDA chunk-prefill kernel; see .agents/specs/kimi-linear.md §17)
#
# KDA chunk-prefill kernel 1/5: the fused decay-gate + chunk-local cumsum.
# VENDORED kernel source. Regenerate the per-arch cubins via scripts/regen-triton-aot.sh
# together with the cmake/TritonAOTKernels.cmake + CMakeLists.txt declarations in §17.
#
# Ported VERBATIM FROM (vLLM oracle @ pin 555967922):
#   vllm/third_party/flash_linear_attention/ops/kda.py:1182-1254
#     @triton.jit kda_gate_cumsum_fwd_kernel  (driver fused_kda_gate_chunk_cumsum :1257)
#   (upstream flash-linear-attention; MIT). The kernel BODY is byte-for-byte the
#   FLA source. AOT adaptations only:
#     (1) @triton.heuristics / @triton.autotune removed — HAS_BIAS/IS_VARLEN and
#         dims (H,D,BT,BD) PINNED per-shape via the compile SIGNATURE (see §17).
#     (2) the runtime scalars cumsum_scale/beta/threshold are REMOVED and pinned
#         as module literals: the KDA driver ALWAYS passes cumsum_scale=RCP_LN2
#         (natural-log -> log2 fold, kda.py:1295), softplus beta=1.0 and
#         threshold=20.0 (fused_kda_gate_chunk_cumsum defaults, kda.py:1261-1262).
#         Triton's AOT launcher mis-packs an fp32 scalar as an 8-byte double, so a
#         runtime float arg is unusable (same reason chunk_o.py pins `scale`).
#     (3) trailing runtime scalar `NT` (= total chunks) appended as the grid-y
#         carrier; B*H == H (varlen packing has B=1) is the constexpr grid-z.
#         `NT` is unused by the body (dead arg).
#
# Pinned for Kimi-Linear KDA: H=32, D=128 (head_k_dim), BT=64, BD=64 (=> gx=2),
# HAS_BIAS=1 (g_bias = dt_bias[H*D]), IS_VARLEN=1.
# Buffers (mirror FLA dtypes): g(=raw_g projection)=[T,H,D] bf16;
# A(=A_log)=[H] fp32; y(=g cumulative, exp2-space)=[T,H,D] fp32;
# g_bias(=dt_bias)=[H*D] fp32; cu_seqlens=[N+1] i32; chunk_indices=[NT,2] i32.
import triton
import triton.language as tl

# fla/ops/op.py: exp/log (FLA_USE_FAST_OPS=0 default -> tl.exp / tl.log).
exp = tl.exp
log = tl.log

# fold + softplus constants pinned per the KDA driver (see header note 2).
# Instantiated as tl.constexpr: Triton 3.6 rejects a plain-float module global
# accessed from a @jit kernel — the constexpr INSTANTIATION form
# (`x = tl.constexpr(v)`) is required; the annotation form (`x: tl.constexpr = v`)
# is NOT supported for module globals.
RCP_LN2 = tl.constexpr(1.4426950408889634)  # 1 / ln(2)
SOFTPLUS_BETA = tl.constexpr(1.0)
SOFTPLUS_THRESHOLD = tl.constexpr(20.0)


@triton.jit(do_not_specialize=["T", "NT"])
def kda_gate_cumsum_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    T,
    NT,  # AOT grid-carrier (= total chunks): grid-y extent; unused by the body.
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    beta = SOFTPLUS_BETA
    threshold = SOFTPLUS_THRESHOLD
    cumsum_scale = RCP_LN2
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_y = tl.make_block_ptr(
        y + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        o_d = i_d * BD + tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_a = -tl.exp(tl.load(A + i_h).to(tl.float32))
    b_g_scaled = b_g * beta
    b_softplus = tl.where(
        b_g_scaled > threshold,
        b_g,
        (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
    )
    b_gate = b_a * b_softplus

    o_t = tl.arange(0, BT)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
