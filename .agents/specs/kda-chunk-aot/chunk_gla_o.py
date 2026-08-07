# triton_kernels/chunk_gla_o.py  (STAGED — see .agents/specs/kimi-linear.md §17)
#
# KDA chunk-prefill kernel 5/5: the GLA-style output with per-K-channel gk decay.
# STAGED authored harness body (Phase-1 spike); Phase-2 moves it to
# triton_kernels/ + regenerates the sm_121a cubin.
#
# Ported VERBATIM FROM (vLLM oracle @ pin 555967922):
#   vllm/third_party/flash_linear_attention/ops/kda.py:1019-1123
#     @triton.jit chunk_gla_fwd_kernel_o  (driver chunk_gla_fwd_o_gk :1126)
#   (upstream flash-linear-attention; MIT). Kernel BODY byte-for-byte FLA. AOT
#   adaptations only:
#     (1) @triton.heuristics / @triton.autotune removed — IS_VARLEN and dims
#         (H,K,V,BT,BK,BV) PINNED via the compile SIGNATURE (see §17).
#     (2) runtime `scale` REMOVED and pinned to K**-0.5 (the chunk_kda scale,
#         kda.py:1472 `scale = k.shape[-1]**-0.5`; K=128). Triton AOT cannot take a
#         reliable fp32 scalar (see chunk_o.py note 3); the KDA call site always
#         passes scale == K**-0.5.
#     (3) trailing runtime scalar `NT` (= total chunks) appended as the grid-y
#         carrier: FLA grid is (cdiv(V,BV), NT, B*H); B*H == H (varlen B=1) is the
#         constexpr grid-z. `NT` is unused by the body (dead arg).
#
# Pinned for Kimi-Linear KDA: H=32, K=128, V=128, BT=64, BK=64 (=> 2 K-iters),
# BV=64 (=> gx=cdiv(V,BV)=2), IS_VARLEN=1.
# Buffers (mirror FLA dtypes): q=[T,H,K] bf16; v(=v_new)=[T,H,V] bf16;
# g(=gk cumulative, exp2-space)=[T,H,K] fp32; h(=hstate snapshot)=[NT,H,V,K] bf16;
# A(=Aqk)=[T,H,BT] fp32; o(=out, reuses v buffer)=[T,H,V] bf16;
# cu_seqlens=[N+1] i32; chunk_indices=[NT,2] i32.
import triton
import triton.language as tl

exp = tl.exp
exp2 = tl.exp2


@triton.jit(do_not_specialize=["T", "NT"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    NT,  # AOT grid-carrier (= total chunks): grid-y extent; unused by the body.
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # scale = K**-0.5 (chunk_kda q-scale), pinned since Triton AOT can't take an
    # fp32 scalar arg reliably; K == 128 for the Kimi KDA shape (see header note 2).
    scale = K ** -0.5
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
