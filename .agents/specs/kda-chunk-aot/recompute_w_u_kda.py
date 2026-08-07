# triton_kernels/recompute_w_u_kda.py  (STAGED — see .agents/specs/kimi-linear.md §17)
#
# KDA chunk-prefill kernel 4/5: recompute W and U (the WY-representation "apply"),
# the KDA PER-K-CHANNEL variant. Distinct from the GDN wy_fast.py kernel: KDA
# applies exp2(b_gk) per-K-channel and stores kg = k*exp2(gn - gk) (the STORE_KG
# branch). STAGED authored harness body (Phase-1 spike); Phase-2 moves it to
# triton_kernels/ + regenerates the sm_121a cubin.
#
# Ported VERBATIM FROM (vLLM oracle @ pin 555967922):
#   vllm/third_party/flash_linear_attention/ops/kda.py:817-957
#     @triton.jit recompute_w_u_fwd_kernel  (driver recompute_w_u_fwd :960)
#   (upstream flash-linear-attention; MIT). Kernel BODY byte-for-byte FLA. AOT
#   adaptations only:
#     (1) @triton.heuristics / @triton.autotune removed — STORE_QG/STORE_KG/
#         IS_VARLEN and dims (H,K,V,BT,BK,BV) PINNED via the compile SIGNATURE.
#     (2) DOT_PRECISION REMOVED and pinned to "ieee" (driver recompute_w_u_fwd
#         always passes DOT_PRECISION="ieee", kda.py:1002); it is a compile-time
#         constexpr string, baked as a module literal.
#     (3) trailing runtime scalar `NT` appended as the grid-x carrier (grid is
#         (NT, B*H); B*H == H is the constexpr grid-y). `NT` unused by the body.
#
# Pinned for Kimi-Linear KDA: H=32, K=128, V=128, BT=64, BK=64, BV=64,
# STORE_QG=0 (q/qg dead), STORE_KG=1 (kg written), IS_VARLEN=1. Grid (NT, 32, 1).
# Buffers (mirror FLA dtypes): k=[T,H,K] bf16; v=[T,H,V] bf16; beta=[T,H] bf16;
# w=[T,H,K] bf16 (out); u=[T,H,V] bf16 (out); A(=solve_tril inverse)=[T,H,BT] bf16;
# gk(=g cumulative, exp2-space)=[T,H,K] fp32; kg=[T,H,K] bf16 (out); q/qg dead
# (STORE_QG=0); cu_seqlens=[N+1] i32; chunk_indices=[NT,2] i32.
import triton
import triton.language as tl

exp = tl.exp
exp2 = tl.exp2

# driver always passes DOT_PRECISION="ieee" (kda.py:1002); see header note 2.
DOT_PRECISION: tl.constexpr = "ieee"


@triton.jit(do_not_specialize=["T", "NT"])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    NT,  # AOT grid-carrier (grid-x extent = total chunks); unused by the body.
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
