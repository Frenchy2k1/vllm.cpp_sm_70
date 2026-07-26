#!/usr/bin/env python3
# DFlash D3 — CONTEXT-KV PRECOMPUTE + prepare_dflash_inputs + BLOCK-PROPOSAL
# reference capture (spec .agents/specs/dflash-spec-decode.md §6 D3, port map
# DF-DRAFT-KV-PREP). Dumps, from the REAL loaded vLLM in-tree DFlash draft, the
# three fixtures our C++ D3 path is gated against IN ISOLATION (the propose/verify
# e2e is D4):
#   * prepare_ref.json  : the prepare_dflash_inputs outputs (input_ids, query
#                         positions/slots, context positions/slots, sample maps,
#                         seq_lens) for a FIXED synthetic target batch, computed by
#                         a line-for-line numpy replica of _prepare_dflash_inputs_kernel
#                         (dflash/speculator.py:472-618). These are INTEGER, so our
#                         host PrepareDflashInputs must match BIT-exact.
#   * ctxkv_ref.json    : the precomputed context K/V (K normed+RoPE'd, V raw) for a
#                         fixed synthetic context, produced by the REAL loaded draft's
#                         precompute buffers (_build_fused_kv_buffers +
#                         _project_context_kv + _normalize_context_k + the fused RoPE,
#                         qwen3_dflash.py:440-619). Envelope-exact vs our
#                         PrecomputeContextKV (bf16/f32-softmax envelope).
#   * propose_ref.json  : the (1+k) block proposal — per-layer hidden, final hidden,
#                         top-k proposed ids — from the REAL submodules with the
#                         precomputed context K/V prepended to each layer's attention
#                         (the D3 context-aware forward our ForwardBlockLogitsWithContext
#                         computes). STRICT-or-ratified-near-tie on the k proposed ids.
#
# HOW IT RUNS (finalized for the dgx GPU-promotion session, vLLM 0.26.0.dev0 @
# 555967922). MEMORY: the 27B target is MULTIMODAL -> vision-encoder profiling
# OOM-reboots the GB10 (see [[gb10-unified-memory-oom-reboots-box]]); use
# limit_mm_per_prompt={image:0,video:0} + gpu_memory_utilization=0.30 (the D0 fix).
# Run under `flock $HOME/gpu.lock`, oracle venv on PATH (FlashInfer JIT needs ninja),
# VLLM_USE_V2_MODEL_RUNNER=1, no FLASH_ATTN pin:
#   flock $HOME/gpu.lock env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     VLLM_USE_V2_MODEL_RUNNER=1 ~/venvs/vllm-oracle/bin/python \
#     scripts/spec/d3_dflash_kvprep_ref.py --out-dir tests/parity/goldens/dflash_27b_kvprep
import argparse
import os

TARGET = os.environ.get("DFLASH_TARGET", "unsloth/Qwen3.6-27B-NVFP4")
DRAFT = os.environ.get("DFLASH_DRAFT", "z-lab/Qwen3.6-27B-DFlash")
BLOCK = int(os.environ.get("DFLASH_BLOCK", "16"))
MASK_TOKEN_ID = int(os.environ.get("DFLASH_MASK_TOKEN_ID", "248070"))
ANCHOR_TOKEN_ID = int(os.environ.get("DFLASH_ANCHOR_TOKEN_ID", "9707"))  # "Hello"
CTX_LEN = int(os.environ.get("DFLASH_CTX_LEN", "12"))
BLOCK_SIZE = int(os.environ.get("DFLASH_KV_BLOCK_SIZE", "16"))  # paged KV block size


def prepare_numpy_ref(ctx_len, block, mask_id, anchor_id, kv_block_size):
    """Line-for-line numpy replica of _prepare_dflash_inputs_kernel
    (dflash/speculator.py:508-618) for ONE request, no rejection, first prefill.
    Integer only — the bit-exact oracle for our host PrepareDflashInputs."""
    nqpr = block  # 1 + (block-1)
    nspec = block - 1
    # Single request, contiguous prefill: positions 0..ctx_len-1, block table maps
    # each position to a distinct slot via block_size.
    target_positions = list(range(ctx_len))
    ctx_start, ctx_end = 0, ctx_len
    num_rejected = 0
    valid_ctx_end = ctx_end - num_rejected
    last_valid_pos = target_positions[valid_ctx_end - 1]
    num_blocks = (ctx_len + nqpr + kv_block_size - 1) // kv_block_size
    block_table = list(range(100, 100 + num_blocks))  # arbitrary distinct block ids

    def slot(pos):
        bn = min(pos // kv_block_size, len(block_table) - 1)
        return block_table[bn] * kv_block_size + (pos % kv_block_size)

    context_positions = list(target_positions)
    context_slot_mapping = [slot(p) for p in context_positions]
    input_ids, query_positions, query_slot_mapping = [], [], []
    sample_indices, sample_pos, sample_idx_mapping = [], [], []
    for off in range(nqpr):
        qpos = last_valid_pos + 1 + off
        input_ids.append(anchor_id if off == 0 else mask_id)
        query_positions.append(qpos)
        query_slot_mapping.append(slot(qpos))
        if off >= 1:
            sample_indices.append(off)
            sample_pos.append(qpos)
            sample_idx_mapping.append(0)
    return {
        "ctx_len": ctx_len, "block": block, "kv_block_size": kv_block_size,
        "block_table": block_table,
        "input_ids": input_ids, "query_positions": query_positions,
        "query_slot_mapping": query_slot_mapping,
        "context_positions": context_positions,
        "context_slot_mapping": context_slot_mapping,
        "sample_indices": sample_indices, "sample_pos": sample_pos,
        "sample_idx_mapping": sample_idx_mapping,
        "seq_lens": [last_valid_pos + 1 + nqpr],
    }


def _dump_worker(worker, out_dir, block, mask_id, anchor_id, ctx_len, kv_block_size):
    """Runs INSIDE the vLLM worker (via collective_rpc). Reaches the loaded
    DFlashQwen3ForCausalLM and dumps ctxkv_ref + propose_ref using vLLM's OWN
    weights/modules; writes prepare_ref from the numpy replica."""
    import json
    import os

    import torch

    os.makedirs(out_dir, exist_ok=True)
    runner = worker.model_runner
    drafter = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
    if drafter is None:
        for v in vars(runner).values():
            m = getattr(v, "model", None)
            if m is not None and m.__class__.__name__.startswith("DFlash"):
                drafter = v
                break
    if drafter is None:
        raise RuntimeError("could not find the DFlash speculator on the model runner")
    draft = drafter.model  # DFlashQwen3ForCausalLM
    model = draft.model
    dev = next(draft.parameters()).device
    dt = next(draft.parameters()).dtype

    cfg = draft.config
    H = cfg.hidden_size
    Dh = cfg.head_dim
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    eps = cfg.rms_norm_eps
    nlayers = cfg.num_hidden_layers
    scale = Dh ** -0.5
    causal = draft.get_draft_attn_causal()
    T = block

    # ---- prepare_ref (numpy replica, integer) ----
    with open(os.path.join(out_dir, "prepare_ref.json"), "w") as f:
        json.dump(prepare_numpy_ref(ctx_len, block, mask_id, anchor_id, kv_block_size), f)

    # ---- ctxkv_ref: precompute context K/V from a fixed synthetic context via the
    # REAL loaded precompute buffers (qwen3_dflash.py:440-619). ----
    if not hasattr(model, "_num_attn_layers"):
        model._build_fused_kv_buffers()
    context_states = (0.35 * torch.sin(
        torch.arange(ctx_len * H).float() * 0.21).reshape(ctx_len, H)).to(dev, dt)
    context_positions = torch.arange(ctx_len, device=dev)
    with torch.no_grad():
        all_k, all_v = model._project_context_kv(
            context_states, ctx_len, model._num_attn_layers,
            model._num_kv_heads, model._head_dim)
        all_k_normed = model._normalize_context_k(all_k)
        all_k_flat = all_k_normed.view(model._num_attn_layers * ctx_len, model._kv_size)
        pos_rep = context_positions.repeat(model._num_attn_layers)
        cs = model._rope_cos_sin_cache
        if cs.dtype != all_k_flat.dtype:
            cs = cs.to(dtype=all_k_flat.dtype)
        from vllm import _custom_ops as ops
        ops.rotary_embedding(pos_rep, all_k_flat, None, model._rope_head_size, cs,
                             model._rope_is_neox)
        k_final = all_k_flat.view(model._num_attn_layers, ctx_len, Hkv, Dh).float().cpu()
        v_final = all_v.view(model._num_attn_layers, ctx_len, Hkv, Dh).float().cpu()
    with open(os.path.join(out_dir, "ctxkv_ref.json"), "w") as f:
        json.dump({
            "ctx_len": ctx_len, "num_layers": nlayers, "Hkv": Hkv, "Dh": Dh,
            "context_states": context_states.float().cpu().flatten().tolist(),
            "context_positions": context_positions.cpu().tolist(),
            "k": k_final.flatten().tolist(), "v": v_final.flatten().tolist(),
        }, f)

    # ---- propose_ref: the (1+k) block forward with the precomputed context K/V
    # prepended to each layer's attention (the D3 context-aware attention). ----
    input_ids = [anchor_id] + [mask_id] * (block - 1)
    block_positions = list(range(ctx_len, ctx_len + block))  # anchor at ctx_len
    ids = torch.tensor(input_ids, device=dev)
    bpos = torch.tensor(block_positions, device=dev)

    def ctx_attn(q, k, v, ck, cv, layer_causal):
        # q [B,Hq,Dh]; k/v [B,Hkv,Dh] (block); ck/cv [C,Hkv,Dh] (context).
        rep = Hq // Hkv
        kk = torch.cat([ck, k], dim=0).repeat_interleave(rep, dim=1)  # [C+B,Hq,Dh]
        vv = torch.cat([cv, v], dim=0).repeat_interleave(rep, dim=1)
        qf, kf, vf = q.float(), kk.float(), vv.float()
        scores = torch.einsum("thd,shd->hts", qf, kf) * scale  # [Hq,B,C+B]
        B = q.shape[0]
        C = ck.shape[0]
        neg = float("-inf")
        mask = torch.zeros(B, C + B, device=q.device)
        if layer_causal:  # SWA: causal over [context; block], context all in past
            for i in range(B):
                for j in range(C + B):
                    if j > C + i:
                        mask[i, j] = neg
        probs = torch.softmax(scores + mask.unsqueeze(0), dim=-1)
        return torch.einsum("hts,shd->thd", probs, vf).to(q.dtype)

    per_layer = []
    with torch.no_grad():
        hidden = model.embed_input_ids(ids)
        residual = None
        for li, layer in enumerate(model.layers):
            sa = layer.self_attn
            if residual is None:
                residual = hidden
                hn = layer.input_layernorm(hidden)
            else:
                hn, residual = layer.input_layernorm(hidden, residual)
            qkv, _ = sa.qkv_proj(hn)
            q, k, v = qkv.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
            q = sa.q_norm(q.view(-1, Hq, Dh)).view(-1, sa.q_size)
            k = sa.k_norm(k.view(-1, Hkv, Dh)).view(-1, sa.kv_size)
            q, k = sa.rotary_emb(bpos, q, k)
            ck = k_final[li].to(dev, q.dtype)
            cv = v_final[li].to(dev, q.dtype)
            ao = ctx_attn(q.view(-1, Hq, Dh), k.view(-1, Hkv, Dh),
                          v.view(-1, Hkv, Dh), ck, cv, causal[li]).reshape(-1, Hq * Dh)
            attn_out, _ = sa.o_proj(ao)
            hn2, residual = layer.post_attention_layernorm(attn_out, residual)
            hidden = layer.mlp(hn2)
            per_layer.append(hidden.float().cpu().flatten().tolist())
        final, _ = model.norm(hidden, residual)
        logits = draft.logits_processor(draft.lm_head, final)
        if logits is None:
            logits = torch.matmul(final.float(), draft.lm_head.weight.float().t())
        logits = logits.float().cpu()
        tk = torch.topk(logits, k=8, dim=-1)
    with open(os.path.join(out_dir, "propose_ref.json"), "w") as f:
        json.dump({
            "input_ids": input_ids, "block_positions": block_positions,
            "ctx_len": ctx_len, "per_layer_causal": list(map(bool, causal)),
            "per_layer_hidden": per_layer,
            "final_hidden": final.float().cpu().flatten().tolist(),
            "draft_vocab": int(logits.shape[1]),
            "topk_ids": tk.indices.tolist(), "topk_vals": tk.values.tolist(),
        }, f)
    return {"ok": True, "nlayers": nlayers, "ctx_len": ctx_len, "block": block}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/parity/goldens/dflash_27b_kvprep")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--num-spec-tokens", type=int, default=BLOCK)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from vllm import LLM

    llm = LLM(
        model=args.target,
        speculative_config={
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_spec_tokens,
        },
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=True,
        max_model_len=2048,
    )
    out_dir = os.path.abspath(args.out_dir)
    res = llm.collective_rpc(
        _dump_worker,
        args=(out_dir, BLOCK, MASK_TOKEN_ID, ANCHOR_TOKEN_ID, CTX_LEN, BLOCK_SIZE))
    print(f"[d3] worker dump result = {res}")
    print(f"[d3] wrote reference fixtures to {out_dir}")


if __name__ == "__main__":
    main()
