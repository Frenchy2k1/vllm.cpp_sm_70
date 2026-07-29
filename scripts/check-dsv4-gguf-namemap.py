#!/usr/bin/env python3
# DeepSeek-V4 GGUF (`deepseek4` arch) blk.N.* -> our-V4-tensor name map + FULL
# coverage gate (CLAIM-DEEPSEEK-V4-W8). This is the name-map W8 owes: it encodes
# the GGUF->V4 slot map for `unsloth/DeepSeek-V4-Flash-GGUF`, GENERATES the
# complete expected tensor set from the arch config's per-layer topology, and
# asserts EXACT set-equality against the real 1328-tensor manifest — every GGUF
# tensor maps to a known V4 slot, none unmapped, none of our expected slots left
# over. rc=0 iff coverage is total.
#
# Topology (read from the GGUF KV metadata, cross-checked in the manifest):
#   block_count=43, hash_layer_count=3, expert_count=256, sinkhorn_iters=20.
#   Indexer (DSA Lightning-Indexer) is present on the `compress_ratio==4`
#   layers = {2} U even{4..42} = 21 layers; the DSA compressor is present on
#   every `compress_ratio!=0` layer = all but the first two = 41 layers; the
#   first `hash_layer_count`=3 layers {0,1,2} carry the `tid2eid` hash table and
#   NO `exp_probs_b` noaux_tc bias, the other 40 carry the bias.
#
# The manifest fixture (real tensor names, from the shard GGUF headers via HTTP
# range — no 91 GiB download) lives beside this script; regenerate with
# scripts/dsv4_gguf_manifest_names.txt. When the fixture is absent the checker
# still validates the map is internally complete (generates 1328, all mapped).
import os, re, sys

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}                      # hash_layer_count = 3
INDEXER_LAYERS = {2} | set(range(4, 43, 2))  # compress_ratio == 4  -> 21 layers
COMPRESSOR_LAYERS = set(range(2, 43))        # compress_ratio != 0  -> 41 layers

# ----------------------------------------------------------------------------
# The name map. Each GGUF `blk.N.<suffix>` (and each top-level name) maps to a
# (V4 weight tower slot, GgufTensorRole). Roles drive keep-quant residency
# (gguf_keep_quant.cpp): kMatmulWeight/kStackedExpertWeight KEEP their ~2-3-bit
# blocks (the memory enabler), everything else is a vector/norm/table.
#   MW = matmul weight        SEW = stacked expert weight (256 experts)
#   V  = vector/norm/bias/scale (F32)   ET = embedding table   HASH = i32 table
# Slot names mirror our DeepseekV4 weight tower (deepseek_v4.h / _weights.cpp).
PER_LAYER = {
    # 512-wide MLA
    "attn_norm.weight":            ("layer.attn_norm",            "V"),
    "attn_q_a.weight":             ("layer.mla.wq_a",             "MW"),
    "attn_q_a_norm.weight":        ("layer.mla.q_a_norm",         "V"),
    "attn_q_b.weight":             ("layer.mla.wq_b",             "MW"),
    "attn_kv.weight":              ("layer.mla.wkv",              "MW"),
    "attn_kv_a_norm.weight":       ("layer.mla.kv_a_norm",        "V"),
    "attn_output_a.weight":        ("layer.mla.wo_a",             "MW"),
    "attn_output_b.weight":        ("layer.mla.wo_b",             "MW"),
    "attn_sinks.weight":           ("layer.mla.attn_sink",        "V"),
    # MoE
    "ffn_norm.weight":             ("layer.ffn_norm",             "V"),
    "ffn_gate_inp.weight":         ("layer.moe.gate",             "MW"),
    "ffn_gate_exps.weight":        ("layer.moe.experts.w_gate",   "SEW"),
    "ffn_up_exps.weight":          ("layer.moe.experts.w_up",     "SEW"),
    "ffn_down_exps.weight":        ("layer.moe.experts.w_down",   "SEW"),
    "ffn_gate_shexp.weight":       ("layer.moe.shared.w_gate",    "MW"),
    "ffn_up_shexp.weight":         ("layer.moe.shared.w_up",      "MW"),
    "ffn_down_shexp.weight":       ("layer.moe.shared.w_down",    "MW"),
    # MHC (hyper-connection) per-layer mixing
    "hc_attn_base.weight":         ("layer.mhc.attn_base",        "V"),
    "hc_attn_fn.weight":           ("layer.mhc.attn_fn",          "V"),
    "hc_attn_scale.weight":        ("layer.mhc.attn_scale",       "V"),
    "hc_ffn_base.weight":          ("layer.mhc.ffn_base",         "V"),
    "hc_ffn_fn.weight":            ("layer.mhc.ffn_fn",           "V"),
    "hc_ffn_scale.weight":         ("layer.mhc.ffn_scale",        "V"),
}
HASH_ONLY = {
    "ffn_gate_tid2eid.weight":     ("layer.moe.tid2eid",          "HASH"),
}
GATED_ONLY = {
    "exp_probs_b.bias":            ("layer.moe.e_score_bias",     "V"),
}
COMPRESSOR = {
    "attn_compressor_ape.weight":  ("layer.dsa.compressor.ape",   "V"),
    "attn_compressor_gate.weight": ("layer.dsa.compressor.wgate", "MW"),
    "attn_compressor_kv.weight":   ("layer.dsa.compressor.wkv",   "MW"),
    "attn_compressor_norm.weight": ("layer.dsa.compressor.norm",  "V"),
}
INDEXER = {
    "indexer.attn_q_b.weight":            ("layer.dsa.indexer.wq_b",         "MW"),
    "indexer.proj.weight":                ("layer.dsa.indexer.weights_proj", "V"),
    "indexer_compressor_ape.weight":      ("layer.dsa.indexer.compressor.ape",   "V"),
    "indexer_compressor_gate.weight":     ("layer.dsa.indexer.compressor.wgate", "MW"),
    "indexer_compressor_kv.weight":       ("layer.dsa.indexer.compressor.wkv",   "MW"),
    "indexer_compressor_norm.weight":     ("layer.dsa.indexer.compressor.norm",  "V"),
}
TOP_LEVEL = {
    "token_embd.weight":     ("embed_tokens",      "ET"),
    "output.weight":         ("lm_head",           "MW"),
    "output_norm.weight":    ("final_norm",        "V"),
    "output_hc_base.weight": ("mhc.head_base",     "V"),
    "output_hc_fn.weight":   ("mhc.head_fn",       "V"),
    "output_hc_scale.weight":("mhc.head_scale",    "V"),
}


def expected_names():
    """Generate the full expected GGUF tensor set from the topology."""
    out = {}
    for name, slot in TOP_LEVEL.items():
        out[name] = slot
    for L in range(N_LAYERS):
        table = dict(PER_LAYER)
        table.update(HASH_ONLY if L in HASH_LAYERS else GATED_ONLY)
        if L in COMPRESSOR_LAYERS:
            table.update(COMPRESSOR)
        if L in INDEXER_LAYERS:
            table.update(INDEXER)
        for suffix, slot in table.items():
            out[f"blk.{L}.{suffix}"] = (f"layers.{L}." + slot[0].split('.', 1)[1]
                                        if slot[0].startswith("layer.") else slot[0],
                                        slot[1])
    return out


def load_manifest_names():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dsv4_gguf_manifest_names.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return set(l.strip() for l in f if l.strip())


def main():
    exp = expected_names()
    n = len(exp)
    ok = True
    if n != 1328:
        print(f"FAIL: name map generates {n} tensors, expected 1328")
        ok = False
    else:
        print(f"OK: name map generates {n} tensors (matches split.tensors.count=1328)")

    # every generated tensor maps to a non-empty V4 slot
    unmapped = [k for k, v in exp.items() if not v[0]]
    if unmapped:
        print(f"FAIL: {len(unmapped)} generated tensors map to an empty slot")
        ok = False

    keep_quant_roles = sum(1 for v in exp.values() if v[1] in ("MW", "SEW"))
    print(f"     keep-quant-capable roles (MW/SEW): {keep_quant_roles}; "
          f"experts (SEW): {sum(1 for v in exp.values() if v[1]=='SEW')}")

    real = load_manifest_names()
    if real is None:
        print("WARN: manifest fixture absent — internal-completeness check only")
    else:
        missing = real - set(exp)   # real tensors our map does NOT cover
        leftover = set(exp) - real  # slots we expect that the file lacks
        if missing:
            print(f"FAIL: {len(missing)} real GGUF tensors UNMAPPED, e.g. "
                  f"{sorted(missing)[:5]}")
            ok = False
        if leftover:
            print(f"FAIL: {len(leftover)} mapped slots have NO real tensor, e.g. "
                  f"{sorted(leftover)[:5]}")
            ok = False
        if not missing and not leftover:
            print(f"OK: EXACT coverage — all {len(real)} real GGUF tensors mapped, "
                  f"0 unmapped, 0 leftover")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
