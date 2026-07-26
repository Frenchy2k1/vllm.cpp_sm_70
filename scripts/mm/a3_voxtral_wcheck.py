#!/usr/bin/env python3
"""Compare vLLM's LOADED layer-0 q_proj weight vs the RAW consolidated wq to detect
a rope permutation (interleaved<->NeoX) applied on the mistral load path."""
import glob, struct, json
import numpy as np
import torch
from vllm import LLM

MODEL = "mistralai/Voxtral-Mini-3B-2507"
llm = LLM(model=MODEL, tokenizer_mode="mistral", config_format="mistral",
          load_format="mistral", max_model_len=2048, max_num_seqs=1,
          enforce_eager=True, enable_chunked_prefill=False, gpu_memory_utilization=0.30)

m = llm.llm_engine.model_executor.driver_worker.model_runner.model
attn = m.language_model.model.layers[0].self_attn
qkv = attn.qkv_proj.weight.detach().float().cpu().numpy()  # [q+2kv, H]
print("vLLM qkv_proj.weight", qkv.shape)
q_v = qkv[:4096]   # q rows
k_v = qkv[4096:4096+1024]

# raw consolidated
f = open(glob.glob("/home/mudler/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-3B-2507/snapshots/*/consolidated.safetensors")[0], "rb")
n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
def get(name):
    t = hdr[name]; s, e = t["data_offsets"]
    f.seek(base + s); raw = f.read(e - s)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().numpy().reshape(t["shape"])
wq = get("layers.0.attention.wq.weight")  # [4096,3072]
wk = get("layers.0.attention.wk.weight")
print("raw wq", wq.shape)

print("q identical to raw wq:", np.allclose(q_v, wq, atol=1e-3), "maxdiff", np.abs(q_v-wq).max())
print("k identical to raw wk:", np.allclose(k_v, wk, atol=1e-3), "maxdiff", np.abs(k_v-wk).max())

def permute(w, n_heads, d1, d2):  # Meta interleaved -> HF NeoX
    return w.reshape(n_heads, d1 // n_heads // 2, 2, d2).transpose(0, 2, 1, 3).reshape(d1, d2)
def unpermute(w, n_heads, d1, d2):  # HF NeoX -> Meta interleaved
    return w.reshape(n_heads, 2, d1 // n_heads // 2, d2).transpose(0, 2, 1, 3).reshape(d1, d2)

pq = permute(wq, 32, 4096, 3072)
uq = unpermute(wq, 32, 4096, 3072)
print("permute(wq)==vLLM q:", np.allclose(pq, q_v, atol=1e-2), "maxdiff", np.abs(pq-q_v).max())
print("unpermute(wq)==vLLM q:", np.allclose(uq, q_v, atol=1e-2), "maxdiff", np.abs(uq-q_v).max())
pk = permute(wk, 8, 1024, 3072)
print("permute(wk)==vLLM k:", np.allclose(pk, k_v, atol=1e-2), "maxdiff", np.abs(pk-k_v).max())

# also dump the rope inv_freq / theta actually used
try:
    re = attn.rotary_emb
    print("rotary head_size", getattr(re, "head_size", getattr(re, "rotary_dim", "?")),
          "base", getattr(re, "base", "?"), "is_neox", getattr(re, "is_neox_style", "?"))
except Exception as e:
    print("rope introspect err", e)
print("WCHECK OK")
