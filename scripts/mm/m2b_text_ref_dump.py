#!/usr/bin/env python3
# M2b/M2c — dump vLLM Qwen3-VL text-backbone reference tensors for the fixed M1
# fixture (image, prompt), to anchor the C++ unit gates (RED-first). Four small
# committed references + a manifest with sha256:
#
#   1. rope_index_positions_i32   [3, T]  — vLLM Qwen3VL._get_mrope_input_positions
#      (get_rope_index): image tokens get (t,h,w) grid positions, text sequential.
#   2. rotary cos_sin cache + q/k in/out — the 3-section MRoPE application. A fixed
#      seeded q [T,Hq,D] / k [T,Hk,D] rotated by vLLM MRotaryEmbedding.forward_native
#      with positions [3,T]; plus the (bf16-as-f32) cos_sin_cache used, so the C++
#      gate feeds the IDENTICAL cache to vt::RopeFromCache and isolates the
#      section-select + rotation math (mrope_interleaved=True, section=[24,20,20]).
#   3. deepstack scatter — a fixed seeded multiscale [N,3*H] scattered by
#      _compute_deepstack_embeds into [3, T, H] (zero at non-visual positions).
#   4. embed-merge — a fixed seeded main [N,H] + text embeds [T,H] merged by
#      _merge_multimodal_embeddings (masked scatter at is_multimodal positions).
#
# Grounds: vllm/model_executor/models/qwen3_vl.py @ e24d1b24
#   _get_mrope_input_positions (:2567), _iter_mm_grid_hw (:2482),
#   _compute_deepstack_embeds (:2761), Qwen3LLMModel.forward deepstack (:1589);
#   vllm/model_executor/models/utils.py::_merge_multimodal_embeddings (:524);
#   vllm/model_executor/layers/rotary_embedding/mrope.py MRotaryEmbedding (:201).
#
# All little-endian; f32 unless noted. Env: CKPT_DIR, FIX_DIR (the committed
# qwen3vl fixture), OUT_DIR.
import glob
import hashlib
import json
import os

import numpy as np
import torch

CKPT = os.environ["CKPT_DIR"]
FIX = os.environ["FIX_DIR"]
OUT = os.environ["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

manifest = {"model_id": "Qwen/Qwen3-VL-4B-Instruct", "refs": {}}

# vLLM CustomOp (MRotaryEmbedding) instantiation requires an active vLLM config
# context, even on CPU. Enter one for the whole script.
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

_mcfg = ModelConfig(model=CKPT, tokenizer=CKPT, trust_remote_code=True,
                    dtype="bfloat16", enforce_eager=True)
_vllm_config = VllmConfig(model_config=_mcfg)
_ctx = set_current_vllm_config(_vllm_config)
_ctx.__enter__()


def dump(name, arr, note=""):
    a = np.ascontiguousarray(arr)
    p = os.path.join(OUT, name + ".bin")
    a.tofile(p)
    manifest["refs"][name] = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "file": name + ".bin",
        "note": note,
    }
    print(f"  dumped {name} {a.shape} {a.dtype}", flush=True)


cfg = json.load(open(os.path.join(CKPT, "config.json")))
tc = cfg.get("text_config", cfg)
H = int(tc["hidden_size"])              # 2560
Hq = int(tc["num_attention_heads"])     # 32
Hkv = int(tc["num_key_value_heads"])    # 8
Dh = int(tc.get("head_dim", H // Hq))   # 128
theta = float(tc.get("rope_theta", 1e6))
rope_scaling = tc.get("rope_scaling", {}) or {}
mrope_section = list(rope_scaling.get("mrope_section", [24, 20, 20]))
mrope_interleaved = bool(rope_scaling.get("mrope_interleaved", False))
vcfg = cfg["vision_config"]
merge = int(vcfg["spatial_merge_size"])
out_hidden = int(vcfg["out_hidden_size"])       # 2560 (== H)
n_deepstack = len(vcfg["deepstack_visual_indexes"])  # 3
image_token_id = int(cfg["image_token_id"])
video_token_id = int(cfg["video_token_id"])
vstart = int(cfg["vision_start_token_id"])
vend = int(cfg["vision_end_token_id"])
print(f"H={H} Hq={Hq} Hkv={Hkv} Dh={Dh} theta={theta} "
      f"mrope_section={mrope_section} interleaved={mrope_interleaved} "
      f"out_hidden={out_hidden} n_deepstack={n_deepstack}", flush=True)

manifest["config"] = dict(
    hidden_size=H, num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=Dh,
    rope_theta=theta, mrope_section=mrope_section, mrope_interleaved=mrope_interleaved,
    out_hidden_size=out_hidden, n_deepstack=n_deepstack,
    image_token_id=image_token_id, video_token_id=video_token_id,
    vision_start_token_id=vstart, vision_end_token_id=vend, spatial_merge_size=merge)

# ---- fixture ids + grid ------------------------------------------------------
fixman = json.load(open(os.path.join(FIX, "manifest.json")))
input_ids = list(fixman["expanded_prompt_token_ids"])
T = len(input_ids)
grid_thw = np.fromfile(os.path.join(FIX, "image_grid_thw_i64.bin"), dtype=np.int64)  # [1,28,28]
gt, gh, gw = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
N = (gt * gh * gw) // (merge * merge)   # 196 merged visual tokens
print(f"T={T} grid_thw=({gt},{gh},{gw}) N_visual={N}", flush=True)
is_multimodal = np.array([tid == image_token_id for tid in input_ids], dtype=bool)
assert int(is_multimodal.sum()) == N, (int(is_multimodal.sum()), N)

# ===========================================================================
# 1. get_rope_index  (vLLM Qwen3VL._get_mrope_input_positions)
# ===========================================================================
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration


class _Pos:
    def __init__(self, offset):
        self.offset = offset


class _Feat:
    def __init__(self, offset, thw):
        self.modality = "image"
        self.mm_position = _Pos(offset)
        self.data = {"image_grid_thw": torch.tensor(thw, dtype=torch.long)}


class _VC:  # minimal config shim for _get_mrope_input_positions
    def __init__(self):
        self.video_token_id = video_token_id
        self.vision_start_token_id = vstart
        self.vision_end_token_id = vend

        class _V:
            pass
        self.vision_config = _V()
        self.vision_config.spatial_merge_size = merge


offset = input_ids.index(image_token_id)
feats = [_Feat(offset, [gt, gh, gw])]
llm_positions, delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
    input_ids, feats, _VC())
positions = llm_positions.numpy().astype(np.int32)  # [3, T]
assert positions.shape == (3, T), positions.shape
print(f"  mrope_position_delta={delta}  pos[:, :3]=\n{positions[:, :3]}", flush=True)
dump("rope_index_positions_i32", positions, "vLLM _get_mrope_input_positions [3,T] int32")

# ===========================================================================
# 2. 3-section MRoPE application (MRotaryEmbedding.forward_native)
# ===========================================================================
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

rope = MRotaryEmbedding(
    head_size=Dh, rotary_dim=Dh, max_position_embeddings=int(tc["max_position_embeddings"]),
    base=theta, is_neox_style=True, dtype=torch.bfloat16,
    mrope_section=mrope_section, mrope_interleaved=mrope_interleaved)

# Reduced head counts keep the fixture small; rope is per-head-identical and
# per-pair, so 4 q-heads / 2 kv-heads still catch head-indexing and section bugs
# at the REAL rotary_dim (Dh=128) + section=[24,20,20]. (Production uses Hq/Hkv.)
Hq_t, Hkv_t = 4, 2
g = torch.Generator().manual_seed(2026)
q_in = (torch.randn(T, Hq_t * Dh, generator=g, dtype=torch.float32) * 0.5).to(torch.bfloat16)
k_in = (torch.randn(T, Hkv_t * Dh, generator=g, dtype=torch.float32) * 0.5).to(torch.bfloat16)
pos_t = torch.from_numpy(positions.astype(np.int64))  # [3,T]
q_out, k_out = rope.forward_native(pos_t, q_in.clone(), k_in.clone())

# the cos_sin cache actually used (bf16, indexed by positions) — dump slice [0:Pmax]
Pmax = int(positions.max()) + 1
cache = rope._match_cos_sin_cache_dtype(q_in)[:Pmax].to(torch.float32).numpy()
dump("rope_cos_sin_cache_f32", cache, f"MRotaryEmbedding cos_sin_cache[0:{Pmax}] (bf16->f32) [Pmax,rotary_dim]")
dump("rotary_q_in_f32", q_in.to(torch.float32).numpy(), "seeded q [T, Hq*Dh] bf16->f32")
dump("rotary_k_in_f32", k_in.to(torch.float32).numpy(), "seeded k [T, Hkv*Dh] bf16->f32")
dump("rotary_q_out_f32", q_out.to(torch.float32).numpy(), "vLLM MRoPE q_out [T, Hq*Dh]")
dump("rotary_k_out_f32", k_out.to(torch.float32).numpy(), "vLLM MRoPE k_out [T, Hkv*Dh]")

# ===========================================================================
# 3. DeepStack scatter (_compute_deepstack_embeds), fp32 to isolate scatter math
# ===========================================================================
from vllm.model_executor.models.utils import _merge_multimodal_embeddings

# Reduced hidden (Ht) keeps the scatter fixtures tiny; the scatter/reshape/permute
# logic is H-independent (a generic loop over H), so Ht=16 catches index/stride
# bugs identically. (Production merge/deepstack run at the real H=2560.)
Ht = 16
gd = torch.Generator().manual_seed(7)
multiscale = torch.randn(N, n_deepstack * Ht, generator=gd, dtype=torch.float32)
# mirror _compute_deepstack_embeds body (single image => one item)
inputs_embeds_dummy = torch.zeros(T, Ht, dtype=torch.float32)
is_mm_t = torch.from_numpy(is_multimodal)
deepstack_input_embeds = inputs_embeds_dummy.new_zeros(T, n_deepstack * Ht)
deepstack_input_embeds = _merge_multimodal_embeddings(
    inputs_embeds=deepstack_input_embeds,
    multimodal_embeddings=[multiscale],
    is_multimodal=is_mm_t)
deepstack_input_embeds = deepstack_input_embeds.view(T, n_deepstack, Ht).permute(1, 0, 2)  # [3,T,Ht]
dump("deepstack_multiscale_f32", multiscale.numpy(), "seeded multiscale [N, n_deepstack*H]")
dump("deepstack_out_f32", deepstack_input_embeds.contiguous().numpy(), "[n_deepstack, T, H] scattered")

# ===========================================================================
# 4. embed-merge (_merge_multimodal_embeddings masked scatter)
# ===========================================================================
gm = torch.Generator().manual_seed(11)
main_embeds = torch.randn(N, Ht, generator=gm, dtype=torch.float32)
text_embeds = torch.randn(T, Ht, generator=gm, dtype=torch.float32)
merged = _merge_multimodal_embeddings(
    inputs_embeds=text_embeds.clone(),
    multimodal_embeddings=[main_embeds],
    is_multimodal=is_mm_t)
dump("merge_main_f32", main_embeds.numpy(), "seeded visual main embeds [N,H]")
dump("merge_text_f32", text_embeds.numpy(), "seeded text embeds [T,H]")
dump("merge_out_f32", merged.numpy(), "_merge_multimodal_embeddings result [T,H]")

manifest["T"] = T
manifest["N_visual"] = N
manifest["test_dims"] = {"Hq_t": Hq_t, "Hkv_t": Hkv_t, "Dh": Dh, "Ht": Ht}
manifest["is_multimodal_first_last"] = [int(np.argmax(is_multimodal)),
                                        int(len(is_multimodal) - 1 - np.argmax(is_multimodal[::-1]))]
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("WROTE manifest.json; all M2b/M2c references dumped.", flush=True)
