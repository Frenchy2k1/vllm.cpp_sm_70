#!/usr/bin/env python3
"""G3 reference dump for the Gemma-4 USM-Conformer AUDIO TOWER (`Gemma4AudioModel`)
+ the audio `Gemma4MultimodalEmbedder` (`embed_audio`), from `unsloth/gemma-4-E4B-it`.

Constructs ONLY the audio tower (no 15 GB text backbone) from the checkpoint's
`audio_config`, loads the `model.audio_tower.*` + `model.embed_audio.*` weights
from the single safetensors shard, runs a fixed deterministic synthetic waveform
through the checkpoint's real `Gemma4AudioFeatureExtractor` to get the golden
`input_features` [1,T,128] + `input_features_mask` [1,T], and dumps:

  input_features_f32      [T,128]     the golden tower INPUT (A1's job to reproduce)
  input_features_mask_i32 [T]         valid=1 (all valid here — single clip, no pad)
  subsample_out_f32       [S,1024]    after the 2x Conv2d subsample + input_proj_linear
  position_embeddings_f32 [P,1024]    rel_pos_enc(hidden) (P=context_size//2+1)
  block{0,mid,last}_f32   [S,1024]    after Gemma4AudioLayer {0, N//2, N-1}
  output_proj_f32         [S,1536]    last_hidden_state = output_proj(hidden)
  projected_f32           [S,2560]    embed_audio(last_hidden_state) — merge input

Everything runs in FLOAT32 on CPU (the tower is small; no GPU / no flock). The C++
tower (src/vllm/model_executor/models/gemma4_audio.cpp) is gated f32-vs-f32 so the
USM-Conformer MATH is the parity variable (a wrong chunk/rel-shift/glu/softcap
drives a stage far past its band). Production dtype is bf16 (device-resident bf16
forward = the perf follow-on; correctness-first here, mirroring G2-impl's cadence).

Weights (fp32, ~90 MiB, NOT committed) are dumped to $WEIGHT_OUT keyed by the
tower-relative tensor name; the C++ test reads them via VLLM_GEMMA4_AUDIO_WEIGHTS.
Per-stage refs + the manifest (clamp bounds, config, sha256) land in
tests/parity/goldens/gemma4_e4b_audio/audio_refs/.

Grounds: transformers models/gemma4/modeling_gemma4.py @ 5.13.1
  Gemma4AudioModel.forward:1985-2014, Gemma4AudioSubSampleConvProjection:385-412,
  Gemma4AudioLayer:525-573, Gemma4AudioAttention:249-354 (chunked local + rel-shift
  + softcap + per_dim_scale), Gemma4AudioLightConv1d:484-522 (GLU + depthwise causal
  conv), Gemma4AudioFeedForward:415-447 (half-step residual 0.5), Gemma4RMSNorm:197,
  Gemma4AudioRelPositionalEncoding:218-246; vllm gemma4_mm.py:908-960,1473-1497.
"""
import glob
import hashlib
import json
import math
import os

import numpy as np
import torch

from transformers import AutoFeatureExtractor, Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
from safetensors import safe_open

MODEL = "unsloth/gemma-4-E4B-it"
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/*"))[0]
WEIGHT_OUT = os.path.expanduser(os.environ.get("WEIGHT_OUT", "~/gemma4_audio_weights"))
OUT = os.path.expanduser(os.environ.get(
    "OUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "tests", "parity", "goldens", "gemma4_e4b_audio", "audio_refs")))
os.makedirs(WEIGHT_OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


captures = {}


def save(name, t, out=OUT):
    a = t.detach().to(torch.float32).cpu().contiguous().numpy()
    a.tofile(os.path.join(out, name + ".bin"))
    captures[name] = {"shape": list(a.shape), "sha256": sha256_hex(a.tobytes())}
    print(f"  dumped {name} shape={list(a.shape)} sha256={captures[name]['sha256'][:12]}",
          flush=True)


# ---- build the audio tower ONLY (no text backbone) -----------------------------
full_cfg = Gemma4Config.from_pretrained(MODEL)
acfg = full_cfg.audio_config
text_hidden = full_cfg.text_config.hidden_size
tower = Gemma4AudioModel(acfg).eval().to(torch.float32)

# ---- load audio_tower.* + embed_audio.* weights from the single shard ----------
shard = sorted(glob.glob(os.path.join(SNAP, "*.safetensors")))[0]
sd = {}
embed_proj_w = None
with safe_open(shard, framework="pt") as f:
    for k in f.keys():
        if k.startswith("model.audio_tower."):
            sd[k[len("model.audio_tower."):]] = f.get_tensor(k).to(torch.float32)
        elif k == "model.embed_audio.embedding_projection.weight":
            embed_proj_w = f.get_tensor(k).to(torch.float32)
missing, unexpected = tower.load_state_dict(sd, strict=False)
# non-persistent buffers (inv_timescales, softcap) are built in __init__; they are
# the only allowed "missing" entries.
real_missing = [m for m in missing if not (m.endswith("inv_timescales")
                                           or m.endswith("softcap"))]
assert not real_missing, f"missing tower weights: {real_missing}"
assert not unexpected, f"unexpected tower weights: {unexpected}"
assert embed_proj_w is not None, "embed_audio.embedding_projection.weight not found"
print(f"loaded audio tower: {len(sd)} tensors, embed_proj {list(embed_proj_w.shape)}",
      flush=True)

# ---- dump every tower weight (fp32) keyed by tower-relative name ----------------
nw = 0
for k, v in tower.state_dict().items():
    if k.endswith("inv_timescales") or k.endswith("softcap"):
        continue  # non-persistent, C++ recomputes from config
    v.to(torch.float32).contiguous().numpy().tofile(os.path.join(WEIGHT_OUT, k + ".bin"))
    nw += 1
embed_proj_w.contiguous().numpy().tofile(
    os.path.join(WEIGHT_OUT, "embed_audio.embedding_projection.weight.bin"))
nw += 1
print(f"dumped {nw} weight tensors to {WEIGHT_OUT}", flush=True)

# ---- fixed deterministic input: synth waveform -> real feature extractor --------
fe = AutoFeatureExtractor.from_pretrained(MODEL)
sr = int(fe.sampling_rate)
n = int(2.5 * sr)  # 2.5 s
t = np.arange(n, dtype=np.float64) / sr
wave = (0.6 * np.sin(2 * math.pi * 220.0 * t)
        + 0.3 * np.sin(2 * math.pi * 440.0 * t)
        + 0.1 * np.sin(2 * math.pi * 880.0 * t)).astype(np.float32)
feats = fe([wave], sampling_rate=sr, return_tensors="pt")
input_features = feats["input_features"].to(torch.float32)          # [1,T,128]
input_features_mask = feats["input_features_mask"]                   # [1,T] bool
T = int(input_features.shape[1])
F = int(input_features.shape[2])
save("input_features_f32", input_features[0])
np.asarray(input_features_mask[0].to(torch.int32)).tofile(
    os.path.join(OUT, "input_features_mask_i32.bin"))
print(f"input_features T={T} F={F} valid={int(input_features_mask.sum())}", flush=True)

# ---- per-stage taps via forward hooks ------------------------------------------
taps = {}
N = int(acfg.num_hidden_layers)
tap_layers = {0: "block0", N // 2: "block_mid", N - 1: "block_last"}


def mk_layer_hook(name):
    def hook(mod, inp, out):
        taps[name] = out.detach().clone()
    return hook


tower.subsample_conv_projection.register_forward_hook(
    lambda m, i, o: taps.__setitem__("subsample", o[0].detach().clone()))
tower.rel_pos_enc.register_forward_hook(
    lambda m, i, o: taps.__setitem__("posemb", o.detach().clone()))
for li, nm in tap_layers.items():
    tower.layers[li].register_forward_hook(mk_layer_hook(nm))

out = tower(input_features, input_features_mask)
last_hidden = out.last_hidden_state         # [1,S,1536]
audio_mask = out.attention_mask             # [1,S] bool
S = int(last_hidden.shape[1])

save("subsample_out_f32", taps["subsample"][0])
save("position_embeddings_f32", taps["posemb"][0])
save("block0_f32", taps["block0"][0])
save("block_mid_f32", taps["block_mid"][0])
save("block_last_f32", taps["block_last"][0])
save("output_proj_f32", last_hidden[0])

# ---- projector: RMSNorm(no-weight, eps) -> Linear(1536->text_hidden, no bias) ---
eps = float(acfg.rms_norm_eps)
x = last_hidden.float()
var = x.pow(2).mean(-1, keepdim=True) + eps
normed = x * torch.pow(var, -0.5)
projected = normed @ embed_proj_w.t()       # [1,S,text_hidden]
save("projected_f32", projected[0])

manifest = {
    "model_id": MODEL,
    "vehicle": "unsloth/gemma-4-E4B-it Gemma4AudioModel (USM-Conformer) + embed_audio, "
               "transformers 5.13.1, CPU fp32",
    "ref_dtype": "float32",
    "note": "audio tower constructed from audio_config; weights loaded from the single "
            "safetensors shard; input_features from the checkpoint feature extractor on a "
            "fixed synthetic waveform; f32-vs-f32 tower-math gate",
    "config": {
        "hidden_size": int(acfg.hidden_size),
        "num_hidden_layers": N,
        "num_attention_heads": int(acfg.num_attention_heads),
        "head_dim": int(acfg.hidden_size // acfg.num_attention_heads),
        "output_proj_dims": int(acfg.output_proj_dims),
        "conv_kernel_size": int(acfg.conv_kernel_size),
        "subsampling_conv_channels": list(acfg.subsampling_conv_channels),
        "attention_chunk_size": int(acfg.attention_chunk_size),
        "attention_context_left": int(acfg.attention_context_left),
        "attention_context_right": int(acfg.attention_context_right),
        "attention_logit_cap": float(acfg.attention_logit_cap),
        "attention_invalid_logits_value": float(acfg.attention_invalid_logits_value),
        "gradient_clipping": float(acfg.gradient_clipping),
        "residual_weight": float(acfg.residual_weight),
        "rms_norm_eps": eps,
        "hidden_act": acfg.hidden_act,
        "use_clipped_linears": bool(acfg.use_clipped_linears),
        "feature_size": F,
        "text_hidden_size": int(text_hidden),
        "T": T, "S": S,
    },
    "tap_layers": {str(k): v for k, v in tap_layers.items()},
    "stages": captures,
    "upstream": "transformers modeling_gemma4.py:1985-2014,385-412,525-573,249-354,"
                "484-522,415-447,197,218-246 @ 5.13.1; vllm gemma4_mm.py:908-960,1473-1497",
}
with open(os.path.join(OUT, "audio_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("=== G3 AUDIO TOWER REFERENCE OK ===")
print("T,S =", T, S, " head_dim =", manifest["config"]["head_dim"])
print("output_proj[0,:5] :", last_hidden.float()[0, 0, :5].tolist())
print("projected[0,:5]   :", projected.float()[0, 0, :5].tolist())
print("WEIGHT_OUT :", WEIGHT_OUT)
print("OUT        :", OUT)
