#!/usr/bin/env python3
"""A2 reference dump for the Whisper-class AUDIO ENCODER TOWER (audio-track A2).

Feeds the COMMITTED A1 log-mel `input_features` [80,3000] golden through the real
transformers `WhisperEncoder` (from `openai/whisper-small`) on CPU, and dumps:

  - the fixed sinusoidal `embed_positions.weight` [1500,768] as a GOLDEN CONSTANT
    (like A1's mel filterbank, so the encoder-block math is the only parity var);
  - per-stage encoder references, each widened to f32:
      post_conv   [1500,768]  gelu(conv2(gelu(conv1(features)))).transpose
      post_pos    [1500,768]  post_conv + embed_positions.weight
      block0_out  [1500,768]  after encoder layer 0
      final_ln    [1500,768]  after the final encoder layer_norm (== last_hidden_state)
  - a manifest with the encoder config, shapes/dtypes, sha256 content hashes.

NUMERIC CONTRACT: the encoder runs in bf16 (weights + activations), mirroring the
production model dtype and the M2a vision-tower methodology, so the ONLY parity
variable vs the C++ tower is the bf16-kernel-stack accumulation divergence (the
measured bf16-depth envelope). The reference dtype actually used is recorded in the
manifest (`ref_dtype`), so a torch-CPU-bf16 fallback is auditable.

Feature extraction is NOT re-run here; A1 already committed `input_features`. The
encoder runs on CPU: NO GPU / NO flock. Run in the oracle venv on dgx:
  ~/venvs/vllm-oracle/bin/python scripts/mm/a2_audio_encoder_ref.py
Fixtures land in $OUT (default ~/audio_enc_fixture), then are copied into
tests/vllm/multimodal/fixtures/whisper_audio/.

Grounds: transformers models/whisper/modeling_whisper.py @ 5.13.1
  WhisperEncoder.forward:641-721, WhisperEncoderLayer.forward:400-430,
  WhisperAttention.forward:298-368 (k_proj no-bias, q pre-scaled), sinusoids:54.
"""
import hashlib
import json
import os

import numpy as np
import torch

from transformers import WhisperConfig, WhisperModel

MODEL = "openai/whisper-small"
# The committed A1 input_features golden (this repo's fixtures dir), overridable.
FIX = os.environ.get(
    "A1_FIX_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "tests", "vllm", "multimodal", "fixtures", "whisper_audio"),
)
OUT = os.path.expanduser(os.environ.get("OUT_DIR", "~/audio_enc_fixture"))
os.makedirs(OUT, exist_ok=True)

torch.set_grad_enabled(False)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


cfg = WhisperConfig.from_pretrained(MODEL)
D = int(cfg.d_model)              # 768
NLAYERS = int(cfg.encoder_layers)  # 12
NHEADS = int(cfg.encoder_attention_heads)  # 12
FFN = int(cfg.encoder_ffn_dim)    # 3072
NMELS = int(cfg.num_mel_bins)     # 80
L = int(cfg.max_source_positions)  # 1500
NFRAMES = L * cfg.encoder_config.conv2_stride if hasattr(cfg, "encoder_config") else 3000
NFRAMES = 3000

# ---- load the encoder (weights fp32 in the checkpoint) then cast to bf16 --------
model = WhisperModel.from_pretrained(MODEL, dtype=torch.float32)
enc = model.encoder.eval()

REF_DTYPE = torch.bfloat16
try:
    enc = enc.to(REF_DTYPE)
    # smoke a tiny conv to confirm CPU bf16 conv is available before the real run.
    _ = torch.nn.functional.gelu(enc.conv1(torch.zeros(1, NMELS, 6, dtype=REF_DTYPE)))
except (RuntimeError, NotImplementedError) as e:  # pragma: no cover
    print(f"WARN: CPU bf16 conv unavailable ({e}); falling back to fp32 reference")
    enc = model.encoder.eval().to(torch.float32)
    REF_DTYPE = torch.float32

# ---- inputs: the committed A1 log-mel golden -----------------------------------
feats_f32 = np.fromfile(os.path.join(FIX, "input_features_f32.bin"),
                        dtype=np.float32).reshape(NMELS, NFRAMES)
features = torch.from_numpy(feats_f32).to(REF_DTYPE).unsqueeze(0)  # [1,80,3000]

captures = {}


def save(name, t):
    a = t.detach().to(torch.float32).cpu().contiguous().numpy()
    a.tofile(os.path.join(OUT, name + ".bin"))
    captures[name] = {"shape": list(a.shape), "sha256": sha256_hex(a.tobytes())}
    print(f"  dumped {name} shape={list(a.shape)} sha256={captures[name]['sha256'][:12]}",
          flush=True)


# ---- embed_positions.weight golden constant ------------------------------------
embed_pos = enc.embed_positions.weight  # [1500,768], fixed sinusoid
save("enc_embed_positions_f32", embed_pos)

# ---- stage taps ----------------------------------------------------------------
# post_conv = gelu(conv2(gelu(conv1(features)))).transpose  (before pos add).
e = torch.nn.functional.gelu(enc.conv1(features))
e = torch.nn.functional.gelu(enc.conv2(e))
post_conv = e.permute(0, 2, 1)[0]  # [1500,768]
save("enc_post_conv_f32", post_conv)

# Full forward with per-layer hidden states. encoder_states = (pre-L0, pre-L1, ...,
# pre-L11, post-final-LN); last_hidden_state = post-final-LN.
out = enc(features, output_hidden_states=True, return_dict=True)
enc_states = out.hidden_states
post_pos = enc_states[0][0]        # input to layer 0 == post_conv + embed_positions
block0 = enc_states[1][0]          # after encoder layer 0
final_ln = out.last_hidden_state[0]  # after final layer_norm
save("enc_post_pos_f32", post_pos)
save("enc_block0_f32", block0)
save("enc_final_ln_f32", final_ln)

# sanity: post_pos == post_conv + embed_positions (all bf16-rounded)
recon = (post_conv.to(REF_DTYPE) + embed_pos.to(REF_DTYPE)).to(torch.float32)
max_dev = float((recon - post_pos.to(torch.float32)).abs().max())

manifest = {
    "model_id": MODEL,
    "vehicle": "openai/whisper-small WhisperEncoder (transformers 5.13.1, CPU)",
    "ref_dtype": str(REF_DTYPE).replace("torch.", ""),
    "note": "cast fp32 checkpoint -> ref_dtype; input_features from the committed "
            "A1 golden; encoder-block math is the parity variable",
    "config": {
        "d_model": D, "encoder_layers": NLAYERS, "encoder_attention_heads": NHEADS,
        "encoder_ffn_dim": FFN, "num_mel_bins": NMELS, "max_source_positions": L,
        "n_frames": NFRAMES, "activation_function": cfg.activation_function,
        "layer_norm_eps": 1e-5, "attn_scaling": (D // NHEADS) ** -0.5,
        "k_proj_bias": False, "conv1": "Conv1d(80->768,k3,pad1,stride1)",
        "conv2": "Conv1d(768->768,k3,pad1,stride2)",
    },
    "stages": captures,
    "post_pos_reconstruction_maxdev": max_dev,
    "upstream": "transformers modeling_whisper.py:641-721,400-430,298-368,54 @ 5.13.1; "
                "vllm whisper.py:458,353,322,473-476 @ e24d1b24",
}
with open(os.path.join(OUT, "enc_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("=== A2 AUDIO ENCODER REFERENCE OK ===")
print("ref_dtype           :", manifest["ref_dtype"])
print("post_pos recon maxdev:", max_dev)
print("final_ln[0,:5]      :", final_ln.to(torch.float32)[0, :5].tolist())
print("OUT dir             :", OUT)
