#!/usr/bin/env python3
# A2 — dump the openai/whisper-small `encoder.*` weights to a directory of f32
# .bin files keyed by encoder-relative tensor name, for the C++ Whisper encoder
# tower unit gate. NOT committed (~130 MiB f32); the test reads it via
# VLLM_WHISPER_ENC_WEIGHTS. The C++ side rounds fp32 -> bf16 on upload, the exact
# rounding the bf16 reference (a2_audio_encoder_ref.py) used, so weights match
# bit-exactly and the block math is the only parity variable.
#
# Grounds: transformers WhisperEncoder submodule names (conv1/conv2/
# embed_positions/layers.N.{self_attn.{q,k,v,out}_proj, self_attn_layer_norm,
# final_layer_norm, fc1, fc2}/layer_norm). k_proj has NO bias.
import os

import numpy as np
import torch

from transformers import WhisperModel

MODEL = "openai/whisper-small"
OUT = os.environ["WEIGHT_OUT"]
os.makedirs(OUT, exist_ok=True)

torch.set_grad_enabled(False)
model = WhisperModel.from_pretrained(MODEL, dtype=torch.float32)
enc = model.encoder.eval()

n = 0
for name, p in enc.state_dict().items():
    t = p.to(torch.float32).contiguous().numpy()
    t.tofile(os.path.join(OUT, name + ".bin"))
    n += 1
print(f"dumped {n} encoder weight tensors to {OUT}", flush=True)
