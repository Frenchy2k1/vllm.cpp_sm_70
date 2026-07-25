#!/usr/bin/env python3
"""M0 oracle reference capture for the multimodal input pipeline.

Drives vLLM 0.25.0 BaseMultiModalProcessor.apply on a FIXED (image, prompt) pair
for Qwen/Qwen3-VL-4B-Instruct and dumps the GOLDEN fixtures that M1's C++ processor
must reproduce bit-for-bit:
  - the deterministic input image as a raw uint8 HWC array (zero decode ambiguity)
  - pixel_values (float32) + image_grid_thw (int64)
  - the pre-expansion prompt token ids (plain tokenize, single <|image_pad|>)
  - the placeholder-EXPANDED prompt token ids (apply output)
  - the MultiModalHasher mm-hash string
  - a manifest with shapes/dtypes/content-hashes/model_id

Fixture image is 448x448 (a multiple of patch_size*merge_size = 32, area within
[65536, 16777216]) so smart_resize is identity and the bicubic resize is a numeric
no-op => pixel_values is a pure rescale+normalize+patchify, deterministically
bit-matchable in C++.
"""
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
OUT = os.path.expanduser("~/mm_fixture")
os.makedirs(OUT, exist_ok=True)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- fixture image
H = W = 448
rng = np.random.RandomState(12345)
arr = rng.randint(0, 256, size=(H, W, 3), dtype=np.uint8)  # HWC RGB uint8
assert arr.flags.c_contiguous
img = Image.fromarray(arr, mode="RGB")
# np.asarray(img) must round-trip exactly (this is what MultiModalHasher hashes)
assert np.array_equal(np.asarray(img), arr)

raw_path = os.path.join(OUT, "image_rgb_uint8_448x448x3.bin")
arr.tofile(raw_path)
img.save(os.path.join(OUT, "image.png"))

# ------------------------------------------------------------------- processor
model_config = ModelConfig(model=MODEL, limit_mm_per_prompt={"image": 1})
processor = MULTIMODAL_REGISTRY.create_processor(model_config)
info = processor.info
tokenizer = info.get_tokenizer()
hf_config = info.get_hf_config()

image_token_id = int(hf_config.image_token_id)
vision_start_token_id = int(hf_config.vision_start_token_id)
vision_end_token_id = int(hf_config.vision_end_token_id)

prompt = "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
pre_ids = list(tokenizer.encode(prompt, add_special_tokens=False))

mm_data = {"image": img}
out = processor(
    prompt,
    mm_items=info.parse_mm_data(mm_data),
    hf_processor_mm_kwargs={},
)


def get(o, k):
    try:
        return o[k]
    except Exception:
        return getattr(o, k)


prompt_ids = list(get(out, "prompt_token_ids"))
mm_kwargs = get(out, "mm_kwargs")
mm_hashes = get(out, "mm_hashes")

item = mm_kwargs["image"][0]
pixel_values = item["pixel_values"].data
grid_thw = item["image_grid_thw"].data

pv = pixel_values.cpu().to(torch.float32).contiguous().numpy()  # bf16 values widened to f32
gt = grid_thw.cpu().to(torch.int64).contiguous().numpy()

pv.tofile(os.path.join(OUT, "pixel_values_f32.bin"))  # PRODUCTION golden (bf16-as-f32)
gt.tofile(os.path.join(OUT, "image_grid_thw_i64.bin"))

# Pre-cast float32 pixel_values (the HF processor output BEFORE vLLM's cast to
# model dtype). vLLM casts mm_kwargs to model dtype (bf16) in
# context.call_hf_processor->_postprocess_output; the C++ processor computes this
# float32 and then rounds to bf16 to reproduce the production golden above.
pvf = (
    info.get_hf_processor()(
        text=[prompt], images=[img], return_tensors="pt"
    )["pixel_values"]
    .cpu()
    .to(torch.float32)
    .contiguous()
    .numpy()
)
pvf.tofile(os.path.join(OUT, "pixel_values_precast_f32.bin"))
assert np.array_equal(
    torch.from_numpy(pvf).to(torch.bfloat16).to(torch.float32).numpy(), pv
), "bf16(precast) must equal production golden"

mm_hash = mm_hashes["image"][0]

# number of expanded image tokens
n_image_tokens = int(prompt_ids.count(image_token_id))
merge_size = int(info.get_hf_processor().image_processor.merge_size)

manifest = {
    "model_id": MODEL,
    "image": {
        "shape": list(arr.shape),
        "dtype": "uint8",
        "layout": "HWC_RGB",
        "sha256": sha256_hex(arr.tobytes()),
        "raw_file": "image_rgb_uint8_448x448x3.bin",
    },
    "pixel_values": {
        "shape": list(pv.shape),
        "dtype": "bfloat16_as_float32",
        "note": "production golden: bf16 values widened to f32 (model dtype cast)",
        "sha256": sha256_hex(pv.tobytes()),
        "file": "pixel_values_f32.bin",
    },
    "pixel_values_precast": {
        "shape": list(pvf.shape),
        "dtype": "float32",
        "note": "HF processor output before model-dtype cast",
        "sha256": sha256_hex(pvf.tobytes()),
        "file": "pixel_values_precast_f32.bin",
    },
    "pixel_contract": {
        "normalize": "v = (float(raw_u8) - 127.5f) / 127.5f  (fused rescale+normalize, float32)",
        "model_dtype": "bfloat16",
        "production_cast": "pixel_values = round_to_nearest_even_bf16(v)",
        "patch_row_index": "r = ((gh*(grid_w/merge) + gw)*merge + mh)*merge + mw",
        "patch_col_index": "k = ((c*temporal_patch_size + t)*patch_size + ph)*patch_size + pw ; t duplicates",
        "grid_source_h": "H = (gh*merge + mh)*patch_size + ph",
        "grid_source_w": "W = (gw*merge + mw)*patch_size + pw",
    },
    "image_grid_thw": {
        "shape": list(gt.shape),
        "dtype": "int64",
        "values": gt.tolist(),
        "file": "image_grid_thw_i64.bin",
    },
    "prompt": prompt,
    "pre_expansion_token_ids": pre_ids,
    "expanded_prompt_token_ids": prompt_ids,
    "expanded_len": len(prompt_ids),
    "n_image_tokens": n_image_tokens,
    "mm_hash": mm_hash,
    "config": {
        "image_token_id": image_token_id,
        "vision_start_token_id": vision_start_token_id,
        "vision_end_token_id": vision_end_token_id,
        "patch_size": int(info.get_hf_processor().image_processor.patch_size),
        "temporal_patch_size": int(
            info.get_hf_processor().image_processor.temporal_patch_size
        ),
        "merge_size": merge_size,
        "image_mean": list(info.get_hf_processor().image_processor.image_mean),
        "image_std": list(info.get_hf_processor().image_processor.image_std),
    },
}

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("=== M0 ORACLE CAPTURE OK ===")
print("model_id           :", MODEL)
print("image sha256       :", manifest["image"]["sha256"])
print("pixel_values shape :", pv.shape, pv.dtype, "sha256", manifest["pixel_values"]["sha256"])
print("image_grid_thw     :", gt.tolist())
print("pre_ids len        :", len(pre_ids))
print("expanded len       :", len(prompt_ids), "n_image_tokens", n_image_tokens)
print("merge_size         :", merge_size, "expected N =", int(gt.prod()) // (merge_size ** 2))
print("mm_hash            :", mm_hash)
print("pixel_values[0,:6] :", pv[0, :6].tolist())
print("OUT dir            :", OUT)
