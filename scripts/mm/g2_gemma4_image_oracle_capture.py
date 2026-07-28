#!/usr/bin/env python3
# Gemma-4 G2 — IMAGE->text oracle capture + staged vision-tower reference dump.
#
# Proves the pinned vLLM 0.25.0 oracle RUNS an IMAGE prompt through
# `unsloth/gemma-4-E4B-it` (Gemma4ForConditionalGeneration) and captures:
#   (1) the greedy image->text golden token ids + K-run self-determinism
#       (the gate-form decision per [[near-tie-distributional-gate]]);
#   (2) the EXACT Gemma4 processor outputs (pixel_values, pixel_position_ids,
#       prompt_token_ids) for the FIXED committed image — the bit-exact input
#       the C++ NaFlex image processor must reproduce;
#   (3) best-effort staged vision-tower reference tensors via forward hooks on
#       the vLLM-loaded tower (patch_embedder / encoder / pooler / embed_vision)
#       for future per-stage unit-gating (the M2a playbook).
#
# vLLM 0.25.0 forces `spawn`; the whole body sits under __main__.
#
# Env: CKPT_DIR (snapshot path OR hub id), OUT_DIR, IMG (fixed png path),
#      K (default 5), N (max_tokens, default 32), GMU (default 0.30 — keep LOW
#      so the shared GB10 unified-memory box never OOM-reboots).
import hashlib
import json
import os
import traceback

import numpy as np


def _sha(obj) -> str:
    return hashlib.sha256(bytes(str(obj), "utf-8")).hexdigest()


def main():
    CKPT = os.environ["CKPT_DIR"]
    OUT = os.environ["OUT_DIR"]
    IMG = os.environ["IMG"]
    K = int(os.environ.get("K", "5"))
    N = int(os.environ.get("N", "32"))
    GMU = float(os.environ.get("GMU", "0.30"))
    os.makedirs(OUT, exist_ok=True)
    ref_dir = os.path.join(OUT, "vision_refs")
    os.makedirs(ref_dir, exist_ok=True)

    import torch
    import transformers
    import vllm
    from PIL import Image

    versions = {"vllm": vllm.__version__, "transformers": transformers.__version__}
    print("VERSIONS", versions, flush=True)

    from vllm import LLM, SamplingParams

    img = Image.open(IMG).convert("RGB")
    img_sha = hashlib.sha256(np.asarray(img).tobytes()).hexdigest()
    print(f"IMG {IMG} size={img.size} array_sha256={img_sha[:16]}", flush=True)

    result = {
        "stage": "load", "ok": False, "versions": versions,
        "model_id": "unsloth/gemma-4-E4B-it",
        "arch": "Gemma4ForConditionalGeneration",
        "modality": "image",
        "image_path": os.path.basename(IMG),
        "image_size": list(img.size),
        "image_array_sha256": img_sha,
    }

    try:
        llm = LLM(model=CKPT, tokenizer=CKPT, trust_remote_code=True,
                  dtype="bfloat16", enforce_eager=True,
                  gpu_memory_utilization=GMU, max_model_len=4096,
                  limit_mm_per_prompt={"image": 1})
        result["stage"] = "loaded"
        print("LLM LOADED OK", flush=True)
    except Exception as e:
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        print("LLM LOAD ABORTED:\n" + result["traceback"], flush=True)
        with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
            json.dump(result, f, indent=2)
        raise SystemExit(3)

    tok = llm.get_tokenizer()
    # Real serving chat path with one image placeholder.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    print("PROMPT:\n", prompt, flush=True)

    mm = {"image": img}

    # ---- best-effort forward hooks on the vLLM-loaded vision tower ----
    captured = {}

    def _grab(name):
        def hook(_m, _inp, out):
            t = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if isinstance(t, tuple):
                t = t[0]
            try:
                captured[name] = t.detach().float().cpu().numpy()
            except Exception as ex:  # noqa: BLE001
                captured[name] = f"<uncapturable: {ex!r}>"
        return hook

    hook_handles = []
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = runner.model
        vt = model.vision_tower
        for sub, nm in [
            (vt.patch_embedder, "patch_embedder"),
            (vt.encoder, "encoder_last_hidden"),
            (vt.pooler, "pooler"),
            (model.embed_vision, "embed_vision_proj"),
        ]:
            hook_handles.append(sub.register_forward_hook(_grab(nm)))
        print("HOOKS registered on vision tower", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"HOOK setup failed (token golden still valid): {ex!r}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=N)
    runs, texts = [], []
    for i in range(K):
        out = llm.generate([{"prompt": prompt, "multi_modal_data": mm}], sp)
        ids = list(out[0].outputs[0].token_ids)
        txt = out[0].outputs[0].text
        runs.append(ids)
        texts.append(txt)
        print(f"run {i}: {len(ids)} toks ids[:16]={ids[:16]}", flush=True)
        print(f"        text={txt!r}", flush=True)

    for h in hook_handles:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass

    ref = runs[0]
    identical = all(r == ref for r in runs)
    div = None
    for pos in range(max(len(r) for r in runs)):
        vals = {(r[pos] if pos < len(r) else None) for r in runs}
        if len(vals) > 1:
            div = pos
            break
    gate = "STRICT" if identical else "NEAR-TIE (distributional)"
    print(f"\nK={K} DETERMINISTIC={identical} first_divergence_pos={div}", flush=True)
    print(f"GATE FORM (by measurement) = {gate}", flush=True)

    # prompt_token_ids (placeholder-expanded real prompt ids)
    out1 = llm.generate([{"prompt": prompt, "multi_modal_data": mm}],
                        SamplingParams(temperature=0.0, max_tokens=1))
    in_ids = list(out1[0].prompt_token_ids)
    print(f"INPUT prompt_token_ids len={len(in_ids)}", flush=True)

    # ---- exact processor outputs for input reproduction (no GPU needed) ----
    proc_meta = {}
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
        pf = proc(images=[img], text=prompt, return_tensors="pt")
        for k in ("pixel_values", "image_position_ids", "pixel_position_ids"):
            if k in pf:
                arr = pf[k].cpu().numpy()
                np.save(os.path.join(ref_dir, f"proc_{k}.npy"), arr)
                proc_meta[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
                print(f"PROC {k} shape={arr.shape} dtype={arr.dtype}", flush=True)
    except Exception as ex:  # noqa: BLE001
        proc_meta["error"] = repr(ex)
        print(f"processor dump failed: {ex!r}", flush=True)

    # ---- persist staged vision refs ----
    ref_meta = {}
    for name, arr in captured.items():
        if isinstance(arr, np.ndarray):
            np.save(os.path.join(ref_dir, f"{name}.npy"), arr)
            ref_meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                              "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
            print(f"REF {name} shape={arr.shape}", flush=True)
        else:
            ref_meta[name] = str(arr)

    result.update({
        "stage": "generated", "ok": True, "ran_image": True,
        "K": K, "max_tokens": N, "gpu_memory_utilization": GMU,
        "deterministic": identical, "first_divergence_pos": div, "gate_form": gate,
        "prompt": prompt, "prompt_token_ids": in_ids,
        "prompt_token_ids_len": len(in_ids),
        "ref_token_ids": ref, "ref_text": texts[0],
        "gen_tokens_sha256": _sha(ref),
        "all_runs": runs,
        "processor_outputs": proc_meta,
        "vision_refs": ref_meta,
    })
    with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE gen_manifest.json — IMAGE LOADS+RUNS+GENERATES CONFIRMED", flush=True)


if __name__ == "__main__":
    main()
