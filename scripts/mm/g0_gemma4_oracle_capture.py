#!/usr/bin/env python3
# Gemma-4 W0 — the DECISIVE oracle-gateability run per
# [[oracle-gateability-model-runs-not-config-constructs]]: prove the pinned vLLM
# 0.25.0 oracle does not merely CONSTRUCT the Gemma-4 config but LOADS the weights
# onto GPU, RUNS a forward, and GENERATES coherent tokens. Vehicle:
# `unsloth/gemma-4-E4B-it` (`Gemma4ForConditionalGeneration`, ungated, smallest
# full mm variant with vision+audio+text configs). TEXT-ONLY greedy generation is
# sufficient for W0 — it still exercises the PLE/YOCO/Gemma-4-MoE backbone + proves
# the load; an image prompt is attempted opportunistically if the text path runs.
#
# Captures the greedy golden (fixed prompt + exact output token ids) + K-run
# self-determinism (the gate-form decision per [[near-tie-distributional-gate]]).
# If the oracle ABORTS (like OLMo-3's nested-rope KeyError), the traceback is the
# honest "constructs-but-doesn't-run" result that keeps Gemma-4 blocked.
#
# vLLM 0.25.0 forces `spawn` (CUDA init) which re-imports the module in the worker,
# so the whole body MUST sit under `if __name__ == "__main__":`.
#
# Env: CKPT_DIR (snapshot path), OUT_DIR, K (default 5), N (max_tokens, default 32),
#      GMU (gpu_memory_utilization, default 0.30 — E4B ~4B active, keep LOW so the
#      shared GB10 box never OOM-reboots).
import hashlib
import json
import os
import traceback


def main():
    CKPT = os.environ["CKPT_DIR"]
    OUT = os.environ["OUT_DIR"]
    K = int(os.environ.get("K", "5"))
    N = int(os.environ.get("N", "32"))
    GMU = float(os.environ.get("GMU", "0.30"))
    os.makedirs(OUT, exist_ok=True)

    import vllm
    import transformers
    versions = {"vllm": vllm.__version__, "transformers": transformers.__version__}
    print("VERSIONS", versions, flush=True)

    from vllm import LLM, SamplingParams

    result = {"stage": "load", "ok": False, "versions": versions,
              "model_id": "unsloth/gemma-4-E4B-it",
              "arch": "Gemma4ForConditionalGeneration"}
    try:
        llm = LLM(model=CKPT, tokenizer=CKPT, trust_remote_code=True,
                  dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=GMU,
                  max_model_len=4096)
        result["stage"] = "loaded"
        print("LLM LOADED OK", flush=True)
    except Exception as e:
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        print("LLM LOAD ABORTED (constructs-but-doesn't-run):\n" + result["traceback"], flush=True)
        with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
            json.dump(result, f, indent=2)
        raise SystemExit(3)

    # TEXT-ONLY greedy via the chat template (real serving path).
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content":
                 "Explain what a large language model is in one sentence."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("PROMPT:\n", prompt, flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=N)
    runs, texts = [], []
    for i in range(K):
        out = llm.generate([{"prompt": prompt}], sp)
        ids = list(out[0].outputs[0].token_ids)
        txt = out[0].outputs[0].text
        runs.append(ids)
        texts.append(txt)
        print(f"run {i}: {len(ids)} toks ids[:16]={ids[:16]}", flush=True)
        print(f"        text={txt!r}", flush=True)

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

    # input ids (placeholder-expanded / real prompt ids)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    out1 = llm.generate([{"prompt": prompt}], sp1)
    in_ids = list(out1[0].prompt_token_ids)
    print(f"INPUT prompt_token_ids len={len(in_ids)}", flush=True)

    result.update({
        "stage": "generated", "ok": True, "ran_text": True, "ran_image": False,
        "K": K, "max_tokens": N, "gpu_memory_utilization": GMU,
        "deterministic": identical, "first_divergence_pos": div, "gate_form": gate,
        "prompt": prompt, "prompt_token_ids": in_ids,
        "ref_token_ids": ref, "ref_text": texts[0],
        "gen_tokens_sha256":
            hashlib.sha256(bytes(str(ref), "utf-8")).hexdigest(),
        "all_runs": runs,
    })
    with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE gen_manifest.json — LOADS+RUNS+GENERATES CONFIRMED", flush=True)


if __name__ == "__main__":
    main()
