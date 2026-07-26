#!/usr/bin/env python3
"""A3 near-tie-robust gate (ratified, exactly as M3c/M3d): teacher-force vLLM on
OUR pipeline's token sequence and PASS iff every position where our token differs
from vLLM's argmax is a genuine bf16 near-tie (vLLM logprob gap <= 0.5 nats).

This is the correct correctness bar when bit-exact is infeasible because our audio
ENCODER uses different bf16 GEMM/attn kernels than vLLM (cuBLASLt + FLASH_ATTN):
the decoder is proven token-exact (ref-audio -> 48/48); the only divergences are
audio-conditioned near-ties within the encoder's bf16-depth envelope."""
import json
import os
import wave

import numpy as np
from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import LLM, SamplingParams

MODEL = "mistralai/Voxtral-Mini-3B-2507"
OUT = os.path.expanduser("~/a3_fixture")

man = json.load(open(os.path.join(OUT, "voxtral_manifest.json")))
gold = json.load(open(os.path.join(OUT, "voxtral_golden.json")))["output_token_ids"]
our = [int(x) for x in open(os.path.join(OUT, "our_tokens.txt")).read().strip().split(",")]
prompt_ids = man["prompt_ids"]
print("our vs golden match:", sum(1 for a, b in zip(our, gold) if a == b), "/", len(gold))

with wave.open(os.path.join(OUT, "voxtral_input_16k_mono.wav"), "rb") as w:
    raw = w.readframes(w.getnframes())
wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
tok = MistralTokenizer.from_hf_hub(MODEL)
au = Audio(audio_array=wav, sampling_rate=16000, format="wav")
req = ChatCompletionRequest(
    messages=[UserMessage(content=[AudioChunk.from_audio(au), TextChunk(text=man["question"])])],
    model=MODEL)
enc = tok.encode_chat_completion(req)
audios = [(a.audio_array, a.sampling_rate) for a in enc.audios]

llm = LLM(model=MODEL, tokenizer_mode="mistral", config_format="mistral",
          load_format="mistral", max_model_len=8192, max_num_seqs=1,
          enforce_eager=True, enable_chunked_prefill=False,
          gpu_memory_utilization=0.30, limit_mm_per_prompt={"audio": 1})

# Teacher-force: full sequence = prompt + OUR tokens; prompt_logprobs gives, at each
# position, the logprob of the actual (our) token + the top-k (=> the argmax logprob).
full = list(prompt_ids) + list(our)
sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5)
out = llm.generate({"prompt_token_ids": full, "multi_modal_data": {"audio": audios}}, sp)
plp = out[0].prompt_logprobs  # list over prompt positions; [0] is None

base = len(prompt_ids)
worst = 0.0
worst_pos = -1
n_div = 0
fails = []
for i in range(len(our)):
    pos = base + i
    d = plp[pos]
    if d is None:
        continue
    our_lp = d[our[i]].logprob
    top_lp = max(e.logprob for e in d.values())
    gap = top_lp - our_lp
    is_argmax = gap <= 1e-9
    if not is_argmax:
        n_div += 1
        if gap > worst:
            worst, worst_pos = gap, i
        if gap > 0.5:
            fails.append((i, our[i], gap))
    extra = ""
    if i < len(gold) and our[i] != gold[i]:
        top5 = sorted(((e.logprob, t) for t, e in d.items()), reverse=True)[:5]
        gld = d.get(gold[i])
        extra = ("  golden_tok %d lp %s | top5=%s" %
                 (gold[i], ("%.4f" % gld.logprob) if gld else "NOT-IN-TOP5",
                  [(t, round(lp, 3)) for lp, t in top5]))
    print(f"pos {i:2d} tok {our[i]:6d} our_lp {our_lp:8.4f} top {top_lp:8.4f} gap {gap:7.4f}"
          + ("" if is_argmax else "  <-DIVERGE") + extra)

print("\n=== NEAR-TIE GATE ===")
print(f"divergent positions: {n_div}, worst gap {worst:.4f} nats @ pos {worst_pos}")
print(f"gate band: <= 0.5 nats; over-band failures: {len(fails)} -> {fails}")
print("RESULT:", "PASS" if len(fails) == 0 else "FAIL")
json.dump({"n_divergent": n_div, "worst_gap_nats": worst, "worst_pos": worst_pos,
           "over_band_failures": fails, "result": "PASS" if not fails else "FAIL",
           "our_tokens": our, "golden_tokens": gold},
          open(os.path.join(OUT, "voxtral_neartie.json"), "w"), indent=1)
