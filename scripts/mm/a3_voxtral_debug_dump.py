#!/usr/bin/env python3
"""A3 debug: dump vLLM's intermediate Voxtral tensors to isolate a token mismatch.

Monkeypatches VoxtralEncoderModel.forward (encoder output [n_enc, d_model]) and
VoxtralForConditionalGeneration.embed_multimodal (audio embeddings [n_tok, text_h])
so the C++ pipeline stages can be compared 1:1. Run under flock on GPU."""
import json
import os
import wave

import numpy as np
import torch

OUT = os.path.expanduser("~/a3_fixture")
MODEL = "mistralai/Voxtral-Mini-3B-2507"
SR = 16000

man = json.load(open(os.path.join(OUT, "voxtral_manifest.json")))
wav_path = os.path.join(OUT, "voxtral_input_16k_mono.wav")
with wave.open(wav_path, "rb") as w:
    raw = w.readframes(w.getnframes())
wav_f32 = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import LLM, SamplingParams
import vllm.model_executor.models.voxtral as vox

captures = {}

_orig_enc = vox.VoxtralEncoderModel.forward
def enc_fwd(self, input_features):
    out = _orig_enc(self, input_features)
    try:
        captures["encoder_out"] = out[0].detach().float().cpu().numpy().copy()
    except Exception as e:
        print("enc capture err", e)
    return out
vox.VoxtralEncoderModel.forward = enc_fwd

_orig_emb = vox.VoxtralForConditionalGeneration.embed_multimodal
def emb_mm(self, **kwargs):
    out = _orig_emb(self, **kwargs)
    try:
        arr = out[0] if isinstance(out, (list, tuple)) else out
        captures["audio_embeds"] = arr.detach().float().cpu().numpy().copy()
    except Exception as e:
        print("emb capture err", e)
    return out
vox.VoxtralForConditionalGeneration.embed_multimodal = emb_mm

# Capture the merged inputs_embeds fed to the Llama text model + its hidden output.
import vllm.model_executor.models.llama as llama_mod
_orig_llama = llama_mod.LlamaModel.forward
def llama_fwd(self, input_ids, positions, *a, **kw):
    ie = kw.get("inputs_embeds", None)
    if ie is None:
        for x in a:
            if hasattr(x, "shape") and x.dim() == 2 and x.shape[-1] == self.config.hidden_size:
                ie = x
    out = _orig_llama(self, input_ids, positions, *a, **kw)
    h = out[0] if isinstance(out, tuple) else out
    # keep the PREFILL call (largest T)
    T = h.shape[0]
    if T > captures.get("_prefill_T", 0):
        captures["_prefill_T"] = T
        if ie is not None:
            captures["merged_embeds"] = ie.detach().float().cpu().numpy().copy()
        captures["text_hidden"] = h.detach().float().cpu().numpy().copy()
    return out
llama_mod.LlamaModel.forward = llama_fwd

tokenizer = MistralTokenizer.from_hf_hub(MODEL)
audio = Audio(audio_array=wav_f32.astype(np.float32), sampling_rate=SR, format="wav")
chunk = AudioChunk.from_audio(audio)
req = ChatCompletionRequest(
    messages=[UserMessage(content=[chunk, TextChunk(text=man["question"])])], model=MODEL
)
enc = tokenizer.encode_chat_completion(req)
prompt_ids = list(enc.tokens)
audios_and_sr = [(a.audio_array, a.sampling_rate) for a in enc.audios]

llm = LLM(model=MODEL, tokenizer_mode="mistral", config_format="mistral",
          load_format="mistral", max_model_len=8192, max_num_seqs=1,
          enforce_eager=True, enable_chunked_prefill=False,
          gpu_memory_utilization=0.30, limit_mm_per_prompt={"audio": 1})
sp = SamplingParams(temperature=0.0, max_tokens=4)
out = llm.generate({"prompt_token_ids": prompt_ids,
                    "multi_modal_data": {"audio": audios_and_sr}}, sp)
print("gen toks:", list(out[0].outputs[0].token_ids))

captures.pop("_prefill_T", None)
for k, v in captures.items():
    print(k, v.shape, "sha", __import__("hashlib").sha256(
        np.ascontiguousarray(v, dtype="<f4").tobytes()).hexdigest()[:12])
    np.ascontiguousarray(v, dtype="<f4").tofile(os.path.join(OUT, f"dbg_{k}.bin"))
print("DEBUG DUMP OK")
