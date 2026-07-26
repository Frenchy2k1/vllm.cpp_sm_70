#!/usr/bin/env python3
"""Text-only Voxtral greedy: isolates the Mistral/Llama decoder from audio/merge.
Dumps the prompt token ids + greedy output ids so the C++ decoder can be compared
1:1 on a pure-text path."""
import json
import os

from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import LLM, SamplingParams

MODEL = "mistralai/Voxtral-Mini-3B-2507"
OUT = os.path.expanduser("~/a3_fixture")

tok = MistralTokenizer.from_hf_hub(MODEL)
req = ChatCompletionRequest(
    messages=[UserMessage(content=[TextChunk(text="Name three primary colors.")])],
    model=MODEL,
)
prompt_ids = list(tok.encode_chat_completion(req).tokens)
print("text prompt ids:", prompt_ids)

llm = LLM(model=MODEL, tokenizer_mode="mistral", config_format="mistral",
          load_format="mistral", max_model_len=8192, max_num_seqs=1,
          enforce_eager=True, enable_chunked_prefill=False, gpu_memory_utilization=0.30)
sp = SamplingParams(temperature=0.0, max_tokens=24)
out = llm.generate({"prompt_token_ids": prompt_ids}, sp)
toks = list(out[0].outputs[0].token_ids)
print("text greedy out:", toks)
print("text greedy decoded:", out[0].outputs[0].text)
json.dump({"prompt_ids": prompt_ids, "output_token_ids": toks,
           "text": out[0].outputs[0].text},
          open(os.path.join(OUT, "voxtral_textonly.json"), "w"), indent=1)
print("TEXTONLY OK")
