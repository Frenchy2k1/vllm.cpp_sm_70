// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
//
// vllm.cpp: ADDED for the Whisper AUDIO ENCODER self-attention (multimodal-speed
// §17). Two things are new here relative to every other vendored FA-2 file in this
// directory, and both are deliberate:
//
//  1. head_dim 64 — the Whisper-large-v3 encoder width (d_model 1280 / 20 heads).
//     It is one of upstream's own head dims (flash_fwd_launch_template.h:180-199
//     `run_mha_fwd_hdim64`); only the explicit instantiation was missing here.
//  2. the NON-SPLIT `run_mha_fwd_` entry (upstream's `mha_fwd`) rather than
//     `run_mha_fwd_splitkv_dispatch`. Every prior caller in this tree attends over a
//     PAGED KV cache, which forces the split-KV kernel; the encoder is dense,
//     non-paged, single-request, q/k/v the same length — upstream's plain batch
//     forward, which is also what vLLM's own encoder dispatch reaches
//     (vllm/model_executor/models/whisper.py WhisperEncoderAttention:255 ->
//     forward:298-317 -> flash_attn_varlen_func, non-causal).
//
// NOTHING in the vendored template changed — this file only asks the existing
// generic `run_mha_fwd_hdim64<T, Is_causal>` for one more (Headdim, entry-point)
// pair, so the compiled 128 / 192 / 256 split-KV kernels every text and MLA path
// calls are byte-identical. Non-causal only: the encoder is bidirectional, and no
// causal hd-64 caller exists, so the causal instantiation is deliberately not built
// (it is one line away if one appears).
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
