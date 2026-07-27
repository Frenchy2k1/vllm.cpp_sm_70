// Ported from: vllm/parser/qwen3.py (qwen3_config), vllm/parser/kimi_k2.py
// (kimi_k2_config), vllm/parser/seed_oss.py (SeedOssParser grammar reuse)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// Declarative ParserEngineConfig builders for the engine-backed families. Only
// the streaming-relevant fields are populated (see parser_engine_config.h scope
// note); the arg_converter / assembly-layer flags live in the ParserEngine
// residual, not here.
#pragma once

#include <string>

#include "vllm/parser/engine/parser_engine_config.h"

namespace vllm::parser::engine {

// qwen3.py:88 (qwen3_config). Shared by seed_oss (qwen3.py grammar, different
// wrapper tokens).
ParserEngineConfig qwen3_config(bool thinking = true, std::string name = "qwen3",
                                std::string think_start = "<think>",
                                std::string think_end = "</think>",
                                std::string tool_start = "<tool_call>",
                                std::string tool_end = "</tool_call>");

// kimi_k2.py:52 (kimi_k2_config).
ParserEngineConfig kimi_k2_config(bool thinking = true);

}  // namespace vllm::parser::engine
