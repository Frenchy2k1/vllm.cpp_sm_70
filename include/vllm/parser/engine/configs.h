// Ported from: vllm/parser/qwen3.py (qwen3_config), vllm/parser/kimi_k2.py
// (kimi_k2_config), vllm/parser/seed_oss.py (SeedOssParser grammar reuse),
// vllm/parser/minimax_m2.py (minimax_m2_config), vllm/parser/glm47_moe.py
// (glm47_moe_config), vllm/parser/deepseek_v4.py (deepseek_v4_config),
// vllm/parser/deepseek_v32.py (deepseek_v32_config), vllm/parser/nemotron_v3.py
// (nemotron_v3_config) @ 555967922 (vLLM 0.26.0.dev0).
//
// Declarative ParserEngineConfig builders for the engine-backed families. The
// streaming-relevant fields (terminals / transitions / content_events) plus the
// assembly-layer arg_converter + flags are populated here (the arg_converter is a
// std::function carried on the config, exactly like qwen3).
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

// minimax_m2.py:94 (minimax_m2_config). XML <invoke name=...>/<parameter name=...>
// tool calls; always initial REASONING; regex arg_converter; validate_tool_names.
ParserEngineConfig minimax_m2_config();

// glm47_moe.py:74 (glm47_moe_config). <tool_call>name<arg_key>k</arg_key>
// <arg_value>v</arg_value></tool_call>; regex arg_converter; validate_tool_names.
// The Glm47MoeParser subclass strips the function name (parser/glm47_moe.{h,cpp}).
ParserEngineConfig glm47_moe_config(bool thinking = true);

// deepseek_v4.py:125 (deepseek_v4_config). <think> reasoning + DSML tool_calls;
// regex _dsml_arg_converter (string="true|false" typed values); arg_structural
// chars={'>'}. thinking defaults False (mirrors upstream).
ParserEngineConfig deepseek_v4_config(bool thinking = false);

// deepseek_v32.py:51 (deepseek_v32_config). DSML function_calls wrapper, no
// reasoning tags; shares _dsml_arg_converter with deepseek_v4.
ParserEngineConfig deepseek_v32_config();

// nemotron_v3.py:34 (nemotron_v3_config). Same grammar as qwen3, with
// strip_trailing_reasoning_whitespace=True and a distinct name.
ParserEngineConfig nemotron_v3_config(bool thinking = true);

}  // namespace vllm::parser::engine
