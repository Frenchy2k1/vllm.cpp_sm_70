// Ported from: vllm/parser/engine/registered_adapters.py + the
// ReasoningParserManager / ToolParserManager name registration that binds an
// engine-backed family to its ParserEngineConfig @ 555967922 (vLLM 0.26.0.dev0).
//
// The unified registry: a parser format name -> its declarative
// ParserEngineConfig. This is the single dispatch surface that replaces the
// per-family hand-rolled streaming parsers for engine-backed formats.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "vllm/parser/engine/parser_engine_config.h"

namespace vllm::parser::engine {

// Returns the engine config for a registered format name, or nullopt if the
// name is not (yet) engine-backed.
std::optional<ParserEngineConfig> get_engine_config(const std::string& name);

bool is_engine_backed(const std::string& name);

// All engine-backed format names currently ported to the C++ engine.
std::vector<std::string> engine_backed_names();

}  // namespace vllm::parser::engine
