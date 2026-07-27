// Ported from: vllm/parser/qwen3.py + vllm/parser/kimi_k2.py @ 555967922.
#include "vllm/parser/engine/configs.h"

namespace vllm::parser::engine {

using S = ParserState;
using E = EventType;

// qwen3.py:88 (qwen3_config).
ParserEngineConfig qwen3_config(bool thinking, std::string name,
                                std::string think_start, std::string think_end,
                                std::string tool_start, std::string tool_end) {
  const std::string FUNC_PREFIX = "<function=";
  const std::string FUNC_END = "</function>";
  const std::string PARAM_START = "<parameter=";
  const std::string PARAM_END = "</parameter>";

  ParserEngineConfig c;
  c.name = std::move(name);
  c.initial_state = thinking ? S::REASONING : S::CONTENT;
  c.terminals = {
      {"THINK_START", think_start}, {"THINK_END", think_end},
      {"TOOL_START", tool_start},   {"TOOL_END", tool_end},
      {"FUNC_PREFIX", FUNC_PREFIX}, {"FUNC_END", FUNC_END},
      {"PARAM_START", PARAM_START}, {"PARAM_END", PARAM_END},
      {"CLOSE_ANGLE", ">"},
  };
  c.token_id_terminals = {
      {"THINK_START", think_start}, {"THINK_END", think_end},
      {"TOOL_START", tool_start},   {"TOOL_END", tool_end},
  };
  c.transitions = {
      {{S::REASONING, "THINK_START"}, Transition(S::REASONING, {})},
      {{S::REASONING, "THINK_END"}, Transition(S::CONTENT, {E::REASONING_END})},
      {{S::CONTENT, "THINK_END"}, Transition(S::CONTENT, {})},
      {{S::REASONING, "TOOL_START"},
       Transition(S::TOOL_PREAMBLE, {E::REASONING_END, E::TOOL_CALL_START})},
      {{S::CONTENT, "TOOL_START"},
       Transition(S::TOOL_PREAMBLE, {E::REASONING_END, E::TOOL_CALL_START})},
      {{S::CONTENT, "FUNC_PREFIX"}, Transition(S::TOOL_NAME, {E::TOOL_CALL_START})},
      {{S::TOOL_PREAMBLE, "TOOL_END"}, Transition(S::CONTENT, {E::TOOL_CALL_END})},
      {{S::TOOL_PREAMBLE, "FUNC_PREFIX"}, Transition(S::TOOL_NAME, {})},
      {{S::TOOL_NAME, "CLOSE_ANGLE"}, Transition(S::TOOL_ARGS, {})},
      {{S::TOOL_NAME, "FUNC_END"}, Transition(S::TOOL_BETWEEN, {E::TOOL_CALL_END})},
      {{S::TOOL_ARGS, "FUNC_END"}, Transition(S::TOOL_BETWEEN, {E::TOOL_CALL_END})},
      {{S::TOOL_ARGS, "PARAM_START"}, Transition(S::TOOL_ARGS, {E::ARG_VALUE_CHUNK})},
      {{S::TOOL_ARGS, "PARAM_END"}, Transition(S::TOOL_ARGS, {E::ARG_VALUE_CHUNK})},
      {{S::TOOL_BETWEEN, "TOOL_END"}, Transition(S::CONTENT, {})},
      {{S::TOOL_BETWEEN, "TOOL_START"}, Transition(S::TOOL_PREAMBLE, {E::TOOL_CALL_START})},
      {{S::TOOL_BETWEEN, "FUNC_PREFIX"}, Transition(S::TOOL_NAME, {E::TOOL_CALL_START})},
  };
  c.tool_args_json = false;
  return c;
}

// kimi_k2.py:52 (kimi_k2_config).
ParserEngineConfig kimi_k2_config(bool thinking) {
  const std::string THINK_START = "<think>";
  const std::string THINK_END = "</think>";
  const std::string TOOL_SECTION_START = "<|tool_calls_section_begin|>";
  const std::string TOOL_SECTION_END = "<|tool_calls_section_end|>";
  const std::string TOOL_CALL_START = "<|tool_call_begin|>";
  const std::string TOOL_CALL_END = "<|tool_call_end|>";
  const std::string TOOL_ARG_START = "<|tool_call_argument_begin|>";

  ParserEngineConfig c;
  c.name = "kimi_k2";
  c.initial_state = thinking ? S::REASONING : S::CONTENT;

  if (thinking) {
    c.terminals["THINK_START"] = THINK_START;
    c.terminals["THINK_END"] = THINK_END;
    c.token_id_terminals["THINK_START"] = THINK_START;
    c.token_id_terminals["THINK_END"] = THINK_END;
  }
  c.terminals["TOOL_SECTION_START"] = TOOL_SECTION_START;
  c.terminals["TOOL_SECTION_END"] = TOOL_SECTION_END;
  c.terminals["TOOL_START"] = TOOL_CALL_START;
  c.terminals["TOOL_END"] = TOOL_CALL_END;
  c.terminals["ARG_START"] = TOOL_ARG_START;
  c.token_id_terminals["TOOL_SECTION_START"] = TOOL_SECTION_START;
  c.token_id_terminals["TOOL_SECTION_END"] = TOOL_SECTION_END;
  c.token_id_terminals["TOOL_START"] = TOOL_CALL_START;
  c.token_id_terminals["TOOL_END"] = TOOL_CALL_END;
  c.token_id_terminals["ARG_START"] = TOOL_ARG_START;

  if (thinking) {
    c.transitions[{S::REASONING, "THINK_START"}] = Transition(S::REASONING, {});
    c.transitions[{S::REASONING, "THINK_END"}] =
        Transition(S::CONTENT, {E::REASONING_END});
    c.transitions[{S::CONTENT, "THINK_END"}] = Transition(S::CONTENT, {});
  }
  c.transitions[{S::REASONING, "TOOL_SECTION_START"}] =
      Transition(S::TOOL_PREAMBLE, {E::REASONING_END});
  c.transitions[{S::CONTENT, "TOOL_SECTION_START"}] =
      Transition(S::TOOL_PREAMBLE, {});
  c.transitions[{S::TOOL_PREAMBLE, "TOOL_START"}] =
      Transition(S::TOOL_NAME, {E::TOOL_CALL_START});
  c.transitions[{S::TOOL_NAME, "ARG_START"}] = Transition(S::TOOL_ARGS, {});
  c.transitions[{S::TOOL_ARGS, "TOOL_END"}] =
      Transition(S::TOOL_BETWEEN, {E::TOOL_CALL_END});
  c.transitions[{S::TOOL_ARGS, "TOOL_SECTION_END"}] =
      Transition(S::TOOL_PREAMBLE, {E::TOOL_CALL_END});
  c.transitions[{S::TOOL_BETWEEN, "TOOL_START"}] =
      Transition(S::TOOL_NAME, {E::TOOL_CALL_START});
  c.transitions[{S::TOOL_PREAMBLE, "TOOL_SECTION_END"}] =
      Transition(S::TOOL_PREAMBLE, {});
  c.transitions[{S::TOOL_BETWEEN, "TOOL_SECTION_END"}] =
      Transition(S::TOOL_PREAMBLE, {});

  c.tool_args_json = true;
  return c;
}

}  // namespace vllm::parser::engine
