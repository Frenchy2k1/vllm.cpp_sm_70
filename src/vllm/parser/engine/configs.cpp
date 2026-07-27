// Ported from: vllm/parser/qwen3.py + vllm/parser/kimi_k2.py +
// vllm/parser/minimax_m2.py + vllm/parser/glm47_moe.py + vllm/parser/deepseek_v4.py
// + vllm/parser/deepseek_v32.py + vllm/parser/nemotron_v3.py @ 555967922.
#include "vllm/parser/engine/configs.h"

#include <algorithm>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/parser/engine/py_json.h"

namespace vllm::parser::engine {

using S = ParserState;
using E = EventType;

namespace {

// Python str.strip() over ASCII whitespace (parser values are ASCII/UTF-8).
bool is_py_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && is_py_space(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && is_py_space(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// qwen3.py:59 _trim_wrapping_newlines — strip one leading + one trailing '\n'.
std::string trim_wrapping_newlines(std::string value) {
  if (!value.empty() && value.front() == '\n') value.erase(value.begin());
  if (!value.empty() && value.back() == '\n') value.pop_back();
  return value;
}

// qwen3.py:50 _PARAM_RE (re.DOTALL). `.` -> [\s\S] for the value group so it
// spans newlines exactly like the Python DOTALL flag.
const std::regex& qwen3_param_re() {
  static const std::regex re(
      R"(<\s*parameter\s*=\s*([^>]*)>([\s\S]*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=)))");
  return re;
}

// qwen3.py:56 _PARTIAL_PARAM_RE (re.DOTALL).
const std::regex& qwen3_partial_param_re() {
  static const std::regex re(R"(<\s*parameter\s*=\s*([^>]+)>([\s\S]*)$)");
  return re;
}

// qwen3.py:68 _qwen3_arg_converter — parse <parameter=NAME>VALUE</parameter>
// markup into an (insertion-ordered) JSON object string.
std::string qwen3_arg_converter(const std::string& raw_args, bool partial) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (auto it = std::sregex_iterator(raw_args.begin(), raw_args.end(),
                                      qwen3_param_re());
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    params[m[1].str()] = trim_wrapping_newlines(m[2].str());
  }
  if (partial) {
    std::string remaining =
        std::regex_replace(raw_args, qwen3_param_re(), std::string());
    std::smatch m;
    if (std::regex_search(remaining, m, qwen3_partial_param_re())) {
      std::string name = m[1].str();
      if (!name.empty()) {
        params[name] = trim_wrapping_newlines(m[2].str());
      }
    }
  }
  return python_json_dumps(params);
}

}  // namespace

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
  // Assembly-layer fields (qwen3.py:194-197).
  c.arg_converter = qwen3_arg_converter;
  c.stream_arg_deltas = true;
  c.strip_trailing_reasoning_whitespace = false;
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

  // Assembly-layer fields (kimi_k2.py:141-146). No arg_converter: args are
  // native JSON streamed/extracted verbatim.
  c.stream_arg_deltas = true;
  c.tool_args_json = true;
  c.strip_trailing_reasoning_whitespace = true;
  c.drop_whitespace_only_content_before_tools = true;
  c.strip_content_whitespace_with_tools = false;
  c.validate_tool_names = false;
  return c;
}

// ── minimax_m2 (minimax_m2.py) ────────────────────────────────────────────
namespace {

// minimax_m2.py:43 _PARAM_RE (re.DOTALL). Numbered groups: 1=dq_name, 2=sq_name,
// 3=bare_name, 4=value. `.` -> [\s\S] for DOTALL parity.
const std::regex& minimax_param_re() {
  static const std::regex re(
      R"RE(<\s*parameter\s+name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^>\s]+))\s*>([\s\S]*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s+name\s*=)))RE");
  return re;
}
// minimax_m2.py:51 _PARTIAL_PARAM_RE (re.DOTALL).
const std::regex& minimax_partial_param_re() {
  static const std::regex re(
      R"RE(<\s*parameter\s+name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^>\s]+))\s*>([\s\S]*)$)RE");
  return re;
}
// minimax_m2.py:64 — dq_name or sq_name or bare_name, then .strip().
std::string minimax_pick_name(const std::smatch& m) {
  std::string n = m[1].matched
                      ? m[1].str()
                      : (m[2].matched ? m[2].str()
                                      : (m[3].matched ? m[3].str() : std::string()));
  return strip(n);
}
// minimax_m2.py:60 _minimax_m2_arg_converter. Values kept verbatim (strings).
std::string minimax_m2_arg_converter(const std::string& raw, bool partial) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (auto it = std::sregex_iterator(raw.begin(), raw.end(), minimax_param_re());
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    std::string name = minimax_pick_name(m);
    if (name.empty()) continue;
    params[name] = m[4].str();
  }
  if (partial) {
    std::string remaining =
        std::regex_replace(raw, minimax_param_re(), std::string());
    std::smatch m;
    if (std::regex_search(remaining, m, minimax_partial_param_re())) {
      std::string name = minimax_pick_name(m);
      if (!name.empty()) params[name] = m[4].str();
    }
  }
  return python_json_dumps(params);
}

}  // namespace

// minimax_m2.py:94 (minimax_m2_config).
ParserEngineConfig minimax_m2_config() {
  const std::string TOOL_CALL_START = "<minimax:tool_call>";
  const std::string TOOL_CALL_END = "</minimax:tool_call>";
  const std::string THINK_START = "<think>";
  const std::string THINK_END = "</think>";
  const std::string INVOKE_PREFIX_DQ = "<invoke name=\"";
  const std::string INVOKE_PREFIX_SQ = "<invoke name='";
  const std::string INVOKE_PREFIX_UNQUOTED = "<invoke name=";
  const std::string INVOKE_END = "</invoke>";
  const std::string NAME_END_DQ = "\">";
  const std::string NAME_END_SQ = "'>";
  const std::string NAME_END_UNQUOTED = ">";
  const std::string PARAM_START = "<parameter name=";
  const std::string PARAM_END = "</parameter>";

  ParserEngineConfig c;
  c.name = "minimax_m2";
  c.initial_state = S::REASONING;
  c.terminals = {
      {"THINK_START", THINK_START},
      {"THINK_END", THINK_END},
      {"TOOL_START", TOOL_CALL_START},
      {"PARAM_START", PARAM_START},
      {"PARAM_END", PARAM_END},
      {"TOOL_END", TOOL_CALL_END},
      {"INVOKE_PREFIX_DQ", INVOKE_PREFIX_DQ},
      {"INVOKE_PREFIX_SQ", INVOKE_PREFIX_SQ},
      {"INVOKE_PREFIX_UNQUOTED", INVOKE_PREFIX_UNQUOTED},
      {"INVOKE_END", INVOKE_END},
      {"NAME_END_DQ", NAME_END_DQ},
      {"NAME_END_SQ", NAME_END_SQ},
      {"NAME_END_UNQUOTED", NAME_END_UNQUOTED},
  };
  c.token_id_terminals = {
      {"THINK_START", THINK_START},
      {"THINK_END", THINK_END},
      {"TOOL_START", TOOL_CALL_START},
      {"TOOL_END", TOOL_CALL_END},
  };
  c.transitions = {
      {{S::REASONING, "THINK_START"}, Transition(S::REASONING, {})},
      {{S::REASONING, "THINK_END"}, Transition(S::CONTENT, {E::REASONING_END})},
      {{S::CONTENT, "THINK_END"}, Transition(S::CONTENT, {})},
      {{S::REASONING, "TOOL_START"},
       Transition(S::TOOL_PREAMBLE, {E::REASONING_END})},
      {{S::CONTENT, "TOOL_START"}, Transition(S::TOOL_PREAMBLE, {})},
      {{S::TOOL_ARGS, "PARAM_START"}, Transition(S::TOOL_ARGS, {E::ARG_VALUE_CHUNK})},
      {{S::TOOL_ARGS, "PARAM_END"}, Transition(S::TOOL_ARGS, {E::ARG_VALUE_CHUNK})},
      {{S::TOOL_PREAMBLE, "TOOL_END"}, Transition(S::CONTENT, {})},
      {{S::TOOL_BETWEEN, "TOOL_END"}, Transition(S::CONTENT, {})},
      {{S::CONTENT, "TOOL_END"}, Transition(S::CONTENT, {})},
      {{S::TOOL_ARGS, "INVOKE_END"},
       Transition(S::TOOL_BETWEEN, {E::TOOL_CALL_END})},
  };
  // minimax_m2.py:164 — INVOKE_PREFIX_* from CONTENT/PREAMBLE/BETWEEN -> TOOL_NAME.
  for (S st : {S::CONTENT, S::TOOL_PREAMBLE, S::TOOL_BETWEEN}) {
    for (const char* term :
         {"INVOKE_PREFIX_DQ", "INVOKE_PREFIX_SQ", "INVOKE_PREFIX_UNQUOTED"}) {
      c.transitions[{st, term}] = Transition(S::TOOL_NAME, {E::TOOL_CALL_START});
    }
  }
  // minimax_m2.py:180 — NAME_END_* from TOOL_NAME -> TOOL_ARGS.
  for (const char* term : {"NAME_END_DQ", "NAME_END_SQ", "NAME_END_UNQUOTED"}) {
    c.transitions[{S::TOOL_NAME, term}] = Transition(S::TOOL_ARGS, {});
  }
  // Assembly-layer fields (minimax_m2.py:192-195).
  c.arg_converter = minimax_m2_arg_converter;
  c.stream_arg_deltas = true;
  c.tool_args_json = false;
  c.validate_tool_names = true;
  return c;
}

// ── glm47_moe (glm47_moe.py) ──────────────────────────────────────────────
namespace {

// glm47_moe.py:44 _ARG_RE (re.DOTALL): 1=key, 2=value.
const std::regex& glm47_arg_re() {
  static const std::regex re(
      R"RE(<arg_key>([\s\S]*?)</arg_key>\s*<arg_value>([\s\S]*?)</arg_value>)RE");
  return re;
}
// glm47_moe.py:49 _PARTIAL_ARG_RE (re.DOTALL).
const std::regex& glm47_partial_arg_re() {
  static const std::regex re(
      R"RE(<arg_key>([\s\S]*?)</arg_key>\s*<arg_value>([\s\S]*)$)RE");
  return re;
}
// glm47_moe.py:56 _glm47_arg_converter. Values kept verbatim (strings).
std::string glm47_arg_converter(const std::string& raw, bool partial) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (auto it = std::sregex_iterator(raw.begin(), raw.end(), glm47_arg_re());
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    params[strip(m[1].str())] = m[2].str();
  }
  if (partial) {
    std::string remaining =
        std::regex_replace(raw, glm47_arg_re(), std::string());
    std::smatch m;
    if (std::regex_search(remaining, m, glm47_partial_arg_re())) {
      std::string key = strip(m[1].str());
      if (!key.empty()) params[key] = m[2].str();
    }
  }
  return python_json_dumps(params);
}

}  // namespace

// glm47_moe.py:74 (glm47_moe_config).
ParserEngineConfig glm47_moe_config(bool thinking) {
  const std::string THINK_START = "<think>";
  const std::string THINK_END = "</think>";
  const std::string TOOL_CALL_START = "<tool_call>";
  const std::string TOOL_CALL_END = "</tool_call>";
  const std::string ARG_KEY_START = "<arg_key>";
  const std::string ARG_KEY_END = "</arg_key>";
  const std::string ARG_VALUE_START = "<arg_value>";
  const std::string ARG_VALUE_END = "</arg_value>";

  ParserEngineConfig c;
  c.name = "glm47_moe";
  c.initial_state = thinking ? S::REASONING : S::CONTENT;

  if (thinking) {
    c.terminals["THINK_START"] = THINK_START;
    c.terminals["THINK_END"] = THINK_END;
    c.token_id_terminals["THINK_START"] = THINK_START;
    c.token_id_terminals["THINK_END"] = THINK_END;
  }
  c.terminals["TOOL_START"] = TOOL_CALL_START;
  c.terminals["TOOL_END"] = TOOL_CALL_END;
  c.terminals["ARG_KEY_START"] = ARG_KEY_START;
  c.terminals["ARG_KEY_END"] = ARG_KEY_END;
  c.terminals["ARG_VALUE_START"] = ARG_VALUE_START;
  c.terminals["ARG_VALUE_END"] = ARG_VALUE_END;
  c.token_id_terminals["TOOL_START"] = TOOL_CALL_START;
  c.token_id_terminals["TOOL_END"] = TOOL_CALL_END;

  if (thinking) {
    // glm47_moe.py:104 reasoning_transitions.
    c.transitions[{S::CONTENT, "THINK_START"}] =
        Transition(S::REASONING, {E::REASONING_START});
    c.transitions[{S::REASONING, "THINK_END"}] =
        Transition(S::CONTENT, {E::REASONING_END});
    c.transitions[{S::CONTENT, "THINK_END"}] = Transition(S::CONTENT, {});
  }
  // glm47_moe.py:142.
  c.transitions[{S::REASONING, "THINK_START"}] = Transition(S::REASONING, {});
  c.transitions[{S::REASONING, "TOOL_START"}] =
      Transition(S::TOOL_NAME, {E::REASONING_END, E::TOOL_CALL_START});
  c.transitions[{S::CONTENT, "TOOL_START"}] =
      Transition(S::TOOL_NAME, {E::TOOL_CALL_START});
  c.transitions[{S::TOOL_NAME, "ARG_KEY_START"}] =
      Transition(S::TOOL_ARGS, {E::ARG_VALUE_CHUNK});
  c.transitions[{S::TOOL_NAME, "TOOL_END"}] =
      Transition(S::CONTENT, {E::TOOL_CALL_END});
  c.transitions[{S::TOOL_ARGS, "TOOL_END"}] =
      Transition(S::CONTENT, {E::TOOL_CALL_END});
  // glm47_moe.py:75 arg_tag_transitions.
  for (const char* term :
       {"ARG_KEY_START", "ARG_KEY_END", "ARG_VALUE_START", "ARG_VALUE_END"}) {
    c.transitions[{S::TOOL_ARGS, term}] =
        Transition(S::TOOL_ARGS, {E::ARG_VALUE_CHUNK});
  }
  // Assembly-layer fields (glm47_moe.py:168-171).
  c.arg_converter = glm47_arg_converter;
  c.stream_arg_deltas = true;
  c.tool_args_json = false;
  c.validate_tool_names = true;
  return c;
}

// ── deepseek_v4 / deepseek_v32 (DSML) ─────────────────────────────────────
namespace {

const std::string& dsml_delim() {
  // "｜DSML｜" — the U+FF5C fullwidth vertical bar (UTF-8 ef bd 9c) around "DSML",
  // spelled as UTF-8 bytes so no \x escape greedily swallows the following 'D'.
  static const std::string d =
      "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
  return d;
}
// deepseek_v4.py:53 _PARAM_RE (re.DOTALL): 1=name, 2=string flag, 3=value.
const std::regex& dsml_param_re() {
  static const std::regex re(
      "<" + dsml_delim() +
      R"RE(parameter\s+name="([^"]+)"\s+string="(true|false)">([\s\S]*?)</)RE" +
      dsml_delim() + "parameter>");
  return re;
}
// deepseek_v4.py:58 _PARTIAL_PARAM_RE (re.DOTALL).
const std::regex& dsml_partial_param_re() {
  static const std::regex re(
      "<" + dsml_delim() +
      R"RE(parameter\s+name="([^"]+)"\s+string="(true|false)">([\s\S]*)$)RE");
  return re;
}
// deepseek_v4.py:65 _dsml_arg_converter. string="true" -> verbatim string;
// string="false" -> json.loads (typed), fallback string on the complete-match
// path, and (partial path) dropped on parse failure.
std::string dsml_arg_converter(const std::string& raw, bool partial) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  std::size_t last_end = 0;
  for (auto it = std::sregex_iterator(raw.begin(), raw.end(), dsml_param_re());
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    std::string name = m[1].str(), is_str = m[2].str(), value = m[3].str();
    if (is_str == "true") {
      params[name] = value;
    } else {
      try {
        params[name] = nlohmann::ordered_json::parse(value);
      } catch (...) {
        params[name] = value;
      }
    }
    last_end = static_cast<std::size_t>(m.position(0) + m.length(0));
  }
  if (partial) {
    std::string tail = raw.substr(std::min(last_end, raw.size()));
    std::smatch m;
    if (std::regex_search(tail, m, dsml_partial_param_re())) {
      std::string name = m[1].str(), is_str = m[2].str(), value = m[3].str();
      if (is_str == "true") {
        params[name] = value;
      } else {
        try {
          params[name] = nlohmann::ordered_json::parse(value);
        } catch (...) {
          // contextlib.suppress -> leave the key absent (deepseek_v4.py:87).
        }
      }
    }
  }
  return python_json_dumps(params);
}

// Shared DSML tool-call transitions (deepseek_v4.py / deepseek_v32.py). The two
// families differ only in the outer wrapper terminal and reasoning tags.
void add_dsml_tool_transitions(ParserEngineConfig& c) {
  c.transitions[{S::TOOL_PREAMBLE, "INVOKE_PREFIX"}] =
      Transition(S::TOOL_NAME, {E::TOOL_CALL_START});
  c.transitions[{S::TOOL_NAME, "INVOKE_NAME_END"}] = Transition(S::TOOL_ARGS, {});
  c.transitions[{S::TOOL_ARGS, "INVOKE_END"}] =
      Transition(S::TOOL_BETWEEN, {E::TOOL_CALL_END});
  c.transitions[{S::TOOL_ARGS, "TOOL_END"}] =
      Transition(S::CONTENT, {E::TOOL_CALL_END});
  c.transitions[{S::TOOL_BETWEEN, "INVOKE_PREFIX"}] =
      Transition(S::TOOL_NAME, {E::TOOL_CALL_START});
  c.transitions[{S::TOOL_BETWEEN, "TOOL_END"}] = Transition(S::CONTENT, {});
}

}  // namespace

// deepseek_v4.py:125 (deepseek_v4_config).
ParserEngineConfig deepseek_v4_config(bool thinking) {
  const std::string DSML = dsml_delim();
  const std::string THINK_START = "<think>";
  const std::string THINK_END = "</think>";
  const std::string TOOL_START = "<" + DSML + "tool_calls>";
  const std::string TOOL_END = "</" + DSML + "tool_calls>";
  const std::string INVOKE_PREFIX = "<" + DSML + "invoke name=\"";
  const std::string INVOKE_NAME_END = "\">";
  const std::string INVOKE_END = "</" + DSML + "invoke>";
  const std::string PARAM_CLOSE = "</" + DSML + "parameter>";

  ParserEngineConfig c;
  c.name = "deepseek_v4";
  c.initial_state = thinking ? S::REASONING : S::CONTENT;
  c.terminals = {
      {"THINK_START", THINK_START},   {"THINK_END", THINK_END},
      {"TOOL_START", TOOL_START},     {"TOOL_END", TOOL_END},
      {"INVOKE_PREFIX", INVOKE_PREFIX}, {"INVOKE_NAME_END", INVOKE_NAME_END},
      {"INVOKE_END", INVOKE_END},     {"PARAM_CLOSE", PARAM_CLOSE},
  };
  c.token_id_terminals = {
      {"THINK_START", THINK_START},
      {"THINK_END", THINK_END},
      {"TOOL_START", TOOL_START},
      {"TOOL_END", TOOL_END},
  };
  c.transitions[{S::CONTENT, "THINK_START"}] =
      Transition(S::REASONING, {E::REASONING_START});
  c.transitions[{S::CONTENT, "THINK_END"}] = Transition(S::CONTENT, {});
  c.transitions[{S::REASONING, "THINK_START"}] = Transition(S::REASONING, {});
  c.transitions[{S::REASONING, "THINK_END"}] =
      Transition(S::CONTENT, {E::REASONING_END});
  c.transitions[{S::REASONING, "TOOL_START"}] =
      Transition(S::TOOL_PREAMBLE, {E::REASONING_END});
  c.transitions[{S::CONTENT, "TOOL_START"}] = Transition(S::TOOL_PREAMBLE, {});
  add_dsml_tool_transitions(c);
  // deepseek_v4.py:205-208.
  c.arg_converter = dsml_arg_converter;
  c.arg_structural_chars = std::set<char>{'>'};
  c.strip_content_whitespace_with_tools = false;
  c.tool_args_json = false;
  return c;
}

// deepseek_v32.py:51 (deepseek_v32_config).
ParserEngineConfig deepseek_v32_config() {
  const std::string DSML = dsml_delim();
  const std::string FUNC_START = "<" + DSML + "function_calls>";
  const std::string FUNC_END = "</" + DSML + "function_calls>";
  const std::string INVOKE_PREFIX = "<" + DSML + "invoke name=\"";
  const std::string INVOKE_NAME_END = "\">";
  const std::string INVOKE_END = "</" + DSML + "invoke>";
  const std::string PARAM_CLOSE = "</" + DSML + "parameter>";

  ParserEngineConfig c;
  c.name = "deepseek_v32";
  c.initial_state = S::CONTENT;
  c.terminals = {
      {"TOOL_START", FUNC_START},       {"TOOL_END", FUNC_END},
      {"INVOKE_PREFIX", INVOKE_PREFIX}, {"INVOKE_NAME_END", INVOKE_NAME_END},
      {"INVOKE_END", INVOKE_END},       {"PARAM_CLOSE", PARAM_CLOSE},
  };
  c.token_id_terminals = {
      {"TOOL_START", FUNC_START},
      {"TOOL_END", FUNC_END},
  };
  c.transitions[{S::CONTENT, "TOOL_START"}] = Transition(S::TOOL_PREAMBLE, {});
  add_dsml_tool_transitions(c);
  // deepseek_v32.py:98 content_events omit REASONING (no reasoning tags).
  c.content_events = {
      {S::CONTENT, E::TEXT_CHUNK},
      {S::TOOL_NAME, E::TOOL_NAME},
      {S::TOOL_ARGS, E::ARG_VALUE_CHUNK},
  };
  // deepseek_v32.py:103-106.
  c.arg_converter = dsml_arg_converter;
  c.arg_structural_chars = std::set<char>{'>'};
  c.strip_content_whitespace_with_tools = false;
  c.tool_args_json = false;
  return c;
}

// nemotron_v3.py:34 (nemotron_v3_config): qwen3 grammar, distinct name +
// strip_trailing_reasoning_whitespace=True.
ParserEngineConfig nemotron_v3_config(bool thinking) {
  ParserEngineConfig c = qwen3_config(thinking, "nemotron_v3");
  c.strip_trailing_reasoning_whitespace = true;
  return c;
}

// ── gemma4 (gemma4.py) ────────────────────────────────────────────────────
namespace {

// gemma4.py:42 STRING_DELIM (`<|"|>`) + _DELIM_LEN.
const std::string GEMMA4_STRING_DELIM = "<|\"|>";
constexpr std::size_t GEMMA4_DELIM_LEN = 5;

// gemma4.py:57 _strip_partial_delim — strip a trailing partial STRING_DELIM.
std::string gemma4_strip_partial_delim(const std::string& value) {
  for (std::size_t k = GEMMA4_DELIM_LEN; k >= 1; --k) {
    const std::string suffix = GEMMA4_STRING_DELIM.substr(0, k);
    if (value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0)
      return value.substr(0, value.size() - suffix.size());
  }
  return value;
}

nlohmann::ordered_json gemma4_parse_args(const std::string& s, bool partial);

// gemma4.py:204 _parse_gemma4_array.
nlohmann::ordered_json gemma4_parse_array(const std::string& a, bool partial) {
  nlohmann::ordered_json items = nlohmann::ordered_json::array();
  const long n = static_cast<long>(a.size());
  long i = 0;
  auto at = [&](long k) { return a[static_cast<std::size_t>(k)]; };
  auto sub = [&](long b, long e) {
    return a.substr(static_cast<std::size_t>(b), static_cast<std::size_t>(e - b));
  };
  auto is_delim = [&](long k) {
    return k + static_cast<long>(GEMMA4_DELIM_LEN) <= n &&
           sub(k, k + static_cast<long>(GEMMA4_DELIM_LEN)) == GEMMA4_STRING_DELIM;
  };
  auto find_delim = [&](long from) -> long {
    std::size_t p = a.find(GEMMA4_STRING_DELIM, static_cast<std::size_t>(from));
    return p == std::string::npos ? -1 : static_cast<long>(p);
  };
  while (i < n) {
    while (i < n && (at(i) == ' ' || at(i) == ',' || at(i) == '\n' || at(i) == '\t'))
      ++i;
    if (i >= n) break;
    if (is_delim(i)) {
      i += static_cast<long>(GEMMA4_DELIM_LEN);
      long end_pos = find_delim(i);
      if (end_pos == -1) {
        items.push_back(sub(i, n));
        break;
      }
      items.push_back(sub(i, end_pos));
      i = end_pos + static_cast<long>(GEMMA4_DELIM_LEN);
    } else if (at(i) == '{') {
      int depth = 1;
      long obj_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (is_delim(i)) {
          i += static_cast<long>(GEMMA4_DELIM_LEN);
          long nd = find_delim(i);
          i = nd == -1 ? n : nd + static_cast<long>(GEMMA4_DELIM_LEN);
          continue;
        }
        if (at(i) == '{') ++depth;
        else if (at(i) == '}') --depth;
        ++i;
      }
      if (depth > 0) items.push_back(gemma4_parse_args(sub(obj_start, i), true));
      else items.push_back(gemma4_parse_args(sub(obj_start, i - 1), false));
    } else if (at(i) == '[') {
      int depth = 1;
      long sub_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (is_delim(i)) {
          i += static_cast<long>(GEMMA4_DELIM_LEN);
          long nd = find_delim(i);
          i = nd == -1 ? n : nd + static_cast<long>(GEMMA4_DELIM_LEN);
          continue;
        }
        if (at(i) == '[') ++depth;
        else if (at(i) == ']') --depth;
        ++i;
      }
      if (depth > 0) items.push_back(gemma4_parse_array(sub(sub_start, i), true));
      else items.push_back(gemma4_parse_array(sub(sub_start, i - 1), false));
    } else {
      long val_start = i;
      while (i < n && at(i) != ',' && at(i) != ']') ++i;
      if (partial && i >= n) break;
      if (i == val_start) break;  // no progress (malformed) — abort
      std::string raw_val = strip(sub(val_start, i));
      if (partial && !raw_val.empty() && raw_val.back() == '.') break;
      items.push_back(raw_val);
    }
  }
  return items;
}

// gemma4.py:68 _parse_gemma4_args.
nlohmann::ordered_json gemma4_parse_args(const std::string& s, bool partial) {
  nlohmann::ordered_json result = nlohmann::ordered_json::object();
  if (s.empty() || strip(s).empty()) return result;
  const long n = static_cast<long>(s.size());
  long i = 0;
  auto at = [&](long k) { return s[static_cast<std::size_t>(k)]; };
  auto sub = [&](long b, long e) {
    return s.substr(static_cast<std::size_t>(b), static_cast<std::size_t>(e - b));
  };
  auto is_delim = [&](long k) {
    return k + static_cast<long>(GEMMA4_DELIM_LEN) <= n &&
           sub(k, k + static_cast<long>(GEMMA4_DELIM_LEN)) == GEMMA4_STRING_DELIM;
  };
  auto find_delim = [&](long from) -> long {
    std::size_t p = s.find(GEMMA4_STRING_DELIM, static_cast<std::size_t>(from));
    return p == std::string::npos ? -1 : static_cast<long>(p);
  };
  while (i < n) {
    while (i < n && (at(i) == ' ' || at(i) == ',' || at(i) == '\n' || at(i) == '\t'))
      ++i;
    if (i >= n) break;
    long key_start = i;
    while (i < n && at(i) != ':') ++i;
    if (i >= n) break;
    std::string key = strip(sub(key_start, i));
    // key[_DELIM_LEN:-_DELIM_LEN] when wrapped in STRING_DELIM.
    if (key.size() >= GEMMA4_DELIM_LEN &&
        key.compare(0, GEMMA4_DELIM_LEN, GEMMA4_STRING_DELIM) == 0 &&
        key.compare(key.size() - GEMMA4_DELIM_LEN, GEMMA4_DELIM_LEN,
                    GEMMA4_STRING_DELIM) == 0) {
      long lo = static_cast<long>(GEMMA4_DELIM_LEN);
      long hi = static_cast<long>(key.size()) - static_cast<long>(GEMMA4_DELIM_LEN);
      key = hi > lo ? key.substr(static_cast<std::size_t>(lo),
                                 static_cast<std::size_t>(hi - lo))
                    : std::string();
    }
    ++i;  // skip ':'
    if (i >= n) {
      if (!partial) result[key] = "";
      break;
    }
    while (i < n && (at(i) == ' ' || at(i) == '\n' || at(i) == '\t')) ++i;
    if (i >= n) {
      if (!partial) result[key] = "";
      break;
    }
    if (is_delim(i)) {
      i += static_cast<long>(GEMMA4_DELIM_LEN);
      long val_start = i;
      long end_pos = find_delim(i);
      if (end_pos == -1) {
        std::string value = sub(val_start, n);
        if (partial) value = gemma4_strip_partial_delim(value);
        result[key] = value;
        break;
      }
      result[key] = sub(val_start, end_pos);
      i = end_pos + static_cast<long>(GEMMA4_DELIM_LEN);
    } else if (at(i) == '{') {
      int depth = 1;
      long obj_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (is_delim(i)) {
          i += static_cast<long>(GEMMA4_DELIM_LEN);
          long nd = find_delim(i);
          i = nd == -1 ? n : nd + static_cast<long>(GEMMA4_DELIM_LEN);
          continue;
        }
        if (at(i) == '{') ++depth;
        else if (at(i) == '}') --depth;
        ++i;
      }
      if (depth > 0) result[key] = gemma4_parse_args(sub(obj_start, i), true);
      else result[key] = gemma4_parse_args(sub(obj_start, i - 1), false);
    } else if (at(i) == '[') {
      int depth = 1;
      long arr_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (is_delim(i)) {
          i += static_cast<long>(GEMMA4_DELIM_LEN);
          long nd = find_delim(i);
          i = nd == -1 ? n : nd + static_cast<long>(GEMMA4_DELIM_LEN);
          continue;
        }
        if (at(i) == '[') ++depth;
        else if (at(i) == ']') --depth;
        ++i;
      }
      if (depth > 0) result[key] = gemma4_parse_array(sub(arr_start, i), true);
      else result[key] = gemma4_parse_array(sub(arr_start, i - 1), false);
    } else {
      long val_start = i;
      while (i < n && at(i) != ',' && at(i) != '}' && at(i) != ']') ++i;
      if (partial && i >= n) break;
      if (i == val_start) break;  // no progress (malformed) — abort
      std::string raw_val = strip(sub(val_start, i));
      if (partial && !raw_val.empty() && raw_val.back() == '.') break;
      result[key] = raw_val;
    }
  }
  return result;
}

// gemma4.py:285 _gemma4_arg_converter.
std::string gemma4_arg_converter(const std::string& raw_args, bool partial) {
  std::string text = strip(raw_args);
  if (!text.empty() && text.back() == '}') text.pop_back();
  return python_json_dumps(gemma4_parse_args(text, partial));
}

}  // namespace

// gemma4.py:296 (gemma4_config). Channel-based reasoning + <|tool_call> tool
// calls; custom key:value arg format via gemma4_arg_converter. The Gemma4Parser
// subclass adds the _preprocess_feed / _events_to_delta hooks (parser/gemma4.*).
ParserEngineConfig gemma4_config() {
  const std::string CHANNEL_START = "<|channel>";
  const std::string CHANNEL_END = "<channel|>";
  const std::string TOOL_CALL_START = "<|tool_call>";
  const std::string TOOL_CALL_END = "<tool_call|>";

  ParserEngineConfig c;
  c.name = "gemma4";
  c.initial_state = S::CONTENT;
  c.terminals = {
      {"THINK_START", CHANNEL_START}, {"THINK_END", CHANNEL_END},
      {"TOOL_START", TOOL_CALL_START}, {"TOOL_END", TOOL_CALL_END},
      {"CALL_PREFIX", "call:"}, {"OPEN_BRACE", "{"},
  };
  c.token_id_terminals = {
      {"THINK_START", CHANNEL_START}, {"THINK_END", CHANNEL_END},
      {"TOOL_START", TOOL_CALL_START}, {"TOOL_END", TOOL_CALL_END},
  };
  c.transitions = {
      {{S::CONTENT, "THINK_START"}, Transition(S::REASONING, {E::REASONING_START})},
      {{S::REASONING, "THINK_START"}, Transition(S::REASONING, {})},
      {{S::REASONING, "THINK_END"}, Transition(S::CONTENT, {E::REASONING_END})},
      {{S::REASONING, "TOOL_START"},
       Transition(S::TOOL_PREAMBLE, {E::REASONING_END, E::TOOL_CALL_START})},
      {{S::CONTENT, "TOOL_START"},
       Transition(S::TOOL_PREAMBLE, {E::REASONING_END, E::TOOL_CALL_START})},
      {{S::TOOL_PREAMBLE, "TOOL_END"}, Transition(S::CONTENT, {E::TOOL_CALL_END})},
      {{S::TOOL_PREAMBLE, "CALL_PREFIX"}, Transition(S::TOOL_NAME, {})},
      {{S::TOOL_NAME, "OPEN_BRACE"}, Transition(S::TOOL_ARGS, {})},
      {{S::TOOL_ARGS, "TOOL_END"}, Transition(S::CONTENT, {E::TOOL_CALL_END})},
      {{S::CONTENT, "TOOL_END"}, Transition(S::CONTENT, {})},
      {{S::CONTENT, "THINK_END"}, Transition(S::CONTENT, {})},
  };
  // gemma4.py:370 content_events (== default; set explicitly to mirror upstream).
  c.content_events = {
      {S::CONTENT, E::TEXT_CHUNK}, {S::REASONING, E::REASONING_CHUNK},
      {S::TOOL_NAME, E::TOOL_NAME}, {S::TOOL_ARGS, E::ARG_VALUE_CHUNK},
  };
  // gemma4.py:376-379 assembly fields.
  c.arg_converter = gemma4_arg_converter;
  c.tool_args_json = false;
  c.arg_structural_chars = std::set<char>{',', ':', '{', '}', '[', ']', '<'};
  c.preserve_tokens = {GEMMA4_STRING_DELIM};
  return c;
}

// ── inkling (inkling.py) ──────────────────────────────────────────────────
namespace {

// inkling.py:70 _scan_json_value — end index (exclusive) of the JSON object at
// raw[start], or -1 when still unterminated.
long inkling_scan_json_value(const std::string& raw, long start) {
  int depth = 0;
  bool in_string = false, escape = false;
  const long n = static_cast<long>(raw.size());
  for (long i = start; i < n; ++i) {
    char ch = raw[static_cast<std::size_t>(i)];
    if (escape) {
      escape = false;
      continue;
    }
    if (in_string) {
      if (ch == '\\') escape = true;
      else if (ch == '"') in_string = false;
      continue;
    }
    if (ch == '"') in_string = true;
    else if (ch == '{') ++depth;
    else if (ch == '}') {
      --depth;
      if (depth == 0) return i + 1;
    }
  }
  return -1;
}

// inkling.py:98 _args_value_span — verbatim span of the top-level "args" value
// from a (possibly incomplete) {"name":...,"args":{...}} wrapper. std::nullopt
// when the value has not started; throws (mirrors Python ValueError, caught by
// the arg-converter callers) when the value is not a JSON object.
std::optional<std::string> inkling_args_value_span(const std::string& raw) {
  int depth = 0;
  bool in_string = false, escape = false;
  long string_start = -1;
  std::optional<std::string> last_string;
  const long n = static_cast<long>(raw.size());
  static const std::string WS = " \t\r\n";
  for (long i = 0; i < n; ++i) {
    char ch = raw[static_cast<std::size_t>(i)];
    if (escape) {
      escape = false;
      continue;
    }
    if (in_string) {
      if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
        if (depth == 1)
          last_string = raw.substr(static_cast<std::size_t>(string_start + 1),
                                   static_cast<std::size_t>(i - string_start - 1));
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
      string_start = i;
    } else if (ch == ':' && depth == 1 && last_string && *last_string == "args") {
      long value_start = i + 1;
      while (value_start < n &&
             WS.find(raw[static_cast<std::size_t>(value_start)]) != std::string::npos)
        ++value_start;
      if (value_start >= n) return std::nullopt;
      if (raw[static_cast<std::size_t>(value_start)] != '{')
        throw std::runtime_error("Inkling tool call args must be a JSON object");
      long value_end = inkling_scan_json_value(raw, value_start);
      if (value_end == -1)
        return raw.substr(static_cast<std::size_t>(value_start));
      return raw.substr(static_cast<std::size_t>(value_start),
                        static_cast<std::size_t>(value_end - value_start));
    } else if (ch == '{' || ch == '[') {
      ++depth;
    } else if (ch == '}' || ch == ']') {
      --depth;
    }
  }
  return std::nullopt;
}

// inkling.py:146 _inkling_arg_converter — carve the "args" object out of the
// tool-call JSON wrapper.
std::string inkling_arg_converter(const std::string& raw_args, bool partial) {
  std::optional<std::string> span = inkling_args_value_span(raw_args);
  if (!span) return partial ? std::string() : std::string("{}");
  return *span;
}

}  // namespace

// inkling.py:175 (inkling_config). Typed content blocks (thinking / text /
// invoke_tool_json) in a single state machine; the InklingParser subclass adds
// the "args" wrapper-key unwrap + trailing-text flush (parser/inkling.*).
ParserEngineConfig inkling_config() {
  const std::string MESSAGE_MODEL = "<|message_model|>";
  const std::string CONTENT_TEXT = "<|content_text|>";
  const std::string CONTENT_THINKING = "<|content_thinking|>";
  const std::string CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>";
  const std::string CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>";
  const std::string CONTENT_TOOL_ERROR = "<|content_tool_error|>";
  const std::string CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>";
  const std::string END_MESSAGE = "<|end_message|>";

  ParserEngineConfig c;
  c.name = "inkling";
  c.initial_state = S::MESSAGE_HEADER;
  c.terminals = {
      {"MSG_MODEL", MESSAGE_MODEL},
      {"TEXT_START", CONTENT_TEXT},
      {"THINK_START", CONTENT_THINKING},
      {"THINK_END", END_MESSAGE},
      {"END_SAMPLING", CONTENT_MODEL_END_SAMPLING},
      {"TOOL_START", CONTENT_INVOKE_TOOL_JSON},
      {"TOOL_TEXT", CONTENT_INVOKE_TOOL_TEXT},
      {"TOOL_ERROR", CONTENT_TOOL_ERROR},
  };
  c.token_id_terminals = {};
  c.transitions = {
      {{S::CONTENT, "MSG_MODEL"}, Transition(S::MESSAGE_HEADER, {})},
      {{S::CONTENT, "TEXT_START"}, Transition(S::CONTENT, {})},
      {{S::CONTENT, "THINK_START"}, Transition(S::REASONING, {E::REASONING_START})},
      {{S::CONTENT, "TOOL_START"}, Transition(S::TOOL_ARGS, {E::TOOL_CALL_START})},
      {{S::CONTENT, "TOOL_TEXT"}, Transition(S::CONTENT, {})},
      {{S::CONTENT, "TOOL_ERROR"}, Transition(S::CONTENT, {})},
      {{S::MESSAGE_HEADER, "MSG_MODEL"}, Transition(S::MESSAGE_HEADER, {})},
      {{S::MESSAGE_HEADER, "TEXT_START"}, Transition(S::CONTENT, {})},
      {{S::MESSAGE_HEADER, "THINK_START"},
       Transition(S::REASONING, {E::REASONING_START})},
      {{S::MESSAGE_HEADER, "TOOL_START"},
       Transition(S::TOOL_ARGS, {E::TOOL_CALL_START})},
      {{S::MESSAGE_HEADER, "TOOL_TEXT"}, Transition(S::CONTENT, {})},
      {{S::MESSAGE_HEADER, "TOOL_ERROR"}, Transition(S::CONTENT, {})},
      {{S::REASONING, "THINK_START"}, Transition(S::REASONING, {})},
      {{S::REASONING, "TOOL_START"},
       Transition(S::TOOL_ARGS, {E::REASONING_END, E::TOOL_CALL_START})},
  };
  // inkling.py:256 — block-end terminals (THINK_END / END_SAMPLING) behave
  // identically regardless of label.
  for (const char* end : {"THINK_END", "END_SAMPLING"}) {
    c.transitions[{S::CONTENT, end}] = Transition(S::CONTENT, {});
    c.transitions[{S::REASONING, end}] = Transition(S::CONTENT, {E::REASONING_END});
    c.transitions[{S::TOOL_ARGS, end}] = Transition(S::CONTENT, {E::TOOL_CALL_END});
    c.transitions[{S::MESSAGE_HEADER, end}] = Transition(S::CONTENT, {});
  }
  // inkling.py:288-295 assembly fields.
  c.arg_converter = inkling_arg_converter;
  c.stream_arg_deltas = true;
  c.tool_args_json = true;
  c.strip_trailing_reasoning_whitespace = true;
  c.drop_whitespace_only_content_before_tools = true;
  c.strip_content_whitespace_with_tools = false;
  c.validate_tool_names = false;
  return c;
}

}  // namespace vllm::parser::engine
