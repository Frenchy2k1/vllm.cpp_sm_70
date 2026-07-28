# vllm.cpp parity harness; oracle = upstream vLLM unified parser ASSEMBLY layer.
"""Capture golden DeltaMessage / ExtractedToolCallInformation sequences from the
vLLM 0.26 parser ASSEMBLY layer (``vllm/parser/engine/parser_engine.py`` +
``vllm/parser/qwen3.py`` + ``vllm/parser/kimi_k2.py``) and emit the C++ gate's
goldens ``.inc``.

Like the engine-core dump, the assembly layer is a PURE FUNCTION of the
(delta_text, delta_token_ids) stream, so this needs no GPU, no model and no
installed vLLM wheel: it copies the engine + assembly + family modules from the
pinned vLLM source tree, rewrites their intra-engine imports to a local package
``pe``, and stubs the few serving-layer imports the assembly touches
(protocol DeltaMessage/ToolCall structs, a deterministic make_tool_call_id, the
Parser base's history-count init, and the no-schema tool_parsers.utils helpers).

The tool-call id for the qwen3 "random" id_type is normally a random uuid; the
stub make_tool_call_id is DETERMINISTIC (``chatcmpl-tool-<idx>``) so the gate can
compare ids exactly. The C++ gate injects the identical deterministic factory.

Usage::

    VLLM_SOURCE=/home/mudler/_git/vllm python3 \
      tools/parity/dump_parser_engine_assembly.py \
      --out tests/vllm/parser/engine/test_parser_engine_assembly_goldens.inc

Regenerate whenever the pin advances and re-run the C++ gate
``test_parser_engine_assembly``; any changed sequence is a real divergence to
reconcile 1:1 with upstream.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile

ENGINE_MODULES = [
    "events",
    "token_id_scanner",
    "incremental_lexer",
    "parser_engine_config",
    "streaming_parser_engine",
    "parser_engine",
]
FAMILY_MODULES = [
    "qwen3",
    "kimi_k2",
    "seed_oss",
    # ROAD-V1-C8: the remaining engine-backed families ported to the C++ engine.
    "minimax_m2",
    "glm47_moe",
    "deepseek_v4",
    "deepseek_v32",
    "nemotron_v3",
    # ROAD-V1-C8 (C8-2): the last two deferred engine-backed families.
    "gemma4",
    "inkling",
]


def build_sandbox(vllm_source):
    """Copy the engine + assembly + family modules into a temp package ``pe``
    and lay down stub ``vllm.*`` modules for the serving-layer imports."""
    tmp = tempfile.mkdtemp(prefix="peasm_")
    pkg = os.path.join(tmp, "pe")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()

    src_engine = os.path.join(vllm_source, "vllm", "parser", "engine")
    src_parser = os.path.join(vllm_source, "vllm", "parser")

    def copy_rewrite(text):
        # Engine modules live under pe.*; family modules import each other via
        # vllm.parser.<name> and the engine via vllm.parser.engine.<name>.
        text = text.replace("from vllm.parser.engine.", "from pe.")
        text = text.replace("from vllm.parser.qwen3 import", "from pe.qwen3 import")
        text = text.replace("from vllm.parser.kimi_k2 import", "from pe.kimi_k2 import")
        # nemotron_v3 imports qwen3; deepseek_v32 imports deepseek_v4.
        text = text.replace(
            "from vllm.parser.deepseek_v4 import", "from pe.deepseek_v4 import")
        return text

    for m in ENGINE_MODULES:
        text = open(os.path.join(src_engine, m + ".py")).read()
        open(os.path.join(pkg, m + ".py"), "w").write(copy_rewrite(text))
    for m in FAMILY_MODULES:
        text = open(os.path.join(src_parser, m + ".py")).read()
        open(os.path.join(pkg, m + ".py"), "w").write(copy_rewrite(text))

    # ---- stub vllm.* package tree ----
    def write(relpath, text):
        full = os.path.join(tmp, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").write(text)

    write("vllm/__init__.py", "")
    write("vllm/logger.py", "import logging\n"
          "def init_logger(name):\n    return logging.getLogger(name)\n")

    write("vllm/parser/__init__.py", "")
    write("vllm/parser/utils.py",
          "def count_history_tool_calls(request):\n    return 0\n")
    write("vllm/parser/abstract_parser.py", _ABSTRACT_PARSER_STUB)

    write("vllm/entrypoints/__init__.py", "")
    write("vllm/entrypoints/chat_utils.py", _CHAT_UTILS_STUB)
    write("vllm/entrypoints/openai/__init__.py", "")
    write("vllm/entrypoints/openai/engine/__init__.py", "")
    write("vllm/entrypoints/openai/engine/protocol.py", _PROTOCOL_STUB)
    # glm47_moe imports the request protocol types at module load (used only for
    # type hints); a bare placeholder class is enough for the pure-function gate.
    write("vllm/entrypoints/openai/chat_completion/__init__.py", "")
    write("vllm/entrypoints/openai/chat_completion/protocol.py",
          "class ChatCompletionRequest:\n    pass\n")
    write("vllm/entrypoints/openai/responses/__init__.py", "")
    write("vllm/entrypoints/openai/responses/protocol.py",
          "class ResponsesRequest:\n    pass\n")

    write("vllm/tool_parsers/__init__.py", "")
    write("vllm/tool_parsers/utils.py", _TOOL_UTILS_STUB)

    sys.path.insert(0, tmp)
    return tmp


_ABSTRACT_PARSER_STUB = '''
from dataclasses import dataclass, field

@dataclass
class StreamState:
    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    prompt_reasoning_checked: bool = False
    previous_text: str = ""
    previous_token_ids: list = field(default_factory=list)
    history_tool_call_cnt: int = 0
    history_tool_call_cnt_initialized: bool = False
    tool_call_id_type: str = "random"
    function_name_returned: bool = False
    engine_based: bool = False

class Parser:
    def _initialize_history_tool_call_cnt(self, request):
        state = self._stream_state
        if state.history_tool_call_cnt_initialized:
            return
        if state.tool_call_id_type != "kimi_k2":
            state.history_tool_call_cnt_initialized = True
            return
        state.history_tool_call_cnt = 0  # empty history in the gate
        state.history_tool_call_cnt_initialized = True
'''

_CHAT_UTILS_STUB = '''
def get_tool_call_id_type(model_config):
    return "random"

def make_tool_call_id(id_type="random", func_name=None, idx=None):
    # DETERMINISTIC for gate parity (the real impl uses a random uuid for
    # id_type="random"). The C++ gate injects the identical factory.
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    return f"chatcmpl-tool-{idx}"
'''

_PROTOCOL_STUB = '''
class DeltaFunctionCall:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments

class DeltaToolCall:
    def __init__(self, index=0, id=None, type=None, function=None):
        self.index = index
        self.id = id
        self.type = type
        self.function = function

class DeltaMessage:
    def __init__(self, role=None, content=None, reasoning=None, tool_calls=None):
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls

class FunctionCall:
    def __init__(self, id=None, name=None, arguments=None):
        self.id = id
        self.name = name
        self.arguments = arguments

class ToolCall:
    def __init__(self, id=None, function=None, type="function"):
        self.id = id
        self.function = function
        self.type = type

class ExtractedToolCallInformation:
    def __init__(self, tools_called=False, tool_calls=None, content=None):
        self.tools_called = tools_called
        self.tool_calls = tool_calls or []
        self.content = content

class FunctionDefinition:
    pass
'''

# The schema-aware coercion helpers are copied VERBATIM from the pinned oracle
# vllm/tool_parsers/utils.py @ 555967922 (extract_types_from_schema:620,
# coerce_to_schema_type:693, _TYPE_ALIASES:666, _is_json_finite:151) so the
# captured goldens exercise the REAL _fix_arg_types / _streamable_string_keys
# behavior when a request carries typed tool schemas. find_tool_properties /
# find_tool_name are faithful reductions over the gate's simple tool objects
# (.function.name / .function.parameters); with tools=None they degenerate
# exactly as upstream ({} / False), keeping the no-schema path byte-identical.
_TOOL_UTILS_STUB = '''
import json
import math


def find_tool_properties(tools, tool_name):
    if not tools:
        return {}
    for tool in tools:
        fn = getattr(tool, "function", None)
        if fn is None:
            continue
        if fn.name == tool_name:
            params = fn.parameters or {}
            return params.get("properties", {})
    return {}


def find_tool_name(tools, tool_name):
    if not tools:
        return False
    for tool in tools:
        fn = getattr(tool, "function", None)
        if fn is not None and fn.name == tool_name:
            return True
    return False


def _is_json_finite(obj):
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def extract_types_from_schema(schema):
    if schema is None or not isinstance(schema, dict):
        return ["string"]
    types = set()
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")
    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                types.update(extract_types_from_schema(choice))
    return list(types) if types else ["string"]


_TYPE_ALIASES = {
    "str": "string", "text": "string", "varchar": "string", "char": "string",
    "enum": "string", "int": "integer", "int32": "integer", "int64": "integer",
    "uint": "integer", "uint32": "integer", "uint64": "integer", "long": "integer",
    "short": "integer", "unsigned": "integer", "float": "number",
    "float32": "number", "float64": "number", "double": "number", "bool": "boolean",
    "dict": "object", "arr": "array", "list": "array", "sequence": "array",
}


def coerce_to_schema_type(value, schema_type):
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    normalized_types = {
        _TYPE_ALIASES.get(key, key)
        for t in schema_type for key in [t.strip().lower()]
    }
    type_priority = [
        "null", "integer", "number", "boolean", "object", "array", "string",
    ]
    for candidate_type in type_priority:
        if candidate_type not in normalized_types:
            continue
        if candidate_type == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate_type == "string":
            return value
        if candidate_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate_type == "number":
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                continue
            return val if val != int(val) else int(val)
        if candidate_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ("true", "1"):
                return True
            if lower_val in ("false", "0"):
                return False
            continue
        if candidate_type in ("object", "array"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _is_json_finite(parsed):
                return parsed
            continue
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    if not _is_json_finite(parsed):
        return value
    return parsed
'''


class Req:
    def __init__(self, include_reasoning=True, tools=None, tool_choice="auto"):
        self.include_reasoning = include_reasoning
        self.tools = tools
        self.tool_choice = tool_choice
        self.messages = []


# Minimal function-tool objects mirroring ChatCompletionToolsParam
# (.function.name / .function.parameters) — the only fields the assembly's
# find_tool_properties / find_tool_name read.
class _Fn:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters


class _Tool:
    def __init__(self, name, parameters):
        self.type = "function"
        self.function = _Fn(name, parameters)


def tool(name, properties):
    """A function tool whose parameters carry the given typed `properties`."""
    return _Tool(name, {"type": "object", "properties": properties})


class MockTok:
    """Text-only mock: get_vocab() maps terminal texts to ids (never used since
    the gate streams carry no token ids), and exposes no all_special table."""

    def __init__(self):
        self._vocab = {
            "<think>": 1001, "</think>": 1002,
            "<seed:think>": 1011, "</seed:think>": 1012,
            "<tool_call>": 1020, "</tool_call>": 1021,
            "<seed:tool_call>": 1022, "</seed:tool_call>": 1023,
            "<|tool_calls_section_begin|>": 1003,
            "<|tool_calls_section_end|>": 1004,
            "<|tool_call_begin|>": 1005,
            "<|tool_call_end|>": 1006,
            "<|tool_call_argument_begin|>": 1007,
            # ROAD-V1-C8 families: outer wrapper tokens the parsers look up at
            # construction (never fired: gate streams carry no token ids).
            "<minimax:tool_call>": 1030,
            "</minimax:tool_call>": 1031,
            # gemma4: the channel markers must resolve to non-None token ids so
            # _reasoning_start/end_token_id are set (gates _preprocess_feed). The
            # C++ gate injects the identical vocab for gemma4.
            "<|channel>": 1040,
            "<channel|>": 1041,
            "<|tool_call>": 1042,
            "<tool_call|>": 1043,
        }

    def get_vocab(self):
        return self._vocab


# --------------------------------------------------------------------------- #
# Scenarios: each is (name, cfg, thinking, include_reasoning, [delta_text, ...])
# The deltas are streamed with finished=True on the last one; the full text is
# the concatenation and is also fed to extract_tool_calls.
# --------------------------------------------------------------------------- #
def chars(s):
    return list(s)


def make_scenarios():
    QWEN_FULL = (
        "<think>Let me check the weather.</think>"
        "Sure, checking now."
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>Tokyo</parameter>\n"
        "<parameter=unit>celsius</parameter>\n"
        "</function>\n</tool_call>"
    )
    QWEN_TWO = (
        "<tool_call>\n<function=alpha>\n<parameter=x>1</parameter>\n"
        "</function>\n</tool_call>"
        "<tool_call>\n<function=beta>\n<parameter=y>2</parameter>\n"
        "</function>\n</tool_call>"
    )
    QWEN_UNFINISHED = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo"
    )
    KIMI_FULL = (
        "<think>picking a tool</think>"
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "functions.get_weather:0<|tool_call_argument_begin|>"
        '{"city": "Tokyo", "unit": "celsius"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    scenarios = []
    # 1. qwen3 reasoning + XML tool call, whole-delta cadence.
    scenarios.append(("qwen3_reasoning_xml_wholedelta", "qwen3", True, True, [
        "<think>Let me check the weather.</think>",
        "Sure, checking now.",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n",
        "<parameter=unit>celsius</parameter>\n",
        "</function>\n</tool_call>",
    ]))
    # 2. same, char-by-char cadence (streaming stability).
    scenarios.append(("qwen3_reasoning_xml_charwise", "qwen3", True, True,
                      chars(QWEN_FULL)))
    # 3. reasoning suppressed (include_reasoning=False).
    scenarios.append(("qwen3_reasoning_suppressed", "qwen3", True, False, [
        "<think>Let me check the weather.</think>",
        "Sure, checking now.",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n",
        "</function>\n</tool_call>",
    ]))
    # 4. thinking-off plain content.
    scenarios.append(("qwen3_thinking_off_content", "qwen3", False, True,
                      chars("Just plain content, no tools at all.")))
    # 5. two consecutive tool calls (tool_index increments).
    scenarios.append(("qwen3_two_consecutive_tools", "qwen3", False, True,
                      chars(QWEN_TWO)))
    # 6. unfinished tool call flushed by finish().
    scenarios.append(("qwen3_unfinished_flush", "qwen3", False, True,
                      chars(QWEN_UNFINISHED)))
    # 7. seed_oss variant (same grammar, different wrapper tokens).
    scenarios.append(("seed_oss_reasoning_xml", "seed_oss", True, True, [
        "<seed:think>hmm</seed:think>",
        "<seed:tool_call>\n<function=ping>\n<parameter=host>localhost"
        "</parameter>\n</function>\n</seed:tool_call>",
    ]))
    # 8. kimi_k2 JSON args, char-by-char (held-back top-level brace).
    scenarios.append(("kimi_json_heldback_brace", "kimi_k2", True, True,
                      chars(KIMI_FULL)))
    # 9. kimi_k2 whole-delta cadence.
    scenarios.append(("kimi_json_wholedelta", "kimi_k2", True, True, [
        "<think>picking a tool</think>",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>",
        '{"city": "Tokyo", "unit": "celsius"}',
        "<|tool_call_end|><|tool_calls_section_end|>",
    ]))

    # ── ROAD-V1-C8: the remaining engine-backed families ──────────────────
    # 10. minimax_m2 reasoning + <invoke>/<parameter> XML, whole-delta.
    MINIMAX_FULL = (
        "<think>weighing options</think>"
        "Let me look that up."
        '<minimax:tool_call><invoke name="get_weather">\n'
        '<parameter name="city">Seattle</parameter>\n'
        '<parameter name="unit">celsius</parameter>\n'
        "</invoke></minimax:tool_call>"
    )
    scenarios.append(("minimax_reasoning_xml_wholedelta", "minimax_m2", True,
                      True, [
        "<think>weighing options</think>",
        "Let me look that up.",
        '<minimax:tool_call><invoke name="get_weather">\n',
        '<parameter name="city">Seattle</parameter>\n',
        '<parameter name="unit">celsius</parameter>\n',
        "</invoke></minimax:tool_call>",
    ]))
    # 11. minimax_m2 char-by-char (streaming stability).
    scenarios.append(("minimax_reasoning_xml_charwise", "minimax_m2", True,
                      True, chars(MINIMAX_FULL)))

    # 12. glm47_moe reasoning + <arg_key>/<arg_value>, whole-delta. The function
    #     name carries a trailing newline that the parser .strip()s.
    GLM_FULL = (
        "<think>deciding</think>"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key><arg_value>Seattle</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>"
    )
    scenarios.append(("glm47_reasoning_xml_wholedelta", "glm47_moe", True, True, [
        "<think>deciding</think>",
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key><arg_value>Seattle</arg_value>",
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value>",
        "</tool_call>",
    ]))
    # 13. glm47_moe char-by-char (name-strip + arg carve under fine cadence).
    scenarios.append(("glm47_reasoning_xml_charwise", "glm47_moe", True, True,
                      chars(GLM_FULL)))

    # 14. deepseek_v4 <think> reasoning + DSML tool_calls, whole-delta. A
    #     string="false" typed value exercises the json.loads coercion.
    DS = "｜DSML｜"
    DSV4_FULL = (
        "<think>calling</think>"
        f"<{DS}tool_calls>"
        f'<{DS}invoke name="get_weather">'
        f'<{DS}parameter name="city" string="true">Hangzhou</{DS}parameter>'
        f'<{DS}parameter name="days" string="false">5</{DS}parameter>'
        f"</{DS}invoke>"
        f"</{DS}tool_calls>"
    )
    scenarios.append(("deepseek_v4_reasoning_dsml_wholedelta", "deepseek_v4",
                      True, True, [
        "<think>calling</think>",
        f"<{DS}tool_calls>",
        f'<{DS}invoke name="get_weather">',
        f'<{DS}parameter name="city" string="true">Hangzhou</{DS}parameter>',
        f'<{DS}parameter name="days" string="false">5</{DS}parameter>',
        f"</{DS}invoke>",
        f"</{DS}tool_calls>",
    ]))
    # 15. deepseek_v4 char-by-char.
    scenarios.append(("deepseek_v4_reasoning_dsml_charwise", "deepseek_v4",
                      True, True, chars(DSV4_FULL)))

    # 16. deepseek_v32 DSML function_calls wrapper, no reasoning, whole-delta.
    DSV32_FULL = (
        f"<{DS}function_calls>"
        f'<{DS}invoke name="lookup">'
        f'<{DS}parameter name="q" string="true">weather</{DS}parameter>'
        f'<{DS}parameter name="count" string="false">3</{DS}parameter>'
        f"</{DS}invoke>"
        f"</{DS}function_calls>"
    )
    scenarios.append(("deepseek_v32_dsml_wholedelta", "deepseek_v32", False,
                      True, [
        f"<{DS}function_calls>",
        f'<{DS}invoke name="lookup">',
        f'<{DS}parameter name="q" string="true">weather</{DS}parameter>',
        f'<{DS}parameter name="count" string="false">3</{DS}parameter>',
        f"</{DS}invoke>",
        f"</{DS}function_calls>",
    ]))
    # 17. deepseek_v32 char-by-char.
    scenarios.append(("deepseek_v32_dsml_charwise", "deepseek_v32", False, True,
                      chars(DSV32_FULL)))

    # 18. nemotron_v3 (qwen3 grammar) reasoning + XML tool call, whole-delta.
    scenarios.append(("nemotron_reasoning_xml_wholedelta", "nemotron_v3", True,
                      True, [
        "<think>Let me check.</think>",
        "On it.",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n",
        "</function>\n</tool_call>",
    ]))
    # 19. nemotron_v3 char-by-char.
    NEMO_FULL = (
        "<think>Let me check.</think>On it."
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    )
    scenarios.append(("nemotron_reasoning_xml_charwise", "nemotron_v3", True,
                      True, chars(NEMO_FULL)))

    # ── ROAD-V1-C8 (C8-2): gemma4 + inkling ───────────────────────────────
    # 20. gemma4 explicit <|channel> reasoning (intrinsic `thought\n` prefix
    #     stripped by _events_to_delta) + <|tool_call> custom key:value args,
    #     whole-delta.
    D = '<|"|>'  # STRING_DELIM
    GEMMA_TOOL = (
        "<|tool_call>call:get_weather{"
        f"city:{D}Tokyo{D},unit:{D}celsius{D}"
        "}<tool_call|>"
    )
    GEMMA_EXPLICIT_FULL = (
        "<|channel>thought\nLet me check the weather.<channel|>"
        "Sure, checking now." + GEMMA_TOOL
    )
    scenarios.append(("gemma4_channel_tool_wholedelta", "gemma4", True, True, [
        "<|channel>thought\nLet me check the weather.<channel|>",
        "Sure, checking now.",
        GEMMA_TOOL,
    ]))
    # 21. gemma4 explicit-channel char-by-char (prefix strip under fine cadence).
    scenarios.append(("gemma4_channel_tool_charwise", "gemma4", True, True,
                      chars(GEMMA_EXPLICIT_FULL)))
    # 22. gemma4 ELIDED channel opener — the first delta starts with `thought\n`
    #     (the <|channel> token was consumed upstream), so _preprocess_feed injects
    #     it. whole-delta (char-by-char would not inject: only the first feed is
    #     eligible and a single leading char is not a `thought\n`/<channel|> cue).
    scenarios.append(("gemma4_elided_channel_wholedelta", "gemma4", True, True, [
        "thought\nWeighing the options.<channel|>",
        GEMMA_TOOL,
    ]))

    # 23. inkling typed content blocks: thinking + invoke_tool_json + a trailing
    #     text block AFTER the tool block (exercises the "args" wrapper unwrap in
    #     extract/parse AND the _single_pass_parse trailing flush), whole-delta.
    INK_THINK = "<|message_model|><|content_thinking|>Deciding.<|end_message|>"
    INK_TOOL = (
        "<|message_model|><|content_invoke_tool_json|>"
        '{"name": "get_weather", "args": {"city": "SF", "unit": "celsius"}}'
        "<|end_message|>"
    )
    INK_TEXT = "<|message_model|><|content_text|>Here you go.<|end_message|>"
    INK_FULL = INK_THINK + INK_TOOL + INK_TEXT
    scenarios.append(("inkling_think_tool_text_wholedelta", "inkling", True, True,
                      [INK_THINK, INK_TOOL, INK_TEXT]))
    # 24. inkling char-by-char (held-back JSON arg carve under fine cadence).
    scenarios.append(("inkling_think_tool_text_charwise", "inkling", True, True,
                      chars(INK_FULL)))
    # 25. inkling with a NON-OBJECT "args" value (JSON array) — the
    #     inkling_arg_converter raises, so extract/parse fall back to the
    #     name-from-args path (_extract_name_and_args -> _extract_args_value),
    #     which is where the "args" wrapper-key override matters (without it the
    #     {"args":[…]} wrapper leaks). whole-delta.
    INK_NONOBJ = (
        "<|message_model|><|content_invoke_tool_json|>"
        '{"name": "lookup", "args": [1, 2, 3]}'
        "<|end_message|>"
    )
    scenarios.append(("inkling_nonobject_args_wholedelta", "inkling", True, True,
                      [INK_NONOBJ]))
    # 26. inkling non-object args, char-by-char.
    scenarios.append(("inkling_nonobject_args_charwise", "inkling", True, True,
                      chars(INK_NONOBJ)))

    # ── JSON-schema tool-argument type coercion (_fix_arg_types) ───────────
    # These carry a request `tools` array whose function `parameters` declare
    # typed params, so the assembly coerces the raw-string arguments to the
    # declared JSON types (int/number/bool/string/array/null). Each scenario is
    # a 6-tuple (…, tools). Without these tools the same streams emit the raw
    # strings verbatim (the RED-first identity output the C++ gate proves fails).
    TYPED_TOOL = [tool("get_weather", {
        "days": {"type": "integer"},
        "unit": {"type": "string"},
        "active": {"type": "boolean"},
        "temp": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "integer"}},
    })]
    # 27. qwen3 XML tool call with a typed schema, whole-delta. days "5"->5,
    #     active "true"->true, temp "3.14"->3.14, unit stays "celsius" (string),
    #     tags "[1, 2, 3]"->[1, 2, 3] (array).
    scenarios.append(("qwen3_typed_schema_wholedelta", "qwen3", False, True, [
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=days>5</parameter>\n",
        "<parameter=unit>celsius</parameter>\n",
        "<parameter=active>true</parameter>\n",
        "<parameter=temp>3.14</parameter>\n",
        "<parameter=tags>[1, 2, 3]</parameter>\n",
        "</function>\n</tool_call>",
    ], TYPED_TOOL))
    # 28. same, char-by-char — exercises per-tick _fix_arg_types + the
    #     _streamable_string_keys held-back streaming (only `unit` streams its
    #     open string; the typed values are withheld until closed).
    QWEN_TYPED_FULL = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=days>5</parameter>\n"
        "<parameter=unit>celsius</parameter>\n"
        "<parameter=active>true</parameter>\n"
        "<parameter=temp>3.14</parameter>\n"
        "<parameter=tags>[1, 2, 3]</parameter>\n"
        "</function>\n</tool_call>"
    )
    scenarios.append(("qwen3_typed_schema_charwise", "qwen3", False, True,
                      chars(QWEN_TYPED_FULL), TYPED_TOOL))
    # 29. schema MISMATCH + nullable: `days` (integer) gets "abc" (uncoercible,
    #     stays the string "abc"); `count` (["integer","null"]) gets "null"
    #     (-> null); `n` (integer) gets "7" (-> 7). Proves a value that does not
    #     match its type is left as-is exactly as vLLM, while siblings coerce.
    MISMATCH_TOOL = [tool("configure", {
        "days": {"type": "integer"},
        "count": {"type": ["integer", "null"]},
        "n": {"type": "integer"},
    })]
    scenarios.append(("qwen3_schema_mismatch_wholedelta", "qwen3", False, True, [
        "<tool_call>\n<function=configure>\n",
        "<parameter=days>abc</parameter>\n",
        "<parameter=count>null</parameter>\n",
        "<parameter=n>7</parameter>\n",
        "</function>\n</tool_call>",
    ], MISMATCH_TOOL))
    # 30. kimi_k2 JSON-native args coerced in the one-shot extract path: the
    #     model emits `"days": "5"` (string) which the schema coerces to int 5.
    KIMI_TYPED = (
        "<think>picking a tool</think>"
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "functions.get_weather:0<|tool_call_argument_begin|>"
        '{"unit": "celsius", "days": "5"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    scenarios.append(("kimi_typed_schema_wholedelta", "kimi_k2", True, True, [
        "<think>picking a tool</think>",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>",
        '{"unit": "celsius", "days": "5"}',
        "<|tool_call_end|><|tool_calls_section_end|>",
    ], TYPED_TOOL))
    return scenarios


def make_parser(pe_pkg, cfg, thinking, tok):
    if cfg == "qwen3":
        return pe_pkg["qwen3"].Qwen3Parser(tok, chat_template_kwargs={
            "enable_thinking": thinking})
    if cfg == "seed_oss":
        return pe_pkg["seed_oss"].SeedOssParser(tok, chat_template_kwargs={
            "enable_thinking": thinking})
    if cfg == "kimi_k2":
        return pe_pkg["kimi_k2"].KimiK2Parser(tok, chat_template_kwargs={
            "thinking": thinking})
    if cfg == "minimax_m2":
        # Always reasoning-initial; no chat_template_kwargs consumed.
        return pe_pkg["minimax_m2"].MinimaxM2Parser(tok)
    if cfg == "glm47_moe":
        return pe_pkg["glm47_moe"].Glm47MoeParser(tok, chat_template_kwargs={
            "enable_thinking": thinking})
    if cfg == "deepseek_v4":
        return pe_pkg["deepseek_v4"].DeepSeekV4Parser(tok, chat_template_kwargs={
            "thinking": thinking})
    if cfg == "deepseek_v32":
        return pe_pkg["deepseek_v32"].DeepSeekV32Parser(tok)
    if cfg == "nemotron_v3":
        return pe_pkg["nemotron_v3"].NemotronV3Parser(tok, chat_template_kwargs={
            "enable_thinking": thinking})
    if cfg == "gemma4":
        return pe_pkg["gemma4"].Gemma4Parser(tok, chat_template_kwargs={
            "enable_thinking": thinking})
    if cfg == "inkling":
        return pe_pkg["inkling"].InklingParser(tok)
    raise SystemExit("unknown cfg: " + cfg)


# --------------------------------------------------------------------------- #
# .inc emission
# --------------------------------------------------------------------------- #
def esc(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append("\\x%02x" % ord(ch))
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def opt(s):
    return "std::nullopt" if s is None else "std::optional<std::string>(" + esc(s) + ")"


def emit_tc(tc):
    # streaming DeltaToolCall
    return "GTC{%d, %s, %s, %s, %s}" % (
        tc.index, opt(tc.id), opt(tc.type),
        opt(tc.function.name if tc.function else None),
        opt(tc.function.arguments if tc.function else None),
    )


def emit_delta(msg):
    if msg is None:
        return "GDelta{false, std::nullopt, std::nullopt, {}}"
    tcs = ""
    if msg.tool_calls:
        tcs = ", ".join(emit_tc(t) for t in msg.tool_calls)
    return "GDelta{true, %s, %s, {%s}}" % (opt(msg.content), opt(msg.reasoning), tcs)


def emit_extract(info):
    xtcs = ", ".join(
        'GXTC{%s, %s, %s}' % (esc(tc.id), esc(tc.function.name), esc(tc.function.arguments))
        for tc in info.tool_calls)
    return "GExtract{%s, %s, {%s}}" % (
        "true" if info.tools_called else "false", opt(info.content), xtcs)


def emit_tools(tools):
    # GTool{name, parameters_json} — parameters_json is the raw JSON schema the
    # C++ gate parses into ParserTool.parameters (empty string => no parameters).
    if not tools:
        return "{}"
    items = []
    for t in tools:
        params = t.function.parameters
        pj = "" if params is None else json.dumps(params)
        items.append("GTool{%s, %s}" % (esc(t.function.name), esc(pj)))
    return "{%s}" % ", ".join(items)


def emit_parse(check, reasoning, content, tool_calls):
    # `parse()` returns (reasoning, content, tool_calls|None). The C++ parse()
    # tuple carries FunctionCall{name, arguments} (no id), so gate name+arguments.
    if tool_calls is None:
        has, tcs = "false", ""
    else:
        has = "true"
        tcs = ", ".join('{%s, %s}' % (esc(t.name), esc(t.arguments))
                        for t in tool_calls)
    return "%s, GParse{%s, %s, %s, {%s}}" % (
        "true" if check else "false", opt(reasoning), opt(content), has, tcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-source", default=os.environ.get("VLLM_SOURCE",
                    "/home/mudler/_git/vllm"))
    args = ap.parse_args()

    tmp = build_sandbox(args.vllm_source)
    try:
        pe_pkg = {m: importlib.import_module("pe." + m) for m in FAMILY_MODULES}

        lines = []
        lines.append("// GENERATED by tools/parity/dump_parser_engine_assembly.py")
        lines.append("// Oracle: vLLM 0.26.0.dev0 @ 555967922 parser ASSEMBLY layer.")
        lines.append("// DO NOT EDIT — regenerate from the pinned oracle.")
        lines.append("static const std::vector<GScenario> kAssemblyGoldens = {")

        for scen in make_scenarios():
            name, cfg, thinking, incl, deltas = scen[:5]
            tools = scen[5] if len(scen) > 5 else None
            tok = MockTok()
            # --- streaming pass ---
            parser = make_parser(pe_pkg, cfg, thinking, tok)
            req = Req(include_reasoning=incl, tools=tools)
            stream_out = []
            for i, dt in enumerate(deltas):
                finished = (i == len(deltas) - 1)
                msg = parser.parse_delta(dt, [], req, finished=finished)
                stream_out.append(emit_delta(msg))
            # --- one-shot extract pass ---
            parser2 = make_parser(pe_pkg, cfg, thinking, tok)
            full = "".join(deltas)
            info = parser2.extract_tool_calls(
                full, Req(include_reasoning=incl, tools=tools))

            # --- non-streaming parse() pass (gates the gemma4/inkling seams that
            #     the streaming/extract paths do not reach: inkling's
            #     _single_pass_parse trailing flush). Checked only for the two new
            #     configs; captured for all for a uniform golden record. ---
            parser3 = make_parser(pe_pkg, cfg, thinking, tok)
            p_reasoning, p_content, p_tools = parser3.parse(
                full, Req(include_reasoning=incl, tools=tools))
            check_parse = cfg in ("gemma4", "inkling")

            deltas_cpp = ", ".join(esc(d) for d in deltas)
            stream_cpp = ",\n      ".join(stream_out)
            lines.append("  GScenario{")
            lines.append("    %s, %s, %s, %s," % (
                esc(name), esc(cfg),
                "true" if thinking else "false",
                "true" if incl else "false"))
            lines.append("    {%s}," % deltas_cpp)
            lines.append("    {\n      %s\n    }," % stream_cpp)
            lines.append("    %s," % emit_extract(info))
            lines.append("    %s," % emit_parse(
                check_parse, p_reasoning, p_content, p_tools))
            lines.append("    %s," % emit_tools(tools))
            lines.append("  },")

        lines.append("};")
        open(args.out, "w").write("\n".join(lines) + "\n")
        print("wrote", args.out, "(%d scenarios)" % len(make_scenarios()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
