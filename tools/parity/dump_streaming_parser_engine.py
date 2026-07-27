# vllm.cpp parity harness; oracle = upstream vLLM unified parser engine.
"""Capture golden SemanticEvent sequences from the vLLM 0.26 streaming parser
engine and emit the C++ gate's ``goldens.inc``.

The parser engine is a PURE FUNCTION of the (delta_text, delta_token_ids)
stream, so this needs no GPU, no model and no installed vLLM wheel: it copies
the six self-contained engine modules from the pinned vLLM source tree
(``vllm/parser/engine/{events,token_id_scanner,incremental_lexer,
parser_engine_config,streaming_parser_engine}.py``), rewrites their intra-engine
imports to a local package, rebuilds the qwen3 + kimi_k2 configs inline (only
the streaming-relevant fields; the arg_converter and assembly-layer flags never
reach the streaming engine), drives the identical delta streams the C++ gate
uses, and writes the expected event arrays.

Usage::

    VLLM_SOURCE=/home/mudler/_git/vllm python3 \
      tools/parity/dump_streaming_parser_engine.py \
      --out tests/vllm/parser/engine/test_streaming_parser_engine_goldens.inc

Regenerate whenever the pin advances and re-run the C++ gate
``test_streaming_parser_engine``; any changed event sequence is a real
divergence to reconcile 1:1 with upstream.
"""

import argparse
import importlib.util
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
]


def load_engine(vllm_source):
    src = os.path.join(vllm_source, "vllm", "parser", "engine")
    tmp = tempfile.mkdtemp(prefix="peref_")
    pkg = os.path.join(tmp, "pe")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for m in ENGINE_MODULES:
        text = open(os.path.join(src, m + ".py")).read()
        text = text.replace("from vllm.parser.engine.", "from pe.")
        open(os.path.join(pkg, m + ".py"), "w").write(text)
    sys.path.insert(0, tmp)
    spe = importlib.import_module("pe.streaming_parser_engine")
    cfgmod = importlib.import_module("pe.parser_engine_config")
    evmod = importlib.import_module("pe.events")
    return spe, cfgmod, evmod, tmp


def build_configs(cfgmod, evmod):
    PEC = cfgmod.ParserEngineConfig
    PS = cfgmod.ParserState
    T = cfgmod.Transition
    E = evmod.EventType

    def qwen3(thinking, name="qwen3", ts="<think>", te="</think>",
              tcs="<tool_call>", tce="</tool_call>"):
        return PEC(
            name=name,
            initial_state=PS.REASONING if thinking else PS.CONTENT,
            terminals={"THINK_START": ts, "THINK_END": te, "TOOL_START": tcs,
                       "TOOL_END": tce, "FUNC_PREFIX": "<function=",
                       "FUNC_END": "</function>", "PARAM_START": "<parameter=",
                       "PARAM_END": "</parameter>", "CLOSE_ANGLE": ">"},
            token_id_terminals={"THINK_START": ts, "THINK_END": te,
                                "TOOL_START": tcs, "TOOL_END": tce},
            transitions={
                (PS.REASONING, "THINK_START"): T(PS.REASONING, ()),
                (PS.REASONING, "THINK_END"): T(PS.CONTENT, (E.REASONING_END,)),
                (PS.CONTENT, "THINK_END"): T(PS.CONTENT, ()),
                (PS.REASONING, "TOOL_START"): T(PS.TOOL_PREAMBLE, (E.REASONING_END, E.TOOL_CALL_START)),
                (PS.CONTENT, "TOOL_START"): T(PS.TOOL_PREAMBLE, (E.REASONING_END, E.TOOL_CALL_START)),
                (PS.CONTENT, "FUNC_PREFIX"): T(PS.TOOL_NAME, (E.TOOL_CALL_START,)),
                (PS.TOOL_PREAMBLE, "TOOL_END"): T(PS.CONTENT, (E.TOOL_CALL_END,)),
                (PS.TOOL_PREAMBLE, "FUNC_PREFIX"): T(PS.TOOL_NAME, ()),
                (PS.TOOL_NAME, "CLOSE_ANGLE"): T(PS.TOOL_ARGS, ()),
                (PS.TOOL_NAME, "FUNC_END"): T(PS.TOOL_BETWEEN, (E.TOOL_CALL_END,)),
                (PS.TOOL_ARGS, "FUNC_END"): T(PS.TOOL_BETWEEN, (E.TOOL_CALL_END,)),
                (PS.TOOL_ARGS, "PARAM_START"): T(PS.TOOL_ARGS, (E.ARG_VALUE_CHUNK,)),
                (PS.TOOL_ARGS, "PARAM_END"): T(PS.TOOL_ARGS, (E.ARG_VALUE_CHUNK,)),
                (PS.TOOL_BETWEEN, "TOOL_END"): T(PS.CONTENT, ()),
                (PS.TOOL_BETWEEN, "TOOL_START"): T(PS.TOOL_PREAMBLE, (E.TOOL_CALL_START,)),
                (PS.TOOL_BETWEEN, "FUNC_PREFIX"): T(PS.TOOL_NAME, (E.TOOL_CALL_START,)),
            },
            tool_args_json=False,
        )

    def kimi(thinking):
        rt = {"THINK_START": "<think>", "THINK_END": "</think>"} if thinking else {}
        rtr = ({
            (PS.REASONING, "THINK_START"): T(PS.REASONING, ()),
            (PS.REASONING, "THINK_END"): T(PS.CONTENT, (E.REASONING_END,)),
            (PS.CONTENT, "THINK_END"): T(PS.CONTENT, ()),
        } if thinking else {})
        return PEC(
            name="kimi_k2",
            initial_state=PS.REASONING if thinking else PS.CONTENT,
            terminals={**rt, "TOOL_SECTION_START": "<|tool_calls_section_begin|>",
                       "TOOL_SECTION_END": "<|tool_calls_section_end|>",
                       "TOOL_START": "<|tool_call_begin|>", "TOOL_END": "<|tool_call_end|>",
                       "ARG_START": "<|tool_call_argument_begin|>"},
            token_id_terminals={**rt, "TOOL_SECTION_START": "<|tool_calls_section_begin|>",
                                "TOOL_SECTION_END": "<|tool_calls_section_end|>",
                                "TOOL_START": "<|tool_call_begin|>", "TOOL_END": "<|tool_call_end|>",
                                "ARG_START": "<|tool_call_argument_begin|>"},
            transitions={
                **rtr,
                (PS.REASONING, "TOOL_SECTION_START"): T(PS.TOOL_PREAMBLE, (E.REASONING_END,)),
                (PS.CONTENT, "TOOL_SECTION_START"): T(PS.TOOL_PREAMBLE, ()),
                (PS.TOOL_PREAMBLE, "TOOL_START"): T(PS.TOOL_NAME, (E.TOOL_CALL_START,)),
                (PS.TOOL_NAME, "ARG_START"): T(PS.TOOL_ARGS, ()),
                (PS.TOOL_ARGS, "TOOL_END"): T(PS.TOOL_BETWEEN, (E.TOOL_CALL_END,)),
                (PS.TOOL_ARGS, "TOOL_SECTION_END"): T(PS.TOOL_PREAMBLE, (E.TOOL_CALL_END,)),
                (PS.TOOL_BETWEEN, "TOOL_START"): T(PS.TOOL_NAME, (E.TOOL_CALL_START,)),
                (PS.TOOL_PREAMBLE, "TOOL_SECTION_END"): T(PS.TOOL_PREAMBLE, ()),
                (PS.TOOL_BETWEEN, "TOOL_SECTION_END"): T(PS.TOOL_PREAMBLE, ()),
            },
            tool_args_json=True,
        )

    return {"qwen3_think": qwen3(True), "qwen3_nothink": qwen3(False),
            "kimi_think": kimi(True)}


class MockTok:
    def __init__(self, vocab):
        self._v = dict(vocab)
        self._r = {b: a for a, b in vocab.items()}

    def get_vocab(self):
        return self._v

    def decode(self, ids):
        return "".join(self._r.get(i, "") for i in ids)


def chunk(s, n):
    return [s[i:i + n] for i in range(0, len(s), n)]


def scenarios():
    txt = ("<think>let me think</think>Here you go<tool_call>\n"
           "<function=get_weather>\n<parameter=city>Paris</parameter>\n"
           "<parameter=unit>celsius</parameter>\n</function>\n</tool_call>")
    txt2 = ("<tool_call>\n<function=a>\n<parameter=x>1</parameter>\n</function>\n</tool_call>"
            "<tool_call>\n<function=b>\n<parameter=y>2</parameter>\n</function>\n</tool_call>")
    txt3 = "<tool_call>\n<function=partial>\n<parameter=k>val"
    ktxt = ("<think>reason</think><|tool_calls_section_begin|><|tool_call_begin|>"
            "functions.get_weather:0<|tool_call_argument_begin|>"
            '{"city": "Tokyo", "n": 3}<|tool_call_end|><|tool_calls_section_end|>')
    vocab = {"<think>": 1001, "</think>": 1002,
             "<|tool_calls_section_begin|>": 1003, "<|tool_calls_section_end|>": 1004,
             "<|tool_call_begin|>": 1005, "<|tool_call_end|>": 1006,
             "<|tool_call_argument_begin|>": 1007}
    kdeltas = [("<think>", [1001]), ("weighing", []), ("</think>", [1002]),
               ("<|tool_calls_section_begin|>", [1003]), ("<|tool_call_begin|>", [1005]),
               ("functions.get_weather:0", []), ("<|tool_call_argument_begin|>", [1007]),
               ('{"city": "Tokyo"}', []), ("<|tool_call_end|>", [1006]),
               ("<|tool_calls_section_end|>", [1004])]
    return [
        ("qwen3_xml_char3", "qwen3_think", [(c, []) for c in chunk(txt, 3)], None),
        ("qwen3_xml_whole", "qwen3_think", [(txt, [])], None),
        ("qwen3_nothink", "qwen3_nothink", [("Just plain text answer.", [])], None),
        ("qwen3_two_calls", "qwen3_nothink", [(t, []) for t in chunk(txt2, 5)], None),
        ("qwen3_unfinished", "qwen3_nothink", [(t, []) for t in chunk(txt3, 4)], None),
        ("kimi_json_char2", "kimi_think", [(c, []) for c in chunk(ktxt, 2)], None),
        ("kimi_json_whole", "kimi_think", [(ktxt, [])], None),
        ("kimi_tokenid", "kimi_think", kdeltas, MockTok(vocab)),
    ]


def cesc(s):
    out = []
    for b in s.encode("utf-8"):
        c = chr(b)
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif c == "\r":
            out.append("\\r")
        elif 32 <= b < 127:
            out.append(c)
        else:
            out.append("\\x%02x" % b)
    return '"' + "".join(out) + '"'


HEADER = (
    "  // GENERATED golden event sequences captured from vLLM 0.26.0.dev0\n"
    "  // (@ 555967922) vllm/parser/engine/streaming_parser_engine.py driven by\n"
    "  // vllm/parser/qwen3.py + vllm/parser/kimi_k2.py configs (tokenizer=None for\n"
    "  // the text-only scenarios; a fixed mock vocab for kimi_tokenid). See\n"
    "  // .agents/specs/streaming-parser-engine.md. DO NOT hand-edit: this is\n"
    "  // the exact-parity oracle.\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-source", default=os.environ.get("VLLM_SOURCE", "/home/mudler/_git/vllm"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spe, cfgmod, evmod, tmp = load_engine(args.vllm_source)
    try:
        configs = build_configs(cfgmod, evmod)
        lines = [HEADER.rstrip("\n")]
        first = True
        for (name, tag, deltas, tok) in scenarios():
            eng = spe.StreamingParserEngine(configs[tag], tok)
            events = []
            for (dt, ids) in deltas:
                for e in eng.feed(dt, ids):
                    events.append((e.type.name, e.value, e.tool_index))
            for e in eng.finish():
                events.append((e.type.name, e.value, e.tool_index))
            block = ["  scenarios.push_back(Scenario{"]
            block.append('    "%s", "%s", %s,' % (name, tag, "true" if tok is not None else "false"))
            block.append("    /*deltas=*/{")
            for (dt, ids) in deltas:
                block.append("      {%s, {%s}}," % (cesc(dt), ", ".join(str(i) for i in ids)))
            block.append("    },")
            block.append("    /*expected=*/{")
            for (t, v, idx) in events:
                block.append("      {E::%s, %s, %d}," % (t, cesc(v), idx))
            block.append("    },")
            block.append("  });")
            if first:
                # keep the header attached to the first push_back block
                block[0] = block[0]
                first = False
            lines.append("\n".join(block))
        open(args.out, "w").write("\n".join(lines) + "\n")
        print("wrote", args.out, "with 8 scenarios")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
