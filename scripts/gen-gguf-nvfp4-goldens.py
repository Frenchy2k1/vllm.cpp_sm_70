#!/usr/bin/env python3
"""Emit tests/vllm/gguf_nvfp4_goldens.inc: real NVFP4 bytes from BOTH containers
for the same Qwen3.6-27B weights, so the C++ suite can gate cross-format
equivalence with no multi-GB asset checked in.

    python3 scripts/gen-gguf-nvfp4-goldens.py \
        ~/bench/q36-27b-nvfp4.gguf \
        ~/bench/q36-27b-nvfp4-vllm/model.safetensors \
        > tests/vllm/gguf_nvfp4_goldens.inc

Needs numpy only for the exact float32 reciprocal assertion. Deliberately does
NOT use gguf-py: the GGUF v2/v3 header walk below is self-contained, so the
generator runs anywhere the two files do.
"""
import sys, json, struct
import numpy as np

GGUF_MAGIC = 0x46554747

# value types
UINT8,INT8,UINT16,INT16,UINT32,INT32,FLOAT32,BOOL,STRING,ARRAY,UINT64,INT64,FLOAT64 = range(13)
FMT = {UINT8:('<B',1),INT8:('<b',1),UINT16:('<H',2),INT16:('<h',2),UINT32:('<I',4),
       INT32:('<i',4),FLOAT32:('<f',4),BOOL:('<?',1),UINT64:('<Q',8),INT64:('<q',8),
       FLOAT64:('<d',8)}

class R:
    def __init__(self, f):
        self.f = f
    def raw(self, n):
        b = self.f.read(n)
        assert len(b) == n, "short read"
        return b
    def u32(self): return struct.unpack('<I', self.raw(4))[0]
    def u64(self): return struct.unpack('<Q', self.raw(8))[0]
    def string(self, ver):
        n = self.u64() if ver >= 2 else self.u32()
        return self.raw(n).decode('utf-8', 'replace')
    def value(self, t, ver):
        if t in FMT:
            fmt, sz = FMT[t]
            return struct.unpack(fmt, self.raw(sz))[0]
        if t == STRING:
            return self.string(ver)
        if t == ARRAY:
            et = self.u32()
            n = self.u64() if ver >= 2 else self.u32()
            if et in FMT:
                fmt, sz = FMT[et]
                buf = self.raw(sz*n)
                return list(struct.unpack('<%d%s' % (n, fmt[1]), buf))
            elif et == STRING:
                return [self.string(ver) for _ in range(n)]
            else:
                raise RuntimeError("nested array type %d" % et)
        raise RuntimeError("bad kv type %d" % t)

def dump(path):
    with open(path, 'rb') as f:
        r = R(f)
        magic = r.u32()
        assert magic == GGUF_MAGIC, hex(magic)
        ver = r.u32()
        n_tensors = r.u64() if ver >= 2 else r.u32()
        n_kv = r.u64() if ver >= 2 else r.u32()
        kv = {}
        for _ in range(n_kv):
            k = r.string(ver)
            t = r.u32()
            v = r.value(t, ver)
            kv[k] = (t, v)
        tensors = []
        for _ in range(n_tensors):
            name = r.string(ver)
            nd = r.u32()
            dims = [r.u64() if ver >= 2 else r.u32() for _ in range(nd)]
            tt = r.u32()
            off = r.u64()
            tensors.append(dict(name=name, dims=dims, type=tt, offset=off))
        align = kv.get('general.alignment', (None, 32))[1]
        data_start = f.tell()
        if data_start % align:
            data_start += align - (data_start % align)
        return dict(version=ver, kv=kv, tensors=tensors, align=align,
                    data_start=data_start)



GGUF = sys.argv[1] if len(sys.argv) > 1 else "/home/mudler/bench/q36-27b-nvfp4.gguf"
ST = sys.argv[2] if len(sys.argv) > 2 else "/home/mudler/bench/q36-27b-nvfp4-vllm/model.safetensors"

d = dump(GGUF); tmap = {t['name']: t for t in d['tensors']}; ds = d['data_start']
gf = open(GGUF, 'rb'); sf = open(ST, 'rb')
n = struct.unpack("<Q", sf.read(8))[0]; sh = json.loads(sf.read(n)); sdata = 8 + n

def gg(nm, off, cnt):
    t = tmap[nm]; gf.seek(ds + t['offset'] + off); return gf.read(cnt)
def stb(nm, off, cnt):
    e = sh[nm]; sf.seek(sdata + e['data_offsets'][0] + off); return sf.read(cnt)

# NOTE: only the MLP projections are compared raw. blk.N.ssm_out additionally
# carries the GGUF v-head TILING (tiled head r*num_k+k == HF grouped head
# k*num_v_per_k+r), which the loader undoes with ReorderVCols; a raw byte
# comparison there is expected to differ and is not a container question.
CASES = [
  ("blk.0.ffn_gate.weight", "model.language_model.layers.0.mlp.gate_proj"),
  ("blk.7.ffn_up.weight",   "model.language_model.layers.7.mlp.up_proj"),
  ("blk.63.ffn_down.weight","model.language_model.layers.63.mlp.down_proj"),
]
ROWS = 4
COLS = 512          # in-dim slice: 8 gguf blocks / 256 st packed bytes / 32 scales

def carr(name, b):
    out = ["static const uint8_t %s[%d] = {" % (name, len(b))]
    for i in range(0, len(b), 16):
        out.append("    " + " ".join("0x%02x," % x for x in b[i:i+16]))
    out.append("};")
    return "\n".join(out)

blocks = []
entries = []
for idx, (gname, stem) in enumerate(CASES):
    t = tmap[gname]
    in_dim = t['dims'][0]
    nblk_full = in_dim // 64
    gb = b"".join(gg(gname, r*nblk_full*36, (COLS//64)*36) for r in range(ROWS))
    sp = b"".join(stb(stem+".weight_packed", r*(in_dim//2), COLS//2) for r in range(ROWS))
    ss = b"".join(stb(stem+".weight_scale", r*(in_dim//16), COLS//16) for r in range(ROWS))
    wgs = struct.unpack('<f', stb(stem+".weight_global_scale", 0, 4))[0]
    inv = np.float32(1.0) / np.float32(wgs)
    gsc_bytes = gg(gname.rsplit('.weight',1)[0]+".scale", 0, 4)
    assert gsc_bytes == struct.pack('<f', float(inv)), (gname, gsc_bytes.hex(), struct.pack('<f',float(inv)).hex())
    bits = struct.unpack('<I', gsc_bytes)[0]
    tag = "c%d" % idx
    blocks.append(carr("k_%s_gguf" % tag, gb))
    blocks.append(carr("k_%s_st_packed" % tag, sp))
    blocks.append(carr("k_%s_st_scales" % tag, ss))
    entries.append('    {"%s", %d, %d, 0x%08xU, k_%s_gguf, k_%s_st_packed, k_%s_st_scales},'
                   % (gname, ROWS, COLS, bits, tag, tag, tag))
    print("case", gname, "wgs", wgs, "1/wgs bits 0x%08x" % bits, file=sys.stderr)

hdr = '''// GENERATED by scripts/gen-gguf-nvfp4-goldens.py — DO NOT EDIT BY HAND.
//
// Real NVFP4 bytes for the SAME weight slices of Qwen3.6-27B-NVFP4, taken from
// BOTH containers, so the cross-format equivalence gate needs no multi-GB asset:
//
//   * `gguf`      the ggml type-40 blocks from Qwen3.6-27B-NVFP4.gguf
//                 (4 fp8-e4m3 sub-block scales, then 32 packed-nibble bytes)
//   * `st_packed` / `st_scales`  the compressed-tensors `weight_packed` /
//                 `weight_scale` bytes of the same [4, 512] slice of the same
//                 projection, from Qwen3.6-27B-NVFP4 (nvfp4-pack-quantized)
//   * `global_scale_bits`  f32 bits of the GGUF `<stem>.scale` sidecar, which
//                 the generator ASSERTS is bit-identical to
//                 float32(1) / float32(weight_global_scale) on the safetensors
//                 side (this is the `weight_scale_2` multiply-not-divide form).
//
// The two containers hold BIT-IDENTICAL scale bytes and a PERMUTATION of the
// same nibbles, so the two dequant paths must agree exactly, not approximately.
// See .agents/specs/gguf-nvfp4-notes.md Sec 5.

namespace gguf_nvfp4_goldens {

struct Case {
  const char* name;
  int64_t out_dim;
  int64_t in_dim;
  uint32_t global_scale_bits;
  const uint8_t* gguf;       // out_dim * in_dim/64 * 36
  const uint8_t* st_packed;  // out_dim * in_dim/2
  const uint8_t* st_scales;  // out_dim * in_dim/16
};

'''
ftr = "\nstatic const Case kCases[] = {\n%s\n};\n\n}  // namespace gguf_nvfp4_goldens\n" % "\n".join(entries)
sys.stdout.write(hdr + "\n\n".join(blocks) + ftr)
