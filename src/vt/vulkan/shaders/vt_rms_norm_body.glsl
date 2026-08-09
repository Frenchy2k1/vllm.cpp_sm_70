// vt::RmsNorm — the BODY, shared verbatim by every width variant.
//
// The two wrappers (vt_rms_norm.comp at VT_TG 128, vt_rms_norm_wide.comp at
// VT_TG 1024 + subgroup reduction) differ ONLY in the two macros they set and in
// their `local_size_x` literal; the math below is compiled from this one text, so
// the variants cannot drift apart. Read vt_rms_norm.comp for the port provenance
// and the numeric decisions.
#include "vt_common.glsl"

layout(binding = 0) readonly buffer Xb32 { uint v[]; } X32;
layout(binding = 1) readonly buffer Xb16 { uint16_t v[]; } X16;
layout(binding = 2) readonly buffer Wb32 { uint v[]; } W32;
layout(binding = 3) readonly buffer Wb16 { uint16_t v[]; } W16;
layout(binding = 4) buffer Db32 { uint v[]; } D32;
layout(binding = 5) buffer Db16 { uint16_t v[]; } D16;
layout(binding = 6) buffer Rb32 { uint v[]; } R32;
layout(binding = 7) buffer Rb16 { uint16_t v[]; } R16;

layout(push_constant) uniform Params {
  uint t;
  uint h;
  uint x_dt;
  uint w_dt;
  uint out_dt;
  uint res_dt;
  uint has_res;
  uint gemma;
  uint x_off;
  uint w_off;
  uint out_off;
  uint res_off;
  float eps;
} p;

// The residual store's ROUNDING, as a pure function of the f32 value.
//
// This is the whole of what the old store-then-RE-READ idiom recovered: a store
// through VT_STORE rounds into `dt` and a load through VT_LOAD widens back, and
// both halves are pure. Evaluating them in registers is therefore BIT-IDENTICAL
// to the memory round trip (VT_DT_F32 stores floatBitsToUint and loads
// uintBitsToFloat, which is the identity), while removing a store->load
// dependency on the SAME address from every iteration of the accumulation loop.
float vt_round_through(float v, uint dt) {
  return dt == VT_DT_F32 ? v : vt_from16(vt_to16(v, dt), dt);
}

void main() {
  uint row = gl_WorkGroupID.x;
  uint tid = gl_LocalInvocationID.x;
  uint rbase = row * p.h;

  float partial = 0.0;
  for (uint j = tid; j < p.h; j += VT_TG) {
    float v = VT_LOAD(X32, X16, p.x_dt, p.x_off, rbase + j);
    if (p.has_res != 0u) {
      v += VT_LOAD(R32, R16, p.res_dt, p.res_off, rbase + j);
      VT_STORE(R32, R16, p.res_dt, p.res_off, rbase + j, v);
      v = vt_round_through(v, p.res_dt);  // == re-reading the value just stored
    }
    partial += v * v;
  }
  // The SECOND loop reads back slots this same invocation wrote above, at the
  // same `j` stride, so program order inside one invocation already orders them
  // and no memory barrier is owed. (The version before VK-RMSNORM issued a
  // memoryBarrierBuffer() here because the FIRST loop also read a just-stored
  // value; vt_round_through removed that read, and with it the reason.)

  float sumsq = vt_tg_sum(tid, partial);
  float inv = 1.0 / sqrt(sumsq / float(p.h) + p.eps);

  for (uint j = tid; j < p.h; j += VT_TG) {
    float v = p.has_res != 0u ? VT_LOAD(R32, R16, p.res_dt, p.res_off, rbase + j)
                              : VT_LOAD(X32, X16, p.x_dt, p.x_off, rbase + j);
    float wj = VT_LOAD(W32, W16, p.w_dt, p.w_off, j);
    if (p.gemma != 0u) { wj += 1.0; }
    VT_STORE(D32, D16, p.out_dt, p.out_off, rbase + j, v * inv * wj);
  }
}
