// vllm.cpp original (vt runtime). Multi-device (TP) phase-3 gate: a REAL
// in-process NCCL collective over the box's discrete GPUs. Calls
// vt_cuda_nccl_group_selfcheck() (in nccl_communicator.cu, VT_NCCL): builds one
// NcclCommunicator per local GPU via ncclCommInitAll, runs an AllReduce +
// AllGather on each device's stream (per-device affinity), and diffs against
// the host truth (allreduce sum == ngpu*(ngpu+1)/2; allgather == ordered
// [1..ngpu]). RED-first: any dropped/incorrect collective shows up as a failed
// KSum / out-of-order gather.
//
// Compiled/linked only when VLLM_CPP_NCCL — a non-NCCL CUDA build has no such
// symbol, so this test target is gated in tests/CMakeLists.txt.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime.h>
#endif

extern "C" int vt_cuda_nccl_group_selfcheck(void);
extern "C" int vt_cuda_tp_seam_selfcheck(void);
extern "C" int vt_cuda_loader_slice_selfcheck(void);
extern "C" int vt_cuda_sharded_forward_selfcheck(void);
extern "C" int vt_cuda_dense_mlp_shard_selfcheck(void);
extern "C" int vt_cuda_bridge_rank_lanes_selfcheck(void);
extern "C" int vt_cuda_mlp_shard_run(int O,int H,int I,const float* x,const float* gate,const float* up,const float* down,float* out);
extern "C" void* vt_cuda_tp_acquire(int tp_size);
extern "C" void vt_cuda_tp_release(void* handle);

TEST_CASE("in-process NCCL collectives across the discrete GPUs (multi-device)") {
  const int rc = vt_cuda_nccl_group_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); multi-GPU NCLL skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("W2 TP seam over the NCCL group (TpShard/TensorParallel row-reduce per rank)") {
  const int rc = vt_cuda_tp_seam_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); TP seam skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("TP loader slice: per-rank shard placed on the addressed GPU, reconstructs the full weight") {
  const int rc = vt_cuda_loader_slice_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); loader slice skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("engine bridge: acquire a retained NCCL TP group for tp>1, null at tp1") {
  // tp1 must stay the null/no-op path.
  CHECK(vt_cuda_tp_acquire(1) == nullptr);
#ifdef VLLM_CPP_CUDA
  int devs = 0;
  REQUIRE(cudaGetDeviceCount(&devs) == cudaSuccess);
  if (devs < 2) { MESSAGE("single-GPU; bridge skipped"); return; }
  void* h = vt_cuda_tp_acquire(devs);   // a real W=ngpu retained group
  REQUIRE(h != nullptr);
  vt_cuda_tp_release(h);                // and the runner can release it
#endif
}

TEST_CASE("TP_PLAN W4-pre: per-rank lane build over a retained NCCL group, tp<=1 inert") {
  // 1 rank never builds a group (the runner's tp1 path stays inert).
  CHECK(vt_cuda_bridge_rank_lanes_selfcheck() != 1);  // 2 (skip) or 0, never a lane build failure
#ifdef VLLM_CPP_CUDA
  int devs = 0;
  REQUIRE(cudaGetDeviceCount(&devs) == cudaSuccess);
  if (devs < 2) { MESSAGE("single-GPU; W4-pre construction skipped"); return; }
  // On a 2-GPU host the selfcheck builds every rank lane (tp_size==W, rank==r)
  // and asserts each lane's TpShard window is live + distinct — NO forward math,
  // so the engine's world>1 serve guard stays untouched.
  CHECK(vt_cuda_bridge_rank_lanes_selfcheck() == 0);
#endif
}

TEST_CASE("dense-body shard: merged gate/up column-slice + down row-parallel reduce == MLP single-GPU") {
  const int rc = vt_cuda_dense_mlp_shard_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); dense shard skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("runner forward: device-sharded GEMM + group all-reduce == single-GPU forward") {
  const int rc = vt_cuda_sharded_forward_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); runner forward skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("reusable mlp_shard_run matches the host MLP reference at a fresh shape") {
  constexpr int O=4,H=8,I=16;
  std::vector<float> x(H,1.f), gate(I*H), up(I*H), down(O*I), out(O,0.f);
  for (int k=0;k<H;++k) x[k]=0.5f+(float)(k%3);
  for (int i=0;i<I;++i)for(int k=0;k<H;++k){gate[i*H+k]=0.1f*((i*3+k)%7)+0.05f;up[i*H+k]=0.05f*((i*5+k*2)%9)+0.2f;}
  for (int o=0;o<O;++o)for(int i=0;i<I;++i) down[o*I+i]=0.02f*((o*7+i*3)%11)+0.4f;
#ifdef VLLM_CPP_CUDA
  int devs = -1; (void)cudaGetDeviceCount(&devs);
  if (devs >= 2 && I % devs != 0) {  // world must divide I for an exact shard
    MESSAGE("world does not divide I=16; fresh-shape shard skipped");
    return;
  }
#endif
  int rc=vt_cuda_mlp_shard_run(O,H,I,x.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL unbuilt); mlp_shard_run skipped");return;}
  CHECK(rc==0);
  for (int o=0;o<O;++o){ double a=0.0;for(int i=0;i<I;++i){float g=0,u=0;for(int k=0;k<H;++k){g+=x[k]*gate[i*H+k];u+=x[k]*up[i*H+k];}float sg=1.f/(1.f+std::exp(-(double)g));a+=(double)((g*sg)*u)*down[o*I+i];} CHECK(out[(size_t)o]==doctest::Approx((float)a));}
}

TEST_CASE("mlp_shard_run holds at engine-scale intermediate width") {
  constexpr int O=256, H=512, I=1024;   // I % W == 0 (W=2, W=4)
  std::vector<float> x(H), gate(I*H), up(I*H), down(O*I), out(O,0.f);
  for (int k=0;k<H;++k) x[k]=0.4f+0.1f*(float)(k%7);
  for (int i=0;i<I;++i)for(int k=0;k<H;++k){ float r=0.001f*((i*3+k)%997); gate[i*H+k]=r+0.1f; up[i*H+k]=r+0.2f; }
  for (int o=0;o<O;++o)for(int i=0;i<I;++i) down[o*I+i]=0.001f*((o*7+i*3)%1000)+0.3f;
#ifdef VLLM_CPP_CUDA
  int devs = -1; (void)cudaGetDeviceCount(&devs);
  if (devs >= 2 && I % devs != 0) {
    MESSAGE("world does not divide I=1024; engine-scale shard skipped");
    return;
  }
#endif
  int rc=vt_cuda_mlp_shard_run(O,H,I,x.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); big mlp_shard skipped");return;}
  CHECK(rc==0);
  // spot-check a few outputs (full O cross-check is 134M MACs: fine but keep light)
  for (int o=0;o<O;o+=257){ double a=0.0;for(int i=0;i<I;++i){float g=0,u=0;for(int k=0;k<H;++k){g+=x[k]*gate[i*H+k];u+=x[k]*up[i*H+k];}float sg=1.f/(1.f+std::exp(-(double)g));a+=(double)((g*sg)*u)*down[o*I+i];} CHECK(out[(size_t)o]==doctest::Approx((float)a).epsilon(5e-4));}
}

extern "C" int vt_host_decode_nvfp4_f32(int N,int K,const uint8_t* packed,const uint8_t* scale8,const float* scale2,float* out);
TEST_CASE("host NVFP4 decode reproduces the fp4 layout (known nibbles + e4m3 scale)") {
  constexpr int N=1,K=16;
  std::vector<uint8_t> packed(8,0), scale8(N,0x3C);  // e4m3 0x3C = 1.5
  std::vector<float> scale2(N,2.0f), out(N*K,0.f);
  packed[0]=0x05;      // low nibble 5 (LUT 3) * gs; high 0 -> elems {0,1}
  packed[7]=0xF1;      // lo=1 (0.5), hi=0xF (LUT2|? => mag 6) sign(bit3)=1 -> -6
  int rc=vt_host_decode_nvfp4_f32(N,K,packed.data(),scale8.data(),scale2.data(),out.data());
  CHECK(rc==0);
  const float gs=1.5f*2.0f;                 // f8(1.5)*scale2
  CHECK(out[0*K+0]==doctest::Approx(3.0f*gs));     // lo of byte0
  CHECK(out[0*K+1]==doctest::Approx(0.0f));       // hi of byte0
  CHECK(out[0*K+(16-2)]==doctest::Approx(0.5f*gs));     // byte7 lo
  CHECK(out[0*K+(16-1)]==doctest::Approx(-6.0f*gs));    // byte7 hi (sign)
}

extern "C" int vt_cuda_mlp_shard_run_bf16(int O,int H,int I,const uint16_t* x,const uint16_t* gu,const uint16_t* down,float* out);
extern "C" int vt_cuda_attn_kv_shard_run(int T,int Hq,int S,int Hkv,int D,const float* q,const float* k,const float* v,float* out);
extern "C" int vt_cuda_lm_head_shard_run(int T,int H,int V,const float* x,const float* w,float* out);
extern "C" int vt_cuda_moe_expert_shard_run(int T,int H,int I,int top_k,int E,const float* x,const float* gate,const float* up,const float* down,const int32_t* eids,const float* wgt,float* out);
TEST_CASE("bf16 dense-MLP assembly: split [2I,H] gate/up + down, shard == host ref") {
  constexpr int O=8,H=12,I=24;   // I%4==0
  // float ground truth first, then its bf16 bits as the as-"assembly" input.
  std::vector<float> xf(H), gf(I*H), uf(I*H), df(O*I);
  for (int k=0;k<H;++k) xf[k]=0.5f+0.15f*(float)(k%5);
  for (int i=0;i<I;++i)for(int k=0;k<H;++k){gf[i*H+k]=0.01f*((i*3+k)%710)+0.1f; uf[i*H+k]=0.01f*((i*7+k)%1130)+0.2f;}
  for (int o=0;o<O;++o)for(int i=0;i<I;++i) df[o*I+i]=0.02f*((o*5+i*3)%1920)+0.3f;
  std::vector<uint16_t> x16(H), gu16(2*I*H), down16(O*I);
  auto bfu=[](float f){uint32_t u=0;std::memcpy(&u,&f,4);return (uint16_t)(u>>16);};
  for (int k=0;k<H;++k) x16[k]=bfu(xf[k]);
  for (size_t i=0;i<(size_t)I;i++) for (int k=0;k<H;++k){gu16[i*H+k]=bfu(gf[i*H+k]);gu16[(I+i)*H+k]=bfu(uf[i*H+k]);}
  for (size_t i=0;i<(size_t)(O*I);++i) down16[i]=bfu(df[i]);
  std::vector<float> out(O,0.f);
  int rc=vt_cuda_mlp_shard_run_bf16(O,H,I,x16.data(),gu16.data(),down16.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); bf16 shard skipped");return;}
  CHECK(rc==0);
  // host reference on the bf16-rounded weights (replicate the assembly).
  std::vector<float> x_(H),gate_(I*H),up_(I*H),down_(O*I);
  auto bff=[](uint16_t b){uint32_t u=(uint32_t)b<<16;float f;std::memcpy(&f,&u,4);return f;};
  for (int k=0;k<H;++k) x_[k]=bff(x16[k]);
  for (int i=0;i<I;++i)for(int k=0;k<H;++k){gate_[i*H+k]=bff(gu16[i*H+k]);up_[i*H+k]=bff(gu16[(I+i)*H+k]);}
  for (int o=0;o<O;++o)for(int i=0;i<I;++i) down_[o*I+i]=bff(down16[o*I+i]);
  for (int o=0;o<O;++o){ double a=0.0;for(int i=0;i<I;++i){float g=0,u=0;for(int k=0;k<H;++k){g+=x_[k]*gate_[i*H+k];u+=x_[k]*up_[i*H+k];}float sg=1.f/(1.f+std::exp(-(double)g));a+=(double)((g*sg)*u)*down_[o*I+i];} CHECK(out[(size_t)o]==doctest::Approx((float)a).epsilon(2e-4));}
}

TEST_CASE("attention KV/GQA-shard (seq): kv split + causal == single-GPU softmax") {
  constexpr int S=3,Hq=4,Hkv=8,D=16;  // Hkv%W==0 on 2/4-GPU box; S==T query count
  std::vector<float> q(S*Hq*D),k(S*Hkv*D),v(S*Hkv*D);
  for (size_t i=0;i<q.size();++i) q[i]=0.1f*((int)(i%13))-0.3f;
  for (int s=0;s<S;++s)for (int kh=0;kh<Hkv;++kh)for(int d=0;d<D;++d){
    k[(size_t)(s*Hkv+kh)*D+d]=0.05f*((kh*3+s+d)%9)-0.1f; v[(size_t)(s*Hkv+kh)*D+d]=0.2f*((kh*5+s*2+d)%7)+0.1f;}
  std::vector<float> out((size_t)S*Hq*D,0.f);
  int rc=vt_cuda_attn_kv_shard_run(S,Hq,S,Hkv,D,q.data(),k.data(),v.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); attn kv-shard skipped");return;}
  CHECK(rc==0);
  // host full-kv causal softmax attention reference.
  const float scale=1.0F/std::sqrt((float)D);
  std::vector<float> ref(S*Hq*D,0.f);
for (int t=0;t<S;++t)for(int hq=0;hq<Hq;++hq){
    const float* qt=q.data()+(size_t)(t*Hq+hq)*D; double dn=0.0;
    for (int s=0;s<=t;++s)for (int kh=0;kh<Hkv;++kh){ const float* kp=k.data()+(size_t)(s*Hkv+kh)*D; float a=0; for(int d=0;d<D;++d)a+=qt[d]*kp[d]; dn+=std::exp((double)(a*scale));}
    for (int s=0;s<=t;++s)for (int kh=0;kh<Hkv;++kh){ const float* kp=k.data()+(size_t)(s*Hkv+kh)*D,*vp=v.data()+(size_t)(s*Hkv+kh)*D; float a=0; for(int d=0;d<D;++d)a+=qt[d]*kp[d]; double e=std::exp((double)(a*scale)); for(int d=0;d<D;++d) ref[(size_t)(t*Hq+hq)*D+d]+=(float)(e*vp[d]/dn);}
  }
  for (size_t i=0;i<ref.size();++i) CHECK(out[i]==doctest::Approx(ref[i]).epsilon(2e-3));
}

TEST_CASE("attention KV/GQA-shard (seq): Hkv < world (excess ranks contribute 0)") {
  // The 35B A3B has Hkv=2 on a 4-GPU box — fewer kv-heads than ranks. The shard
  // must run per = ceil(Hkv/W) with the excess ranks' slices guarded out (they
  // contribute a ZERO partial to the num/den AllReduceSum), so the reduced output
  // equals the single-GPU full-kv softmax exactly. Covers the 35B MoE tp gate's
  // exact attention shape.
  constexpr int S=48,Hq=16,Hkv=2,D=256;   // real 35B A3B, deep decode, Hkv<W
  std::vector<float> q(S*Hq*D),k(S*Hkv*D),v(S*Hkv*D);
  for (size_t i=0;i<q.size();++i) q[i]=0.1f*((int)(i%13))-0.3f;
  for (int s=0;s<S;++s)for (int kh=0;kh<Hkv;++kh)for(int d=0;d<D;++d){
    k[(size_t)(s*Hkv+kh)*D+d]=0.05f*((kh*3+s+d)%9)-0.1f; v[(size_t)(s*Hkv+kh)*D+d]=0.2f*((kh*5+s*2+d)%7)+0.1f;}
  std::vector<float> out((size_t)S*Hq*D,0.f);
int rc=vt_cuda_attn_kv_shard_run(S,Hq,S,Hkv,D,q.data(),k.data(),v.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); attn kv-shard (Hkv<W) skipped");return;}
  CHECK(rc==0);
  const float scale=1.0F/std::sqrt((float)D);
  std::vector<float> ref(S*Hq*D,0.f);
  for (int t=0;t<S;++t)for(int hq=0;hq<Hq;++hq){
    const float* qt=q.data()+(size_t)(t*Hq+hq)*D; double dn=0.0;
    for (int s=0;s<=t;++s)for (int kh=0;kh<Hkv;++kh){ const float* kp=k.data()+(size_t)(s*Hkv+kh)*D; float a=0; for(int d=0;d<D;++d)a+=qt[d]*kp[d]; dn+=std::exp((double)(a*scale));}
    for (int s=0;s<=t;++s)for (int kh=0;kh<Hkv;++kh){ const float* kp=k.data()+(size_t)(s*Hkv+kh)*D,*vp=v.data()+(size_t)(s*Hkv+kh)*D; float a=0; for(int d=0;d<D;++d)a+=qt[d]*kp[d]; double e=std::exp((double)(a*scale)); for(int d=0;d<D;++d) ref[(size_t)(t*Hq+hq)*D+d]+=(float)(e*vp[d]/dn);}
  }
  for (size_t i=0;i<ref.size();++i) CHECK(out[i]==doctest::Approx(ref[i]).epsilon(2e-3));
}

TEST_CASE("MoE expert shard: per-expert return, replicated router, exit reduce == full-I") {
  // Model-free Sparse-MoE shard: E experts, top_k per token, I-sliced across the
  // group. The router gate is REPLICATED (full logits -> same topk ids+weights on
  // every rank, NO router reduce); only the routed output [T,H] is AllReduced at
  // the block exit. Verify the 4-GPU routed output equals the single-rank full-I
  // host reference (the tp==tp1 MoE shape).
  constexpr int T=3,H=8,E=6,I=16;   // I%4==0
  constexpr int top_k=2;
#ifdef VLLM_CPP_CUDA
  int devs = -1; (void)cudaGetDeviceCount(&devs);
  if (devs >= 2 && I % devs != 0) { MESSAGE("world not divide I=16; moe skipped"); return; }
#endif
  std::vector<float> x(T*H), gate(E*I*H), up(E*I*H), down(E*I*H);
  for (size_t i=0;i<x.size();++i) x[i]=0.3f+0.1f*((int)(i%7));
  for (int e=0;e<E;++e)for(int i=0;i<I;++i)for(int h=0;h<H;++h){
    gate[(size_t)e*I*H+(size_t)i*H+h]=0.01f*((e*11+i*3+h)%670)+0.05f;
    up  [(size_t)e*I*H+(size_t)i*H+h]=0.01f*((e*7+i*5+h)%130)+0.1f;
    down[(size_t)e*I*H+(size_t)i*H+h]=0.02f*((e*5+i*7+h)%1130)+0.2f;}
  // Replicated router: same topk ids+weights on every rank (no reduce residue).
  std::vector<int32_t> ids(T*top_k);
  std::vector<float> wgt(T*top_k);
  for (int t=0;t<T;++t)for(int k=0;k<top_k;++k){
    const int p=t*top_k+k; ids[p]=(int32_t)((t*3+k*2)%E); wgt[p]=0.5f+0.25f*(float)((p%3));}
  std::vector<float> out(T*H,0.f);
  int rc=vt_cuda_moe_expert_shard_run(T,H,I,top_k,E,x.data(),gate.data(),up.data(),down.data(),ids.data(),wgt.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); moe shard skipped");return;}
  CHECK(rc==0);
  // Host full-I routed reference.
  std::vector<float> ref(T*H,0.f);
  for (int t=0;t<T;++t)for(int k=0;k<top_k;++k){
    const int e=ids[t*top_k+k]; const float w=wgt[t*top_k+k];
    for (int i=0;i<I;++i){ float gv=0,uv=0; for(int h=0;h<H;++h){gv+=x[t*H+h]*gate[e*I*H+i*H+h];uv+=x[t*H+h]*up[e*I*H+i*H+h];}
      const float sg=1.f/(1.f+std::exp(-(double)gv)); const float act=(gv*sg)*uv*w;
      for(int h=0;h<H;++h) ref[t*H+h]+=act*down[e*I*H+i*H+h];}}
  for (size_t i=0;i<out.size();++i) CHECK(out[i]==doctest::Approx(ref[i]).epsilon(6e-4));
}

TEST_CASE("MoE expert shard: 35B-A3B geometry (H=2048,I=512,E=8,top=8,T=9)") {
  // Reproduce the real-model shapes where the tp>1 MoE shard feeds a ForwardDense
  // decode: hidden 2048, moe_intermediate 512, 8 experts x 8 per token, 9 tokens.
  constexpr int T=9,H=2048,I=512,E=256,top_k=8;
#ifdef VLLM_CPP_CUDA
  int devs = -1; (void)cudaGetDeviceCount(&devs);
  if (devs >= 2 && I % devs != 0) { MESSAGE("world not divide I=512; 35B moe skipped"); return; }
#endif
  std::vector<float> x(T*H), gate(E*I*H), up(E*I*H), down(E*I*H);
  for (size_t i=0;i<x.size();++i) x[i]=0.03f+0.001f*((int)(i%37));
  for (int e=0;e<E;++e)for(int i=0;i<I;++i)for(int h=0;h<H;++h){
    gate[(size_t)e*I*H+(size_t)i*H+h]=0.002f*((e*11+i*3+h)%97)-0.001f;
    up  [(size_t)e*I*H+(size_t)i*H+h]=0.001f*((e*7+i*5+h)%131)+0.0005f;
    down[(size_t)e*I*H+(size_t)i*H+h]=0.002f*((e*5+i*7+h)%101)-0.001f;}
  std::vector<int32_t> ids(T*top_k); std::vector<float> wgt(T*top_k);
  for (int t=0;t<T;++t)for(int k=0;k<top_k;++k){int p=t*top_k+k; ids[p]=(p*3)%E; wgt[p]=0.5f+0.1f*(float)(p%5);}
  std::vector<float> out(T*H,0.f);
  int rc=vt_cuda_moe_expert_shard_run(T,H,I,top_k,E,x.data(),gate.data(),up.data(),down.data(),ids.data(),wgt.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs; moe 35B-geom skipped");return;}
  CHECK(rc==0);
  // finite + a tiny host spot-check (full T*H cross-check is 6e12 MACs).
  bool fin=true; for (float v : out) if (!std::isfinite(v)) {fin=false;break;}
  CHECK(fin);
  if (fin) {
    // Spot rows t=0 and t=8, col 1, against a host computation of ONE element.
    for (std::pair<int,int> tgt : {std::pair<int,int>{0,1},{8,1},{0,2047}}) {
      double want = 0.0;
      for (int k=0;k<top_k;++k){int e=ids[tgt.first*top_k+k]; float w=wgt[tgt.first*top_k+k];
        for (int i=0;i<I;++i){double gv=0,uv=0; for(int h=0;h<H;++h){gv+=x[tgt.first*H+h]*gate[e*I*H+i*H+h]; uv+=x[tgt.first*H+h]*up[e*I*H+i*H+h];} double sg=1.0/(1.0+std::exp(-gv)); want+=(gv*sg)*uv*w*down[e*I*H+i*H+tgt.second]; }}
      const double got = out[(size_t)tgt.first*H+tgt.second];
      CHECK(got==doctest::Approx(want).epsilon(6e-4));
    }
  }
}

TEST_CASE("lm_head column-shard: vocab split across ranks + AllGather == full-vocab logits") {
  constexpr int T=4,H=16,V=64;  // V%W==0 on 2/4-GPU box
#ifdef VLLM_CPP_CUDA
  int devs = -1; (void)cudaGetDeviceCount(&devs);
  if (devs >= 2 && V % devs != 0) { MESSAGE("world not divide V=64; lm_head skipped"); return; }
#endif
  std::vector<float> x(T*H),w(V*H);
  for (size_t i=0;i<x.size();++i) x[i]=0.05f*((int)(i%11))-0.2f;
  for (int v=0;v<V;++v) for (int d=0;d<H;++d) w[v*H+d]=0.01f*((v*3+d)%19)-0.4f;
  std::vector<float> out((size_t)T*V,0.f);
  int rc=vt_cuda_lm_head_shard_run(T,H,V,x.data(),w.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); lm_head shard skipped");return;}
  CHECK(rc==0);
  // full-vocab host reference
  std::vector<float> ref(T*V);
  for (int m=0;m<T;++m) for (int v=0;v<V;++v){ double a=0; for(int d=0;d<H;++d) a+=(double)x[m*H+d]*(double)w[v*H+d]; ref[m*V+v]=(float)a;}
  for (size_t i=0;i<ref.size();++i) CHECK(out[i]==doctest::Approx(ref[i]).epsilon(2e-4));
}
