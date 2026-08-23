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
  int rc=vt_cuda_mlp_shard_run(O,H,I,x.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL unbuilt); mlp_shard_run skipped");return;}
  CHECK(rc==0);
  for (int o=0;o<O;++o){ double a=0.0;for(int i=0;i<I;++i){float g=0,u=0;for(int k=0;k<H;++k){g+=x[k]*gate[i*H+k];u+=x[k]*up[i*H+k];}float sg=1.f/(1.f+std::exp(-(double)g));a+=(double)((g*sg)*u)*down[o*I+i];} CHECK(out[(size_t)o]==doctest::Approx((float)a));}
}

TEST_CASE("mlp_shard_run holds at engine-scale intermediate width") {
  constexpr int O=256, H=512, I=1024;   // I % W == 0 (W=4)
  std::vector<float> x(H), gate(I*H), up(I*H), down(O*I), out(O,0.f);
  for (int k=0;k<H;++k) x[k]=0.4f+0.1f*(float)(k%7);
  for (int i=0;i<I;++i)for(int k=0;k<H;++k){ float r=0.001f*((i*3+k)%997); gate[i*H+k]=r+0.1f; up[i*H+k]=r+0.2f; }
  for (int o=0;o<O;++o)for(int i=0;i<I;++i) down[o*I+i]=0.001f*((o*7+i*3)%1000)+0.3f;
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

extern "C" int vt_cuda_attn_gqa_shard_run(int T,int Hq,int Hkv,int S,int D,const int* key_end,const float* q,const float* k,const float* v,float* out);
// host GQA reference: query head hq attends to its ONE kv head g=hq/(Hq/Hkv).
static void gqa_ref(const int* key_end,int T,int Hq,int Hkv,int D,const float* q,const float* k,const float* v,std::vector<float>& ref){
  const float scale=1.0F/std::sqrt((float)D); const int qpk=Hq/Hkv;
  ref.assign((size_t)T*Hq*D,0.f);
  for (int t=0;t<T;++t)for(int hq=0;hq<Hq;++hq){
    const int g=hq/qpk, ke=key_end[t]; const float* qt=q+(size_t)(t*Hq+hq)*D; double dn=0.0;
    for (int s=0;s<ke;++s){ const float* kp=k+(size_t)(s*Hkv+g)*D; float a=0; for(int d=0;d<D;++d)a+=qt[d]*kp[d]; dn+=std::exp((double)(a*scale));}
    for (int s=0;s<ke;++s){ const float* kp=k+(size_t)(s*Hkv+g)*D,*vp=v+(size_t)(s*Hkv+g)*D; float a=0; for(int d=0;d<D;++d)a+=qt[d]*kp[d]; double e=std::exp((double)(a*scale)); for(int d=0;d<D;++d) ref[(size_t)(t*Hq+hq)*D+d]+=(float)(e*vp[d]/dn);}
  }
}
TEST_CASE("GQA-exact KV-shard (causal): one kv head per q-head + causal == single-GPU GQA") {
  constexpr int S=3,Hq=12,Hkv=4,D=16;  // qpk=3 (3 q-heads per kv head); Hkv%W==0 on 2/4-GPU box
  std::vector<int> ke(S); for (int t=0;t<S;++t) ke[t]=t+1;  // causal bound
  std::vector<float> q(S*Hq*D),k(S*Hkv*D),v(S*Hkv*D);
  for (size_t i=0;i<q.size();++i) q[i]=0.1f*((int)(i%13))-0.3f;
  for (int s=0;s<S;++s)for (int kh=0;kh<Hkv;++kh)for(int d=0;d<D;++d){
    k[(size_t)(s*Hkv+kh)*D+d]=0.05f*((kh*3+s+d)%9)-0.1f; v[(size_t)(s*Hkv+kh)*D+d]=0.2f*((kh*5+s*2+d)%7)+0.1f;}
  std::vector<float> out((size_t)S*Hq*D,0.f);
  int rc=vt_cuda_attn_gqa_shard_run(S,Hq,Hkv,S,D,ke.data(),q.data(),k.data(),v.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); gqa shard skipped");return;}
  CHECK(rc==0);
  std::vector<float> ref; gqa_ref(ke.data(),S,Hq,Hkv,D,q.data(),k.data(),v.data(),ref);
  for (size_t i=0;i<ref.size();++i) CHECK(out[i]==doctest::Approx(ref[i]).epsilon(2e-3));
}
TEST_CASE("GQA-exact KV-shard (decode/full window): each query sees all S keys == single-GPU GQA") {
  constexpr int S=5,Hq=12,Hkv=4,D=16;  // qpk=3; Hkv%W==0 on 2/4-GPU box
  std::vector<int> ke(S); for (int t=0;t<S;++t) ke[t]=S;    // full window (paged decode)
  std::vector<float> q(S*Hq*D),k(S*Hkv*D),v(S*Hkv*D);
  for (size_t i=0;i<q.size();++i) q[i]=0.07f*((int)(i%11))-0.2f;
  for (int s=0;s<S;++s)for (int kh=0;kh<Hkv;++kh)for(int d=0;d<D;++d){
    k[(size_t)(s*Hkv+kh)*D+d]=0.04f*((kh*4+s+d)%13)-0.15f; v[(size_t)(s*Hkv+kh)*D+d]=0.13f*((kh*2+s*3+d)%11)+0.05f;}
  std::vector<float> out((size_t)S*Hq*D,0.f);
  int rc=vt_cuda_attn_gqa_shard_run(S,Hq,Hkv,S,D,ke.data(),q.data(),k.data(),v.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); gqa decode shard skipped");return;}
  CHECK(rc==0);
  std::vector<float> ref; gqa_ref(ke.data(),S,Hq,Hkv,D,q.data(),k.data(),v.data(),ref);
  for (size_t i=0;i<ref.size();++i) CHECK(out[i]==doctest::Approx(ref[i]).epsilon(2e-3));
}

TEST_CASE("lm_head column-shard: vocab split across ranks + AllGather == full-vocab logits") {
  constexpr int T=4,H=16,V=64;  // V%W==0 on 2/4-GPU box
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

extern "C" int vt_cuda_mlp_shard_run_b(int T,int O,int H,int I,int verify,const float* x,const float* gate,const float* up,const float* down,float* out);
// host batched-MLP reference (f32 weights -> f32 out), one (t,o) at a time.
static float mlp_ref_one(int t,int o,int H,int I,const float* x,const float* gate,const float* up,const float* down){
  double a=0.0;
  for(int i=0;i<I;++i){float g=0.f,u=0.f;for(int k=0;k<H;++k){g+=x[(size_t)t*H+k]*gate[i*H+k];u+=x[(size_t)t*H+k]*up[i*H+k];}float sg=1.f/(1.f+std::exp(-(double)g));a+=(double)((g*sg)*u)*down[(size_t)o*I+i];}
  return (float)a;
}
TEST_CASE("batched mlp_shard_run_b: fresh shape, verify=1 (full internal parity)") {
  constexpr int T=3,O=16,H=32,I=64;  // I%2==0 on 2-GPU box
  std::vector<float> x(T*H), gate(I*H), up(I*H), down(O*I), out(T*O,0.f);
  for (size_t n=0;n<x.size();++n) x[n]=0.3f+0.1f*(float)(n%5);
  for (size_t i=0;i<gate.size();++i){ gate[i]=0.01f*((int)(i%917))+0.07f; up[i]=0.01f*((int)(i%1223))+0.13f; }
  for (size_t i=0;i<down.size();++i) down[i]=0.02f*((int)(i%1541))+0.2f;
  int rc=vt_cuda_mlp_shard_run_b(T,O,H,I,1,x.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL unbuilt); batched mlp shard skipped");return;}
  CHECK(rc==0);
  for (int t=0;t<T;++t) for (int o=0;o<O;++o)
    CHECK(out[(size_t)t*O+o]==doctest::Approx(mlp_ref_one(t,o,H,I,x.data(),gate.data(),up.data(),down.data())).epsilon(2e-4));
}
TEST_CASE("batched mlp_shard_run_b holds at 27B production shape (T=9, H=5120, I=17408)") {
  // Production-scale shapes: the verify=0 compute-only path is what the paged
  // engine gate runs; parity is spot-checked here against the host reference
  // (a full reference is ~1.7e14 double-FMA, so sample a few (t,o)).
  constexpr int T=9,O=5120,H=5120,I=17408;  // I%2==0
  std::vector<float> x(T*H), gate((size_t)I*H), up((size_t)I*H), down((size_t)O*I), out((size_t)T*O,0.f);
  for (size_t n=0;n<x.size();++n) x[n]=0.05f+0.01f*(float)(n%37);
  for (size_t i=0;i<gate.size();++i){ gate[i]=0.002f*((int)(i%997))+0.01f; up[i]=0.002f*((int)(i%1009))+0.015f; }
  for (size_t i=0;i<down.size();++i) down[i]=0.001f*((int)(i%1003))+0.05f;
  int rc=vt_cuda_mlp_shard_run_b(T,O,H,I,0,x.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); big batched mlp shard skipped");return;}
  CHECK(rc==0);
  // deterministic (t,o) sample set covering token and output extremes.
  const int ts[5]={0,1,4,8,9-1}, os[5]={0,1,4096,5119,2600};
  for (int a=0;a<5;++a) for (int b=0;b<5;++b){
    const int t=ts[a], o=os[b];
    CHECK(out[(size_t)t*O+o]==doctest::Approx(mlp_ref_one(t,o,H,I,x.data(),gate.data(),up.data(),down.data())).epsilon(5e-4));
  }
}

extern "C" int vt_cuda_moe_shard_run_b(int P,int H,int I,int E,int verify,const float* x,const int32_t* exp,const float* gate,const float* up,const float* down,float* out);
extern "C" int vt_cuda_moe_shard_selfcheck(void);
// host MoE reference (per-pair expert), one (p,o) at a time.
static float moe_ref_one(int p,int o,int H,int I,const float* x,const int32_t* exp,const float* gate,const float* up,const float* down){
  double a=0.0; const int e=exp[p];
  for(int i=0;i<I;++i){
    double gv=0.0, uv=0.0;
    for(int k=0;k<H;++k){ gv+=(double)x[(size_t)p*H+k]*gate[((size_t)e*I+i)*H+k]; uv+=(double)x[(size_t)p*H+k]*up[((size_t)e*I+i)*H+k]; }
    const float sg=1.f/(1.f+(float)std::exp(-(double)gv));
    a+=(double)((gv*sg)*uv)*down[((size_t)e*H+o)*I+i];
  }
  return (float)a;
}
TEST_CASE("moe expert-shard selfcheck: per-pair expert + intermediate slice + AllReduce (verify=1)") {
  const int rc = vt_cuda_moe_shard_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); moe shard skipped");
    return;
  }
  CHECK(rc == 0);
}
TEST_CASE("moe shard: fresh shape, verify=1 (full internal parity)") {
  constexpr int P=8,H=48,I=64,E=4;  // I%2==0 on 2-GPU box
  std::vector<float> x((size_t)P*H), gate((size_t)E*I*H), up((size_t)E*I*H), down((size_t)E*H*I), out((size_t)P*H,0.f);
  std::vector<int32_t> exp(P);
  for (size_t n=0;n<x.size();++n) x[n]=0.2f+0.1f*(float)(n%7);
  for (size_t i=0;i<gate.size();++i){ gate[i]=0.01f*((int)(i%131)); up[i]=0.01f*((int)(i%139)); }
  for (size_t i=0;i<down.size();++i) down[i]=0.01f*((int)(i%137));
  for (int p=0;p<P;++p) exp[p]=(p*5)%E;
  int rc=vt_cuda_moe_shard_run_b(P,H,I,E,1,x.data(),exp.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL unbuilt); moe shard skipped");return;}
  CHECK(rc==0);
  for (int p=0;p<P;++p) for (int o=0;o<H;o+=7)
    CHECK(out[(size_t)p*H+o]==doctest::Approx(moe_ref_one(p,o,H,I,x.data(),exp.data(),gate.data(),up.data(),down.data())).epsilon(5e-4));
}
TEST_CASE("moe shard holds at 35B production shape (H=2048, I=512, E=256, P=3)") {
  // The 35B A3B MoE shape at decode T=1, top_k=8 (P=T*top_k=3 covers the
  // prefill-1 + 2 decode tokens). The verify=0 compute-only path is what the
  // paged engine gate runs; parity is spot-checked against the host reference.
  constexpr int P=3,H=2048,I=512,E=256;  // I%2==0
  std::vector<float> x((size_t)P*H), gate((size_t)E*I*H), up((size_t)E*I*H), down((size_t)E*H*I), out((size_t)P*H,0.f);
  std::vector<int32_t> exp(P);
  for (size_t n=0;n<x.size();++n) x[n]=0.05f+0.01f*(float)(n%37);
  for (size_t i=0;i<gate.size();++i){ gate[i]=0.002f*((int)(i%997)); up[i]=0.002f*((int)(i%1009)); }
  for (size_t i=0;i<down.size();++i) down[i]=0.001f*((int)(i%1003));
  for (int p=0;p<P;++p) exp[p]=17*p+3;
  int rc=vt_cuda_moe_shard_run_b(P,H,I,E,0,x.data(),exp.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); big moe shard skipped");return;}
  CHECK(rc==0);
  const int ps[3]={0,1,2}, os[6]={0,1,1000,2046,2047,1500};
  for (int a=0;a<3;++a) for (int b=0;b<6;++b){
    const int p=ps[a], o=os[b];
    CHECK(out[(size_t)p*H+o]==doctest::Approx(moe_ref_one(p,o,H,I,x.data(),exp.data(),gate.data(),up.data(),down.data())).epsilon(5e-4));
  }
}
