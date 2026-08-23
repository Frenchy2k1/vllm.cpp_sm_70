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
extern "C" int vt_cuda_attn_kv_shard_run(int T,int Hq,int Hkv,int D,const float* q,const float* k,const float* v,float* out);
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

TEST_CASE("attention KV/GQA-shard: KV split across ranks == single-GPU softmax attention") {
  constexpr int T=3,Hq=4,Hkv=8,D=16;  // Hkv%W==0 on a 2/4-GPU box
  std::vector<float> q(T*Hq*D),k(Hkv*D),v(Hkv*D);
  for (size_t i=0;i<q.size();++i) q[i]=0.1f*((int)(i%13))-0.3f;
  for (int kh=0;kh<Hkv;++kh) for (int d=0;d<D;++d){ k[kh*D+d]=0.05f*((kh*3+d)%9)-0.1f; v[kh*D+d]=0.2f*((kh*5+d)%7)+0.1f;}
  std::vector<float> out((size_t)T*Hq*D,0.f);
  int rc=vt_cuda_attn_kv_shard_run(T,Hq,Hkv,D,q.data(),k.data(),v.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL missing); attn kv-shard skipped");return;}
  CHECK(rc==0);
  // host full-kv softmax attention reference.
  const float scale=1.0F/std::sqrt((float)D);
  std::vector<float> ref(T*Hq*D,0.f),dref(T*Hq,0.f);
  for (int t=0;t<T;++t)for(int hq=0;hq<Hq;++hq){
    const float* qt=q.data()+(size_t)(t*Hq+hq)*D; double dn=0.0;
    for (int kh=0;kh<Hkv;++kh){ const float* kp=k.data()+(size_t)kh*D; float s=0; for(int d=0;d<D;++d)s+=qt[d]*kp[d]; dn+=std::exp((double)(s*scale));}
    for (int kh=0;kh<Hkv;++kh){ const float* kp=k.data()+(size_t)kh*D,*vp=v.data()+(size_t)kh*D; float s=0; for(int d=0;d<D;++d)s+=qt[d]*kp[d]; double e=std::exp((double)(s*scale)); for(int d=0;d<D;++d) ref[(size_t)(t*Hq+hq)*D+d]+=(float)(e*vp[d]/dn);}
  }
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
