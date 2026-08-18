# Parity ledger — sm70-support (Milestone-0 pin)

Rows record a measured oracle-vs-impl comparison and its verdict. A row is
"live" until a later run supersedes it. Update in place, don't append noise.

## 1. 35B engine token-exact · GREEN
- impl: `test_qwen36_paged_engine` (sm_70 build, arch 70) on V100
- model: `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `491c2f1ea524c639598bf8fa787a93fed5a6fbce`
- oracle: repo pinned golden (greedy_ids token-for-token)
- result: **315/315, Status SUCCESS** (2026-08-12, V100)
- verdict: same-model correct; marks the decoder as the load-bearing impl reference.

## Row | 27B dense forward · prefill-exact / tok6 near-tie

| impl | `test_op_parity -tc="qwen27 dense logits*"` (sm_70) |
| model | `unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7a42feba6d83b6e17a03315c694112f2a` |
| result | prefill `argmax_match = 9/9`; greedy `6/16` (divergence @ tok6); top-1000 logit gap `1.296` (non-boundary) |
| analysis | tok6 near-tie: golden native `198`, emulation `271`. Emulation side is produced because the sm_70 build lacks `VT_CUTLASS_NVFP4` (sm_90+) — that guard is intentional (engine gate throws without it). sm_70 hand W4A4 kernel output is the legitimate answer on this silicon. |
| verdict | acceptable at semantic/near-tie; native side obtain 1 only on an sm_90+ production build. NOT to be closed by a special-case. |

## sm70 fp16-WMMA expert GEMM (brick H) · GREEN

| impl | `MoeGroupedGemmNvfp4WmmaF16` (fp16 mma.sync m16n16k16, fp32 acc) + decode launcher, `VT_SM70_MOE_WMMA` (default ON) |
| route | A3B-35B expert decode previously ran the CUDA-core naive fill (Marlin sm_75+, bf16-WMMA sm_80+; Volta has no bf16 MMA). Brick makes experts ride Volta fp16 tensor cores; bf16->fp16 saturates at ±65504 (fp16 5-bit exponent, prevents inf near-tie poison) |
| acceptance | `test_qwen36_paged_engine` (35B-A3B-NVFP4, real activations, incl 6 concurrent x 16 tokens) with brick default ON: **315/315 token-exact, Status SUCCESS** (2026-08-12, V100) |
| A/B | `VT_SM70_MOE_WMMA=0` returns the naive fill (the pre-brick baseline) |

## sm70 wider-head paged attention (superset) · GREEN

| commit | `59c0d05a` |
| change | decode/prefill 192|256 instantiations + dispatch; gate widened to {64,128,192,256} |
| acceptance | `test_sm70_fa2_decode` on V100: decode vs reference reldev 2.00e-5 (D=192), 1.74e-5 (D=256); op-parity 7.99e-6; prefill parity OK; suite 4/4 (tol 3e-2) |

## sm70 y-dg-sync coverage (CUDA-graph capture safety) · GREEN

| commit | `cd5b2782` |
| scope | audit + regression gate: the runner replays steady-state decode inside a captured CUDA graph; any host-sync/malloc the decode emits on the capture stream trips `cudaErrorStreamCaptureImplicit` and dies. Audit: all existing host syncs are warmup-only. Gate: `vt_sm70_fa2_graph_replay_parity()` runs Begin->EndCapture->instantiate->launch and REQUIRES replay==eager |
| acceptance | V100: graph-replay max relative dev **0.000e+00** (bit-identical); `test_sm70_fa2_decode` 5/5 SUCCESS |

## Multi-device phase 2 — per-device CUDA backend affinity · GREEN

| commit | `e14374dc` |
| change | `CudaDeviceScope` (RAII; no-op when device==current → single-GPU byte-identical) applied to every device-touching CudaBackend method; a `Device{kCUDA,i}` backend now allocates/creates streams/launches on GPU i |
| acceptance | `test_cuda_backend`: GPU {0,1,2,3} → alloc device {0,1,2,3} (cudaPointerGetAttributes), Memset/Copy on each queue round-trips; 7/7 cases, 58 assertions, SUCCESS (4×V100 sm_70) |

## Multi-device phase 3 — in-process NCCL collectives primitive · GREEN

| commit | `b4901998` |
| change | real `ncclCommInitAll` over all local GPUs (single process, 4×V100) → one NcclCommunicator per device; the transport (providers) is now actually rnew durable. Per-rank collectives driven from a host thread each (NCCL ours blocking) + per-device affinity |
| acceptance | `test_nccl_group` (VLLM_CPP_NCCL-gated): AllReduce(kSum)→10 on every rank, AllGather→ordered [1000..1003]; 1/1 SUCCESS on the 4×V100 |

## Multi-device phase 4 — TP loader slice · GREEN

| commit | `fc58ae0b` |
| change | `vt_cuda_loader_slice_selfcheck`: dense [K,N] weight sliced by `vllm::TpShard` per rank, each shard memcpied onto ITS device (per-op affinity), the W slices reconstructing the full weight — the placement side a tp>1 weight loader uses |
| acceptance | 4×V100: group selfcheck + TP seam + loader slice all pass (3/3) |

## Multi-device phase 5 — runner-forward primitive · GREEN

| commit | `8d303584` |
| change | `vt_cuda_sharded_forward_selfcheck`: y=Wx sharded (each rank a K-slice of columns on ITS device), per-rank real device GEMM, NCCL group all-reduce → full y == unsharded single-GPU ref. Fixed local-vs-global x indexing |
| acceptance | 4×V100: collectives + TP-seam + loader-slice + runner-forward all pass (4/4) |

## Multi-device phase 6 — DenseMtpBlock tp>1 per-rank shard · GREEN

| commit | `66f48f80` |
| change | `DenseMtpBlock` tp>1 branch replaces the throw: host-decode the resident fp4 (bf16-raw/split) gate/up/down into the shard's `[out,in]` f32 layout, run the group-sharded dense-MLP per token over the in-process NCCL group, write the reduced `[T,H]` bf16 back. tp1 (null) never enters → resident tensor-core path byte-identical |
| acceptance | box sm_70: `test_nccl_group` 10/10, `test_sm70_fa2_decode` 5/5, `test_cuda_backend` 7/7 (58), `test_qwen36_paged_engine` 35B case token-equality 149/149. The 27B/35B paged-engine guards (`qwen27/36`) require `VLLM_CPP_TRITON=ON` + cutlass build — this box build has TRITON=OFF, so those throw as build guards (unchanged by this edit), not as a regression |
| next | real multi-GPU token-equal gate still requires the attention row/col shard step (new sharded-attention primitive) → head → runner attach |

## Multi-device phase 7 — KV/GQA-shard attention primitive · GREEN

| commit | `c643c311` |
| change | `vt_cuda_attn_kv_shard_run` + kernel `AttnKvShardPartial`: softmax num/den are both additive over the kv dimension, so a rank owning a kv-head slice computes partial exp(v) and exp-sum on ITS device; group AllReduceSum reduces both, out = num/den. Full [T,Hq,D] query on every rank. Returns 0 on full-kv host reference parity |
| acceptance | box sm_70: `test_nccl_group` 11/11 incl. model-free KV-split test (T=3,Hq=4,Hkv=8) == single-GPU softmax attention |

## Multi-device phase 8 — lm_head column-shard + AllGather primitive · GREEN

| commit | `87e78ce` |
| change | `vt_cuda_lm_head_shard_run` + `LmHeadShardPartial`: vocab-column-parallel head, each rank computes [T,per] partial logits on ITS device, group AllGather concatenates the disjoint-vocab slices into full [T,V] (rank-major → token-major reassembly). Returns 0 vs full-vocab host reference |
| acceptance | `test_nccl_group` 12/12, 479 assertions on the box |

## 27B-AWQ oracle capture · deterministic

| lane | 1Cat-vLLM 1.2.2 on V100 (`192.168.10.20:8000`, model `qwen3.6-27b-awq-mtp`) |
| battery | 8 prompts × 3 runs, greedy T=0, `enable_thinking:false`, max_tokens 32 |
| artifact | `tests/parity/goldens/qwen36_27b_awq_oracle/{trace,meta,prompt_ids}.json*` (commit `39d81e89`) |
| determinism | 24/24 rows identical across runs (no near-ties in battery) |
| caveat | **quant-route**: AWQ(INT4) vs our NVFP4 fast paths — grade semantically (greedy token-string + per-position argmax), never bit-exact. |
| comparator | `.agents/compare-oracle-impl.py` — smoke: oracle-vs-itself = 8 exact / 0 diverged |

## Box fact (evidence)

```
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
Tesla V100-DGXS-32GB, 7.0, 32768 MiB   (x4)
```
sm70 premise: CONFIRMED indirectly (`h100` hostname is a red herring).