# vLLM-DFlash timing on the SAME 8-prompt x 256-token c1 workload as vllm-bench.
# Mirrors the PROVEN d0 capture config (mm-off, gpu_util 0.30, VLLM_USE_V2_MODEL_RUNNER=1).
# --use-cudagraph => enforce_eager=False (vLLM PRODUCTION graphed config, the honest bar).
import os, sys, json, time, argparse
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH","")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER","1")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION","1")

def metric_val(metrics, name):
    for m in metrics:
        if getattr(m,'name',None)==name:
            return getattr(m,'value',None) if hasattr(m,'value') else sum(getattr(m,'values',[]) or [0])
    return None

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["spec-on","spec-off"],required=True)
    ap.add_argument("--target",required=True); ap.add_argument("--draft",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--max-tokens",type=int,default=256)
    ap.add_argument("--k",type=int,default=16)
    ap.add_argument("--gpu-mem-util",type=float,default=0.30)
    ap.add_argument("--max-model-len",type=int,default=4096)
    ap.add_argument("--use-cudagraph",action="store_true")
    a=ap.parse_args()
    from vllm import LLM, SamplingParams
    prompts=[e["conversations"][0]["value"] for e in json.load(open(a.dataset))]
    kw=dict(model=a.target, enforce_eager=not a.use_cudagraph,
            gpu_memory_utilization=a.gpu_mem_util, max_model_len=a.max_model_len,
            max_num_seqs=4, max_num_batched_tokens=4096,
            limit_mm_per_prompt={"image":0,"video":0}, disable_log_stats=False)
    if a.mode=="spec-on":
        kw["speculative_config"]={"method":"dflash","model":a.draft,
            "num_speculative_tokens":a.k,"max_model_len":a.max_model_len}
    llm=LLM(**kw)
    sp=SamplingParams(temperature=0.0,top_p=1.0,max_tokens=a.max_tokens,ignore_eos=True)
    # warmup (1 prompt, excluded)
    _=llm.generate([prompts[0]], sp)
    # timed: each prompt sequentially (c1), sum output tokens / wall time
    t0=time.perf_counter(); tot=0; per=[]
    for p in prompts:
        s=time.perf_counter(); o=llm.generate([p], sp)[0]; e=time.perf_counter()
        n=len(o.outputs[0].token_ids); tot+=n; per.append((n,e-s))
    el=time.perf_counter()-t0
    acc=None
    try:
        m=llm.get_metrics()
        nd=metric_val(m,"vllm:spec_decode_num_drafts")
        na=metric_val(m,"vllm:spec_decode_num_accepted_tokens")
        if nd: acc=(na or 0)/nd + 1.0
    except Exception as ex:
        acc=f"NA:{ex}"
    tpot_ms=1000.0*el/max(tot,1)
    otput=tot/el
    print("VLLM_TIMING="+json.dumps({"mode":a.mode,"graphed":a.use_cudagraph,
        "prompts":len(prompts),"total_out_tokens":tot,"elapsed_s":round(el,3),
        "tpot_ms":round(tpot_ms,3),"output_tput_tok_s":round(otput,3),
        "acceptance_len":acc,"per_prompt":[(n,round(d,3)) for n,d in per]}))
