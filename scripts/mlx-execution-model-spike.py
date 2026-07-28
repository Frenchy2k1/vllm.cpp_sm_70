# Does MLX's LAZY-GRAPH execution model explain mlx-lm's edge?
#
# Same weight-streaming work as one Qwen3-1.7B decode step, run two ways:
#   eager : mx.eval() after every op  -- what vllm.cpp's MLX *provider* does
#   lazy  : one mx.eval() per step    -- what mlx-lm does
# If lazy >> eager, the execution model is the lever and an MLX-graph Metal path
# is worth building. If lazy ~= our native 27.24 tok/s, it buys nothing.
import time, mlx.core as mx

H, I, KV, L, VOCAB = 2048, 6144, 1024, 28, 151936
DT = mx.bfloat16

def mk(*shape): return mx.random.normal(shape).astype(DT)
W = [{"qkv": mk(H, 2048+2*KV), "o": mk(H, H), "gu": mk(H, 2*I), "dn": mk(I, H)} for _ in range(L)]
lm = mk(H, VOCAB)
mx.eval([w[k] for w in W for k in w] + [lm])

def step(x, eager):
    for w in W:
        qkv = x @ w["qkv"]
        if eager: mx.eval(qkv)
        a = qkv[:, :H]
        o = a @ w["o"]
        if eager: mx.eval(o)
        x = x + o
        gu = x @ w["gu"]
        if eager: mx.eval(gu)
        g, u = gu[:, :I], gu[:, I:]
        h = (g * mx.sigmoid(g)) * u
        if eager: mx.eval(h)
        d = h @ w["dn"]
        if eager: mx.eval(d)
        x = x + d
    lg = x @ lm
    mx.eval(lg)
    return x

for mode in ("eager", "lazy"):
    x = mk(1, H)
    for _ in range(3): step(x, mode == "eager")     # warm
    mx.synchronize()
    N = 40
    t0 = time.perf_counter()
    for _ in range(N): step(x, mode == "eager")
    mx.synchronize()
    dt = (time.perf_counter() - t0) / N
    print(f"{mode:6s}: {dt*1000:6.2f} ms/step -> {1/dt:6.2f} tok/s")
