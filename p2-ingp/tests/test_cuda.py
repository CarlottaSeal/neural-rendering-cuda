"""CUDA hash encoding vs the PyTorch reference: forward/backward agreement and timing at the training config.

  scripts\\p2_run.bat p2-ingp\\tests\\test_cuda.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ingp.encoding import HashEncoding  # noqa: E402


def bench(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    torch.manual_seed(0)
    N = 1 << 18  # samples per step in training
    ref = HashEncoding(init_scale=1.0).cuda()  # default 16 levels, F=2, T=2^19
    cu = HashEncoding(init_scale=1.0, backend="cuda").cuda()
    cu.embeddings.data.copy_(ref.embeddings.data)
    x = torch.rand(N, 3, device="cuda")
    g = torch.randn(N, ref.out_dim, device="cuda")

    y_ref, y_cu = ref(x), cu(x)
    fwd_err = (y_ref - y_cu).abs().max().item()
    y_ref.backward(g)
    y_cu.backward(g)
    gr, gc = ref.embeddings.grad, cu.embeddings.grad
    bwd_err = ((gr - gc).abs().max() / gr.abs().max()).item()
    print(f"forward max abs err {fwd_err:.2e}; backward max err / max |grad| {bwd_err:.2e}")
    assert fwd_err < 1e-5 and bwd_err < 1e-5

    for name, enc in (("torch", ref), ("cuda v0", cu)):
        f = bench(lambda: enc(x))
        fb = bench(lambda: enc(x).backward(g))
        print(f"{name:8s} N={N}: fwd {f:7.3f} ms   fwd+bwd {fb:7.3f} ms")


if __name__ == "__main__":
    main()
