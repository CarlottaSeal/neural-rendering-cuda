"""CUDA hash encoding (v0, v1) vs the PyTorch reference: forward/backward agreement and timing.

Timing runs on random points and, if the Lego checkpoint exists, on the points the march produces in a
training step (1.1M, spatially coherent along rays), which is what the prune pass encodes.

  scripts\\p2_run.bat p2-ingp\\tests\\test_cuda.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ingp.encoding import HashEncoding  # noqa: E402

VERSIONS = (0, 1)


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


def lego_points(ckpt="outputs/p2/ref_lego/field.pt", data="data/nerf_synthetic/lego"):
    if not os.path.exists(ckpt):
        return None
    from ingp.data import BlenderScene
    from ingp.model import NGPField
    from ingp.render import OccupancyGrid, march
    aabb = torch.tensor([-1.5] * 3 + [1.5] * 3, device="cuda")
    dt = math.sqrt(3) * 3.0 / 1024
    field = NGPField(aabb).cuda()
    field.load_state_dict(torch.load(ckpt, map_location="cuda"))
    grid = OccupancyGrid(aabb)
    with torch.no_grad():
        for _ in range(30):
            grid.update(lambda p: field.density(p), dt)
    torch.manual_seed(0)
    o, d, _ = BlenderScene(data, "train", "cuda").sample_batch(48872)
    r, t = march(o, d, aabb, grid, dt, True, backend="cuda")
    return field.normalize(o[r] + t[:, None] * d[r]).contiguous(), field.encoding.embeddings.data.clone()


def main():
    torch.manual_seed(0)
    N = 1 << 18  # samples per step in training
    ref = HashEncoding(init_scale=1.0).cuda()  # default 16 levels, F=2, T=2^19
    x = torch.rand(N, 3, device="cuda")
    g = torch.randn(N, ref.out_dim, device="cuda")
    y_ref = ref(x)
    y_ref.backward(g)
    gr = ref.embeddings.grad

    encs = {}
    for v in VERSIONS:
        cu = HashEncoding(init_scale=1.0, backend="cuda").cuda()
        cu.cuda_version = v
        cu.embeddings.data.copy_(ref.embeddings.data)
        y = cu(x)
        y.backward(g)
        fwd_err = (y_ref - y).abs().max().item()
        bwd_err = ((gr - cu.embeddings.grad).abs().max() / gr.abs().max()).item()
        print(f"v{v}: forward max abs err {fwd_err:.2e}; backward max err / max |grad| {bwd_err:.2e}")
        assert fwd_err < 1e-5 and bwd_err < 1e-5
        encs[f"cuda v{v}"] = cu

    print(f"\nrandom points, N={N}")
    for name, enc in [("torch", ref)] + list(encs.items()):
        f = bench(lambda: enc(x))
        fb = bench(lambda: enc(x).backward(g))
        print(f"  {name:8s} fwd {f:7.3f} ms   fwd+bwd {fb:7.3f} ms")

    pts = lego_points()
    if pts is not None:
        xl, table = pts
        gl = torch.randn(xl.shape[0], ref.out_dim, device="cuda")
        print(f"\nLego marched points, N={xl.shape[0]}, trained table")
        for name, enc in encs.items():
            enc.embeddings.data.copy_(table)
            with torch.no_grad():
                f = bench(lambda: enc(xl))
            fb = bench(lambda: enc(xl).backward(gl))
            print(f"  {name:8s} fwd {f:7.3f} ms   fwd+bwd {fb:7.3f} ms")


if __name__ == "__main__":
    main()
