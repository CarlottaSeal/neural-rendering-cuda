"""Fused training MLP (csrc/mlp_train.cu) vs PyTorch autograd: outputs and all gradients, plus timing.

Errors are measured against an fp64 run of the same net, for both the fp32 torch path and the fused kernels.

  scripts\\p2_run.bat p2-ingp\\tests\\test_mlp_train.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ingp.cuda_ext import FusedMLPFn  # noqa: E402
from ingp.encoding import sh_encoding_deg4  # noqa: E402
from ingp.model import TruncExp  # noqa: E402
from test_cuda import bench, lego_points  # noqa: E402


def torch_mlp(enc, d, ws):
    wd1, wd2, wc1, wc2, wc3 = ws
    h = torch.relu(enc @ wd1.T) @ wd2.T
    sigma = TruncExp.apply(h[:, 0])
    c = torch.cat([h[:, 1:], sh_encoding_deg4(d)], -1)
    rgb = torch.sigmoid(torch.relu(torch.relu(c @ wc1.T) @ wc2.T) @ wc3.T)
    return sigma, rgb


def tf32(x):
    """round an fp32 value to TF32 (nearest, ties away from zero, like cvt.rna.tf32.f32), return fp64"""
    x = x.float().contiguous()
    return ((x.view(torch.int32) + 0x1000) & ~0x1FFF).view(torch.float32).double()


def tf32_emulation(enc, d, ws, gs, gr):
    """the WMMA kernels' math in fp64, rounding to TF32 where they do: MMA operands are TF32, sums are exact"""
    enc, d, gs, gr = enc.double(), d.double(), gs.double(), gr.double()
    wd1, wd2, wc1, wc2, wc3 = (tf32(w) for w in ws)
    h1 = tf32(torch.relu(tf32(enc) @ wd1.T))
    hd = h1 @ wd2.T
    sigma = torch.exp(hd[:, 0])
    c = tf32(torch.cat([hd[:, 1:], sh_encoding_deg4(d.float()).double()], -1))
    a1 = tf32(torch.relu(c @ wc1.T))
    a2 = tf32(torch.relu(a1 @ wc2.T))
    rgb = torch.sigmoid(a2 @ wc3.T)
    go = gr * rgb * (1 - rgb)
    g_a2 = (tf32(go) @ wc3) * (a2 > 0)
    g_a1 = (tf32(g_a2) @ wc2) * (a1 > 0)
    g_hd = torch.cat([(gs * torch.exp(hd[:, 0].clamp(max=15)))[:, None], (tf32(g_a1) @ wc1)[:, :15]], -1)
    g_h1 = (tf32(g_hd) @ wd2) * (h1 > 0)
    g_enc = tf32(g_h1) @ wd1
    # mlp_wgrad: fp32 gradients times the saved (rounded) layer inputs; Wd1 uses the unrounded encoding
    return [sigma, rgb, g_enc, g_h1.T @ enc, g_hd.T @ h1, g_a1.T @ c, g_a2.T @ a1, go.T @ a2]


def run(fn, enc, d, ws, gs, gr):
    enc = enc.detach().requires_grad_()
    ws = [w.detach().requires_grad_() for w in ws]
    sigma, rgb = fn(enc, d, ws)
    torch.autograd.backward([sigma, rgb], [gs, gr])
    return [sigma, rgb, enc.grad] + [w.grad for w in ws]


def main():
    from ingp.model import NGPField
    aabb = torch.tensor([-1.5] * 3 + [1.5] * 3, device="cuda")
    field = NGPField(aabb).cuda()
    field.load_state_dict(torch.load("outputs/p2/ref_lego/field.pt", map_location="cuda"))
    field.encoding.backend = "cuda"
    xl, _ = lego_points()
    torch.manual_seed(0)
    xl = xl[torch.randperm(xl.shape[0], device="cuda")[:262144]]
    with torch.no_grad():
        enc = field.encoding(xl).contiguous()
    N = enc.shape[0]
    d = torch.nn.functional.normalize(torch.randn(N, 3, device="cuda"), dim=-1)
    ws = [w.detach() for w in field.weights()]
    gs, gr = torch.randn(N, device="cuda") * 1e-3, torch.randn(N, 3, device="cuda") * 1e-3

    fused = [lambda e, dd, w, v=v: FusedMLPFn.apply(e, dd, v, *w) for v in (0, 1)]
    names = ["sigma", "rgb", "d enc", "dWd1", "dWd2", "dWc1", "dWc2", "dWc3"]
    for n_pts in (N, 1000 * 16 + 5):  # the second size leaves a partial 16-point tile
        sl = slice(0, n_pts)
        args = enc[sl], d[sl], ws, gs[sl], gr[sl]
        ref64 = run(torch_mlp, enc[sl].double(), d[sl].double(), [w.double() for w in ws], gs[sl].double(), gr[sl].double())
        ref32 = run(torch_mlp, *args)
        outs = [run(f, *args) for f in fused]
        emu = tf32_emulation(*args)
        # max error over max |ref| for the fp32 paths; relative L2 error for TF32, where single values that sit on
        # a rounding boundary differ by one TF32 ulp between the kernel's fp32 sums and the fp64 emulation
        print(f"N={n_pts}: max err / max |fp64| (torch, SIMT); rel. L2 err vs fp64 and vs fp64 TF32 emulation (WMMA)")
        print(f"{'':8s} {'torch fp32':>11s} {'SIMT fp32':>11s} {'WMMA/fp64':>11s} {'WMMA/emul':>11s}")
        rel = lambda x, y: ((x.double() - y).norm() / y.norm()).item()  # noqa: E731
        for i, n in enumerate(names):
            a64 = ref64[i]
            scale = a64.abs().max().item()
            e32, e0 = ((x[i].double() - a64).abs().max().item() / scale for x in (ref32, outs[0]))
            e1, ee = rel(outs[1][i], a64), rel(outs[1][i], emu[i])
            print(f"{n:8s} {e32:11.2e} {e0:11.2e} {e1:11.2e} {ee:11.2e}")
            assert e0 < max(10 * e32, 1e-5), n
            assert ee < 1e-3 and ee < e1 / 10, n

    def step(fn):
        e = enc.detach().requires_grad_()
        w = [x.detach().requires_grad_() for x in ws]
        sigma, rgb = fn(e, d, w)
        torch.autograd.backward([sigma, rgb], [gs, gr])

    with torch.no_grad():
        tf = [bench(lambda f=f: f(enc, d, ws)) for f in [torch_mlp] + fused]
    tb = [bench(lambda f=f: step(f)) for f in [torch_mlp] + fused]
    print(f"N={N}, ms        torch    SIMT    WMMA")
    print(f"no-grad fwd   {tf[0]:7.3f} {tf[1]:7.3f} {tf[2]:7.3f}")
    print(f"fwd+bwd       {tb[0]:7.3f} {tb[1]:7.3f} {tb[2]:7.3f}")


if __name__ == "__main__":
    main()
