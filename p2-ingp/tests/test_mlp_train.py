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

    fused = lambda e, dd, w: FusedMLPFn.apply(e, dd, *w)  # noqa: E731
    ref64 = run(torch_mlp, enc.double(), d.double(), [w.double() for w in ws], gs.double(), gr.double())
    ref32 = run(torch_mlp, enc, d, ws, gs, gr)
    out = run(fused, enc, d, ws, gs, gr)
    names = ["sigma", "rgb", "d enc", "dWd1", "dWd2", "dWc1", "dWc2", "dWc3"]
    print(f"N={N}, error relative to max |fp64 value|")
    print(f"{'':8s} {'torch fp32':>12s} {'fused':>12s}")
    for n, a64, a32, af in zip(names, ref64, ref32, out):
        scale = a64.abs().max().item()
        e32 = (a32.double() - a64).abs().max().item() / scale
        ef = (af.double() - a64).abs().max().item() / scale
        print(f"{n:8s} {e32:12.2e} {ef:12.2e}")
        assert ef < max(10 * e32, 1e-5), n

    def step(fn):
        e = enc.detach().requires_grad_()
        w = [x.detach().requires_grad_() for x in ws]
        sigma, rgb = fn(e, d, w)
        torch.autograd.backward([sigma, rgb], [gs, gr])

    with torch.no_grad():
        t_ref_f = bench(lambda: torch_mlp(enc, d, ws))
        t_cu_f = bench(lambda: fused(enc, d, ws))
    t_ref = bench(lambda: step(torch_mlp))
    t_cu = bench(lambda: step(fused))
    print(f"no-grad fwd: torch {t_ref_f:.3f} ms   fused {t_cu_f:.3f} ms   ({t_ref_f / t_cu_f:.1f}x)")
    print(f"fwd+bwd:     torch {t_ref:.3f} ms   fused {t_cu:.3f} ms   ({t_ref / t_cu:.1f}x)")


if __name__ == "__main__":
    main()
