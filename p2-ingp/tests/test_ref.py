"""Checks for the PyTorch reference: hash encoding and packed volume rendering against naive loops, plus gradcheck.

  python p2-ingp/tests/test_ref.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ingp.encoding import PRIMES, HashEncoding  # noqa: E402
from ingp.render import composite  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"


def naive_hash_encode(enc, x):
    """One point at a time, one corner at a time, straight from the paper's definition."""
    T = enc.table_size
    out = torch.zeros(x.shape[0], enc.out_dim, dtype=x.dtype)
    for n in range(x.shape[0]):
        for l, res in enumerate(enc.resolutions):
            p = [float(v) * res for v in x[n].clamp(0, 1 - 1e-6)]
            p0 = [math.floor(v) for v in p]
            f = [p[k] - p0[k] for k in range(3)]
            acc = torch.zeros(enc.n_features, dtype=x.dtype)
            for c in range(8):
                v = [p0[k] + ((c >> k) & 1) for k in range(3)]
                w = 1.0
                for k in range(3):
                    w *= f[k] if (c >> k) & 1 else 1 - f[k]
                s = res + 1
                if s ** 3 <= T:
                    idx = v[0] + v[1] * s + v[2] * s * s
                else:
                    idx = ((v[0] * PRIMES[0]) ^ (v[1] * PRIMES[1]) ^ (v[2] * PRIMES[2])) % (2 ** 32) % T
                acc += w * enc.embeddings[l, idx].detach().cpu()
            out[n, l * enc.n_features:(l + 1) * enc.n_features] = acc
    return out


def test_hash_matches_naive():
    enc = HashEncoding(n_levels=8, n_features=2, log2_table=12, n_min=4, n_max=256, init_scale=1.0).double().to(DEV)
    dense = [r for r in enc.resolutions if (r + 1) ** 3 <= enc.table_size]
    assert 0 < len(dense) < len(enc.resolutions), "want both dense and hashed levels"
    x = torch.rand(64, 3, dtype=torch.float64, device=DEV)
    err = (enc(x).cpu() - naive_hash_encode(enc, x.cpu())).abs().max().item()
    print(f"hash vs naive: max abs err {err:.2e} (levels {enc.resolutions}, dense {dense})")
    assert err < 1e-12


def test_hash_gradcheck():
    enc = HashEncoding(n_levels=2, n_features=2, log2_table=6, n_min=2, n_max=8, init_scale=1.0).double().to(DEV)
    x = torch.rand(16, 3, dtype=torch.float64, device=DEV)
    table = enc.embeddings.detach().clone().requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda t: HashFn.apply(enc, x, t), (table,))
    print("hash gradcheck:", ok)
    assert ok


class HashFn(torch.autograd.Function):
    """Wraps the encoding so gradcheck can differentiate w.r.t. the table passed as an input."""

    @staticmethod
    def forward(ctx, enc, x, table):
        ctx.enc, ctx.x = enc, x
        ctx.save_for_backward(table)
        with torch.no_grad():
            enc.embeddings.data = table
            return enc(x)

    @staticmethod
    def backward(ctx, g):
        (table,) = ctx.saved_tensors
        enc = ctx.enc
        with torch.enable_grad():
            enc.embeddings.data = table.detach()
            enc.embeddings.grad = None
            enc(ctx.x).backward(g)
        return None, None, enc.embeddings.grad.clone()


def naive_composite(sigma, rgb, dt, ray_idx, n_rays):
    color = torch.zeros(n_rays, 3, dtype=rgb.dtype, device=rgb.device)
    acc = torch.zeros(n_rays, dtype=rgb.dtype, device=rgb.device)
    for r in range(n_rays):
        m = ray_idx == r
        a = 1 - torch.exp(-sigma[m] * dt)
        T = torch.cumprod(torch.cat([torch.ones(1, dtype=a.dtype, device=a.device), 1 - a[:-1]]), 0)
        w = T * a
        color[r] = (w[:, None] * rgb[m]).sum(0)
        acc[r] = w.sum()
    return color, acc


def packed_problem(n_rays=12, dtype=torch.float64):
    counts = torch.randint(0, 20, (n_rays,))
    ray_idx = torch.repeat_interleave(torch.arange(n_rays), counts).to(DEV)
    n = ray_idx.numel()
    sigma = (torch.rand(n, dtype=dtype, device=DEV) * 50)
    rgb = torch.rand(n, 3, dtype=dtype, device=DEV)
    return sigma, rgb, ray_idx, n_rays


def test_composite_matches_naive():
    sigma, rgb, ray_idx, n_rays = packed_problem(dtype=torch.float32)
    dt = 0.02
    c, a = composite(sigma, rgb, dt, ray_idx, n_rays)
    c0, a0 = naive_composite(sigma, rgb, dt, ray_idx, n_rays)
    err = max((c - c0).abs().max().item(), (a - a0).abs().max().item())
    print(f"composite vs cumprod loop: max abs err {err:.2e} (rays with 0 samples: {(torch.bincount(ray_idx, minlength=n_rays) == 0).sum().item()})")
    assert err < 1e-5


def test_composite_gradcheck():
    sigma, rgb, ray_idx, n_rays = packed_problem()
    sigma.requires_grad_(True)
    rgb.requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda s, c: composite(s, c, 0.02, ray_idx, n_rays), (sigma, rgb))
    print("composite gradcheck:", ok)
    assert ok


if __name__ == "__main__":
    test_hash_matches_naive()
    test_hash_gradcheck()
    test_composite_matches_naive()
    test_composite_gradcheck()
    print("all passed")
