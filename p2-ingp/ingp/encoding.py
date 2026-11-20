"""Pure-PyTorch multiresolution hash encoding (Müller et al. 2022). Reference for the CUDA kernel.

Level l has resolution N_l = floor(N_min * b^l), with b = exp((ln N_max - ln N_min) / (L - 1)).
A point x in [0,1)^3 is scaled to x * N_l, and its 8 surrounding grid vertices are looked up.
Coarse levels whose (N_l + 1)^3 vertices fit in the table are indexed densely; finer levels use the spatial
hash (x * 1) xor (y * 2654435761) xor (z * 805459861) mod T. The 8 features are trilinearly interpolated.
Levels are concatenated -> L * F features. Autograd provides the backward (scatter-add into the table).
"""
import math

import torch
import torch.nn as nn

PRIMES = (1, 2654435761, 805459861)


def level_resolutions(n_levels, n_min, n_max):
    if n_levels == 1:
        return [n_min]
    b = math.exp((math.log(n_max) - math.log(n_min)) / (n_levels - 1))
    return [int(math.floor(n_min * b ** l)) for l in range(n_levels)]


def corner_offsets(device):
    # bit k of corner c selects +1 along axis k
    c = torch.arange(8, device=device)
    return torch.stack([(c >> k) & 1 for k in range(3)], -1)  # [8,3]


def grid_index(corners, res, table_size):
    """corners: [N,8,3] int64 vertex coords in [0, res]. Returns [N,8] table indices."""
    stride = res + 1
    if stride ** 3 <= table_size:
        return corners[..., 0] + corners[..., 1] * stride + corners[..., 2] * stride * stride
    h = (corners[..., 0] * PRIMES[0]) ^ (corners[..., 1] * PRIMES[1]) ^ (corners[..., 2] * PRIMES[2])
    return h & (table_size - 1)  # table_size is a power of two


def trilinear_weights(frac):
    """frac: [N,3] in [0,1). Returns [N,8] weights in the corner_offsets order."""
    off = corner_offsets(frac.device).to(frac.dtype)  # [8,3]
    w = frac[:, None, :] * off + (1 - frac[:, None, :]) * (1 - off)  # [N,8,3]
    return w.prod(-1)


class HashEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_table=19, n_min=16, n_max=2048, init_scale=1e-4, backend="torch"):
        super().__init__()
        self.n_levels, self.n_features = n_levels, n_features
        self.table_size = 2 ** log2_table
        self.resolutions = level_resolutions(n_levels, n_min, n_max)
        self.embeddings = nn.Parameter(torch.empty(n_levels, self.table_size, n_features).uniform_(-init_scale, init_scale))
        self.register_buffer("res_i32", torch.tensor(self.resolutions, dtype=torch.int32), persistent=False)
        self.backend = backend

    @property
    def out_dim(self):
        return self.n_levels * self.n_features

    def forward(self, x):
        """x: [N,3] in [0,1]. Returns [N, L*F]."""
        if self.backend == "cuda":
            from .cuda_ext import HashEncodeFn
            return HashEncodeFn.apply(x, self.embeddings, self.res_i32)
        x = x.clamp(0.0, 1.0 - 1e-6)
        off = corner_offsets(x.device)
        out = []
        for l, res in enumerate(self.resolutions):
            p = x * res
            p0 = torch.floor(p)
            w = trilinear_weights(p - p0)  # [N,8]
            corners = p0.long()[:, None, :] + off[None]  # [N,8,3]
            idx = grid_index(corners, res, self.table_size)  # [N,8]
            feats = self.embeddings[l][idx]  # [N,8,F]
            out.append((feats * w[..., None]).sum(1))
        return torch.cat(out, -1)


def sh_encoding_deg4(d):
    """Real spherical harmonics up to degree 3 (16 coefficients) of unit directions d [N,3], as in tcnn."""
    x, y, z = d.unbind(-1)
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return torch.stack([
        torch.full_like(x, 0.28209479177387814),
        -0.48860251190291987 * y,
        0.48860251190291987 * z,
        -0.48860251190291987 * x,
        1.0925484305920792 * xy,
        -1.0925484305920792 * yz,
        0.94617469575755997 * zz - 0.31539156525251999,
        -1.0925484305920792 * xz,
        0.54627421529603959 * xx - 0.54627421529603959 * yy,
        0.59004358992664352 * y * (-3.0 * xx + yy),
        2.8906114426405538 * xy * z,
        0.45704579946446572 * y * (1.0 - 5.0 * zz),
        0.3731763325901154 * z * (5.0 * zz - 3.0),
        0.45704579946446572 * x * (1.0 - 5.0 * zz),
        1.4453057213202769 * z * (xx - yy),
        0.59004358992664352 * x * (-xx + 3.0 * yy),
    ], -1)
