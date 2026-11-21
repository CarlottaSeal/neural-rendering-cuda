"""Occupancy grid, ray marching and packed volume rendering, all in plain PyTorch.

Samples are "packed": one flat list for all rays, sorted by ray then by t, each tagged with its ray index.
This is the layout the CUDA kernels will use, so this file doubles as their reference.
"""
import math

import torch


def ray_aabb(o, d, aabb, near=0.0):
    lo, hi = aabb[:3], aabb[3:]
    inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t0, t1 = (lo - o) * inv, (hi - o) * inv
    tmin = torch.minimum(t0, t1).amax(-1).clamp(min=near)
    tmax = torch.maximum(t0, t1).amin(-1)
    return tmin, tmax


class OccupancyGrid:
    """res^3 grid of running-max densities. A cell is occupied if sigma * dt exceeds min(0.01, mean)."""

    def __init__(self, aabb, res=128, decay=0.95, thresh=0.01, device="cuda"):
        self.aabb, self.res, self.decay, self.thresh = aabb, res, decay, thresh
        self.aabb_host = aabb.tolist()  # for kernel arguments, without a device sync per call
        self.density = torch.zeros(res ** 3, device=device)
        self.occ = torch.ones(res ** 3, dtype=torch.bool, device=device)  # everything occupied until the first update
        g = torch.arange(res, device=device)
        self.coords = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)  # cell (i,j,k) -> i*res^2+j*res+k

    def cell_index(self, pts):
        lo, hi = self.aabb[:3], self.aabb[3:]
        c = ((pts - lo) / (hi - lo) * self.res).long().clamp(0, self.res - 1)
        return (c[..., 0] * self.res + c[..., 1]) * self.res + c[..., 2]

    def query(self, pts):
        return self.occ[self.cell_index(pts)]

    @torch.no_grad()
    def update(self, density_fn, dt, chunk=1 << 19):
        lo, hi = self.aabb[:3], self.aabb[3:]
        pts = (self.coords + torch.rand_like(self.coords, dtype=torch.float32)) / self.res * (hi - lo) + lo
        sigma = torch.cat([density_fn(p) for p in pts.split(chunk)])
        self.density = torch.maximum(self.density * self.decay, sigma)
        a = self.density * dt
        self.occ = a > min(self.thresh, a.mean().item())

    @property
    def occupancy(self):
        return self.occ.float().mean().item()


def march(o, d, aabb, grid, dt, jitter, max_candidates=1 << 23, backend="torch"):
    """Uniform steps of dt inside the AABB, keeping samples in occupied cells. Returns packed (ray_idx, t).

    Candidates are generated packed (only inside [tmin, tmax] of each ray) and in ray chunks of at most
    max_candidates, so memory stays bounded however many rays the batch has.
    backend="cuda" runs csrc/march.cu, which draws the same jitter and returns the same samples.
    """
    if backend == "cuda":
        from .cuda_ext import ext
        n = o.shape[0]
        u = torch.rand(n, device=o.device) if jitter else torch.full((n,), 0.5, device=o.device)
        return tuple(ext().march_rays(o.contiguous(), d.contiguous(), u, grid.occ.contiguous(),
                                      grid.aabb_host, grid.res, dt))  # the grid's AABB is the field's
    tmin, tmax = ray_aabb(o, d, aabb)
    u = torch.rand_like(tmin) if jitter else torch.full_like(tmin, 0.5)
    # t_k = tmin + (k + u) dt < tmax  <=>  k < (tmax - tmin)/dt - u
    n = torch.ceil((tmax - tmin) / dt - u).clamp(min=0).long()
    csum = torch.cumsum(n, 0)
    total = int(csum[-1].item()) if n.numel() else 0
    out_r, out_t = [], []
    r0 = 0
    while r0 < n.numel() and total:
        base = int(csum[r0 - 1].item()) if r0 else 0
        r1 = int(torch.searchsorted(csum, base + max_candidates, right=True).item())
        r1 = max(r1, r0 + 1)
        nc = n[r0:r1]
        ray = torch.repeat_interleave(torch.arange(r0, r1, device=o.device), nc)
        if ray.numel():
            k = torch.arange(ray.numel(), device=o.device) - (torch.cumsum(nc, 0) - nc).repeat_interleave(nc)
            t = tmin[ray] + (k + u[ray]) * dt
            keep = grid.query(o[ray] + t[:, None] * d[ray])
            out_r.append(ray[keep])
            out_t.append(t[keep])
        r0 = r1
    if not out_r:
        e = torch.empty(0, device=o.device)
        return e.long(), e
    return torch.cat(out_r), torch.cat(out_t)  # sorted by ray, then t


def transmittance(sigma, dt, ray_idx, n_rays):
    """Packed per-sample transmittance T_i = exp(-sum_{j<i, same ray} sigma_j dt) and alpha_i. Cumsum in fp64."""
    sd = sigma.double() * dt
    excl = torch.cumsum(sd, 0) - sd
    counts = torch.bincount(ray_idx, minlength=n_rays)
    starts = torch.cumsum(counts, 0) - counts
    base = excl[starts[ray_idx]]
    T = torch.exp(-(excl - base)).to(sigma.dtype)
    alpha = (-torch.expm1(-sd)).to(sigma.dtype)
    return T, alpha


def composite(sigma, rgb, dt, ray_idx, n_rays):
    T, alpha = transmittance(sigma, dt, ray_idx, n_rays)
    w = T * alpha
    color = torch.zeros(n_rays, 3, device=rgb.device, dtype=rgb.dtype).index_add_(0, ray_idx, w[:, None] * rgb)
    acc = torch.zeros(n_rays, device=rgb.device, dtype=w.dtype).index_add_(0, ray_idx, w)
    return color, acc


def render_rays(field, grid, o, d, dt, bg, jitter=False, t_thresh=1e-4, march_backend="torch"):
    """Two passes: (1) density only, no grad, to find samples after T < t_thresh; (2) full field on the rest."""
    n_rays = o.shape[0]
    ray_idx, t = march(o, d, field.aabb, grid, dt, jitter, backend=march_backend)
    n_marched = ray_idx.numel()
    if n_marched:
        with torch.no_grad():
            pts = o[ray_idx] + t[:, None] * d[ray_idx]
            T, _ = transmittance(field.density(pts), dt, ray_idx, n_rays)
            keep = T > t_thresh
        ray_idx, t = ray_idx[keep], t[keep]
    if ray_idx.numel() == 0:
        color = torch.zeros(n_rays, 3, device=o.device)
        acc = torch.zeros(n_rays, device=o.device)
    else:
        pts = o[ray_idx] + t[:, None] * d[ray_idx]
        sigma, rgb = field(pts, d[ray_idx])
        color, acc = composite(sigma, rgb, dt, ray_idx, n_rays)
    color = color + (1.0 - acc[:, None]) * bg
    return color, acc, {"marched": n_marched, "samples": ray_idx.numel()}
