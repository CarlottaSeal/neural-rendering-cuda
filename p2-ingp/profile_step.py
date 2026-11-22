"""Steady-state training-step breakdown of the Instant-NGP reference, from a trained checkpoint.

Loads field.pt, rebuilds the occupancy grid, lets the ray batch adapt to --target_samples, then runs
--steps training steps three ways:
  1. plain: no extra syncs, wall time per step (the real it/s)
  2. phased: torch.cuda.synchronize() between phases, wall time per phase
  3. NVTX ranges per phase, for nsys + p1-3dgs/kernel_breakdown.py (GPU kernel time per phase)

  scripts\\p2_run.bat p2-ingp\\profile_step.py --ckpt outputs\\p2\\ref_lego\\field.pt
  nsys profile -t cuda,nvtx -o outputs\\prof\\p2_ref .venv\\Scripts\\python.exe p2-ingp\\profile_step.py --nvtx_only
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingp.data import BlenderScene  # noqa: E402
from ingp.encoding import sh_encoding_deg4  # noqa: E402
from ingp.model import NGPField, TruncExp  # noqa: E402
from ingp.render import OccupancyGrid, composite, march, transmittance  # noqa: E402

PHASES = ["grid_update", "sample", "march", "prune", "encoding", "density_mlp", "color_mlp", "composite", "backward", "optim"]


class Phases:
    """mode: None (no-op), "sync" (wall time per phase), "nvtx" (NVTX range per phase)."""

    def __init__(self, mode):
        self.mode, self.t = mode, {}

    def __call__(self, name):
        self.name = name
        return self

    def __enter__(self):
        if self.mode == "sync":
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        elif self.mode == "nvtx":
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *exc):
        if self.mode == "sync":
            torch.cuda.synchronize()
            self.t[self.name] = self.t.get(self.name, 0.0) + time.perf_counter() - self.t0
        elif self.mode == "nvtx":
            torch.cuda.nvtx.range_pop()


def step(field, grid, scene, opt, n_rays, dt, ph, do_grid, march_backend, t_thresh=1e-4):
    """One training step, same math as train_ref.py + render.render_rays, with the field forward split up."""
    if do_grid:
        with ph("grid_update"):
            grid.update(lambda p: field.density(p), dt)
    with ph("sample"):
        o, d, rgba = scene.sample_batch(n_rays)
        bg = torch.rand(n_rays, 3, device=o.device)
        gt = rgba[:, :3] * rgba[:, 3:] + bg * (1 - rgba[:, 3:])
    with ph("march"):
        ray_idx, t = march(o, d, field.aabb, grid, dt, jitter=True, backend=march_backend)
        n_marched = ray_idx.numel()
    with ph("prune"):
        with torch.no_grad():
            pts = o[ray_idx] + t[:, None] * d[ray_idx]
            T, _ = transmittance(field.density(pts), dt, ray_idx, n_rays)
            keep = T > t_thresh
        ray_idx, t = ray_idx[keep], t[keep]
        pts = o[ray_idx] + t[:, None] * d[ray_idx]
    with ph("encoding"):
        enc = field.encoding(field.normalize(pts))
    with ph("density_mlp"):
        h = field.density_net(enc)
        sigma = TruncExp.apply(h[:, 0])
    with ph("color_mlp"):
        rgb = torch.sigmoid(field.color_net(torch.cat([h[:, 1:], sh_encoding_deg4(d[ray_idx])], -1)))
    with ph("composite"):
        color, acc = composite(sigma, rgb, dt, ray_idx, n_rays)
        color = color + (1.0 - acc[:, None]) * bg
        loss = torch.nn.functional.mse_loss(color, gt)
    with ph("backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    with ph("optim"):
        opt.step()
    return n_marched, ray_idx.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nerf_synthetic/lego")
    ap.add_argument("--ckpt", default="outputs/p2/ref_lego/field.pt")
    ap.add_argument("--backend", default="torch", choices=["torch", "cuda"], help="hash encoding backend")
    ap.add_argument("--march", default="torch", choices=["torch", "cuda"], help="ray marching backend")
    ap.add_argument("--fused_mlp", action="store_true", help="csrc/mlp.cu for the no-grad sigma queries")
    ap.add_argument("--fused_adam", action="store_true", help="torch.optim.Adam(fused=True)")
    ap.add_argument("--target_samples", type=int, default=1 << 18)
    ap.add_argument("--grid_every", type=int, default=16)
    ap.add_argument("--adapt_steps", type=int, default=60)
    ap.add_argument("--steps", type=int, default=48, help="a multiple of grid_every keeps the grid update amortized right")
    ap.add_argument("--nvtx_only", action="store_true", help="skip the timed runs, only emit NVTX steps (for nsys)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=dev)
    dt = math.sqrt(3) * 3.0 / 1024
    scene = BlenderScene(args.data, "train", dev)
    field = NGPField(aabb).to(dev)
    field.load_state_dict(torch.load(args.ckpt, map_location=dev))
    field.encoding.backend = args.backend
    field.fused_mlp = args.fused_mlp
    grid = OccupancyGrid(aabb, device=dev)
    for _ in range(30):  # running max with decay 0.95 settles in a few dozen updates
        grid.update(lambda p: field.density(p), dt)
    lr_end = 1e-2 * 0.33 ** 3  # lr in the last 10% of training
    opt = torch.optim.Adam([
        {"params": field.encoding.parameters(), "weight_decay": 0.0},
        {"params": list(field.density_net.parameters()) + list(field.color_net.parameters()), "weight_decay": 1e-6},
    ], lr=lr_end, betas=(0.9, 0.99), eps=1e-15, fused=args.fused_adam)

    n_rays, it = 4096, 0

    def run(n, mode):
        nonlocal n_rays, it
        ph = Phases(mode)
        marched = samples = 0
        for i in range(n):
            it += 1
            if mode == "nvtx":
                torch.cuda.nvtx.range_push(f"step/{i}")
            m, s = step(field, grid, scene, opt, n_rays, dt, ph, it % args.grid_every == 0, args.march)
            if mode == "nvtx":
                torch.cuda.nvtx.range_pop()
            marched, samples = marched + m, samples + s
            if s:
                n_rays = int(min(1 << 16, max(256, n_rays * args.target_samples / s)))
        return ph.t, marched / n, samples / n

    run(args.adapt_steps, None)
    it = 0  # so each measured run sees the same grid-update phase
    res = {"backend": args.backend, "march": args.march, "fused_mlp": args.fused_mlp, "fused_adam": args.fused_adam, "steps": args.steps, "rays": n_rays, "occupancy": grid.occupancy}
    if not args.nvtx_only:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, res["marched"], res["samples"] = run(args.steps, None)
        torch.cuda.synchronize()
        res["plain_ms_per_step"] = (time.perf_counter() - t0) / args.steps * 1e3
        res["it_per_s"] = 1e3 / res["plain_ms_per_step"]
        it = 0
        t, _, _ = run(args.steps, "sync")
        res["phase_ms"] = {k: t.get(k, 0.0) / args.steps * 1e3 for k in PHASES}
        res["phased_ms_per_step"] = sum(res["phase_ms"].values())
    it = 0
    torch.cuda.synchronize()
    run(args.steps, "nvtx")
    torch.cuda.synchronize()
    res["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
