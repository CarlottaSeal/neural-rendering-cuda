"""CUDA ray marching vs render.march: the packed (ray_idx, t) must match bit for bit, plus timing.

  scripts\\p2_run.bat p2-ingp\\tests\\test_march.py [--ckpt outputs\\p2\\ref_lego\\field.pt]
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ingp.render import OccupancyGrid, march  # noqa: E402
from test_cuda import bench  # noqa: E402

AABB = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]
DT = math.sqrt(3) * 3.0 / 1024


def compare(name, o, d, grid, aabb, jitter):
    torch.manual_seed(1)
    r_ref, t_ref = march(o, d, aabb, grid, DT, jitter)
    torch.manual_seed(1)
    r_cu, t_cu = march(o, d, aabb, grid, DT, jitter, backend="cuda")
    same = r_ref.shape == r_cu.shape and torch.equal(r_ref, r_cu) and torch.equal(t_ref, t_cu)
    print(f"{name}: {o.shape[0]} rays, {r_ref.numel()} samples (cuda {r_cu.numel()}), bitwise equal: {same}")
    if not same and r_ref.shape == r_cu.shape:
        print(f"  ray mismatches {(r_ref != r_cu).sum().item()}, max |dt| {(t_ref - t_cu).abs().max().item():.3e}")
    assert same
    return r_ref.numel()


def edge_cases(aabb):
    g = torch.Generator(device="cuda").manual_seed(0)
    grid = OccupancyGrid(aabb)
    grid.occ = torch.rand(grid.res ** 3, device="cuda", generator=g) < 0.3
    n = 4096
    o = (torch.rand(n, 3, device="cuda", generator=g) * 2 - 1) * 4  # some inside the box, most outside
    d = torch.randn(n, 3, device="cuda", generator=g)
    d[: n // 8, 0] = 0  # axis-parallel rays
    d[n // 8: n // 4, 1:] = 0
    d = d / d.norm(dim=-1, keepdim=True)
    o[-16:] = torch.tensor([5.0, 5.0, 5.0], device="cuda")  # pointing away: no samples
    d[-16:] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    compare("edge cases, jitter", o, d, grid, aabb, True)
    compare("edge cases, no jitter", o, d, grid, aabb, False)


def lego(aabb, ckpt, data):
    from ingp.data import BlenderScene
    from ingp.model import NGPField
    field = NGPField(aabb).cuda()
    field.load_state_dict(torch.load(ckpt, map_location="cuda"))
    grid = OccupancyGrid(aabb)
    with torch.no_grad():
        for _ in range(30):
            grid.update(lambda p: field.density(p), DT)
    scene = BlenderScene(data, "train", "cuda")
    torch.manual_seed(0)
    o, d, _ = scene.sample_batch(48872)  # steady-state batch size of the Lego run
    compare(f"lego, occupancy {grid.occupancy:.3f}", o, d, grid, aabb, True)
    t_ref = bench(lambda: march(o, d, aabb, grid, DT, True))
    t_cu = bench(lambda: march(o, d, aabb, grid, DT, True, backend="cuda"))
    print(f"march, 48872 rays: torch {t_ref:.3f} ms   cuda v0 {t_cu:.3f} ms   ({t_ref / t_cu:.1f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/p2/ref_lego/field.pt")
    ap.add_argument("--data", default="data/nerf_synthetic/lego")
    a = ap.parse_args()
    aabb = torch.tensor(AABB, device="cuda")
    edge_cases(aabb)
    if os.path.exists(a.ckpt):
        lego(aabb, a.ckpt, a.data)


if __name__ == "__main__":
    main()
