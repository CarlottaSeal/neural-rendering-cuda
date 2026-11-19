"""Per-tile workload (number of Gaussians binned to each 16x16 tile) for one view, reproducing the
rasterizer's binning in PyTorch: project means, take the radii the rasterizer returns, and count each
Gaussian in every tile of getRect() using a 2D difference array.

  python p1-3dgs\\tile_load.py -m outputs\\baseline\\train [--view 0]

The count is the length of each tile's [start, end) range, i.e. the upper bound on the forward/backward
per-tile loop. It ignores early termination, so it overstates the cost of tiles that saturate early.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, "third_party", "gaussian-splatting")
sys.path.insert(0, GS)

import torch  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from scene import Scene  # noqa: E402
from gaussian_renderer import render, GaussianModel  # noqa: E402

BLOCK = 16


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--view", default=0, type=int)
    args = get_combined_args(parser)
    if not os.path.isabs(args.source_path):
        args.source_path = os.path.normpath(os.path.join(GS, args.source_path))
    dataset, pipe = model.extract(args), pipeline.extract(args)
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    v = scene.getTestCameras()[args.view]
    W, H = v.image_width, v.image_height
    with torch.no_grad():
        radii = render(v, g, pipe, torch.zeros(3, device="cuda"))["radii"]
        xyz = g.get_xyz
        hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ v.full_proj_transform  # row-vector convention
        w = 1.0 / (hom[:, 3:4] + 1e-7)
        ndc = hom[:, :2] * w
        px = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        py = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5
        vis = radii > 0
        px, py, r = px[vis], py[vis], radii[vis].float()
        gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
        # getRect(): min = clamp((p - r) / 16), max = clamp((p + r + 15) / 16), integer division
        x0 = ((px - r) / BLOCK).floor().clamp(0, gx).long()
        x1 = ((px + r + BLOCK - 1) / BLOCK).floor().clamp(0, gx).long()
        y0 = ((py - r) / BLOCK).floor().clamp(0, gy).long()
        y1 = ((py + r + BLOCK - 1) / BLOCK).floor().clamp(0, gy).long()
        ok = (x1 > x0) & (y1 > y0)
        x0, x1, y0, y1 = x0[ok], x1[ok], y0[ok], y1[ok]
        diff = torch.zeros((gy + 1) * (gx + 1), device="cuda", dtype=torch.int64)
        idx = lambda y, x: y * (gx + 1) + x  # noqa: E731
        for yy, xx, s in ((y0, x0, 1), (y0, x1, -1), (y1, x0, -1), (y1, x1, 1)):
            diff.index_add_(0, idx(yy, xx), torch.full_like(yy, s))
        load = diff.view(gy + 1, gx + 1).cumsum(0).cumsum(1)[:gy, :gx].flatten().double()

    n = load.numel()
    total = load.sum().item()
    s, _ = load.sort(descending=True)
    q = torch.quantile(load, torch.tensor([0.5, 0.9, 0.99], dtype=torch.double, device=load.device))
    print(f"view {args.view}: {W}x{H}, {gx}x{gy} = {n} tiles, {int(vis.sum())} visible Gaussians")
    print(f"tile ranges total {int(total):,} (num_rendered)")
    print(f"per tile: mean {load.mean():.0f}  median {q[0]:.0f}  p90 {q[1]:.0f}  p99 {q[2]:.0f}  max {s[0]:.0f}  "
          f"max/mean {s[0] / load.mean():.1f}x")
    print(f"empty tiles: {(load == 0).sum().item()}")
    for k in (1, 10, 70, 280):
        print(f"heaviest {k:4d} tiles hold {s[:k].sum() / total * 100:5.1f}% of the work")
    # an ideal schedule on S SMs x B resident blocks finishes no earlier than max(total / (S*B), max tile)
    for blocks_per_sm, name in ((4, "backward (4 blocks/SM)"), (6, "forward (6 blocks/SM)")):
        slots = 70 * blocks_per_sm
        print(f"{name}: work per slot {total / slots:,.0f} vs heaviest tile {s[0]:,.0f} "
              f"-> heaviest tile = {s[0] / (total / slots):.2f}x of a balanced slot")


if __name__ == "__main__":
    main()
