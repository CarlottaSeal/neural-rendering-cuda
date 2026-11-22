"""Fused no-grad density MLP (csrc/mlp.cu) vs the PyTorch density net: sigma agreement, prune decisions, timing.

  scripts\\p2_run.bat p2-ingp\\tests\\test_mlp.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ingp.cuda_ext import ext  # noqa: E402
from ingp.render import transmittance  # noqa: E402
from test_cuda import bench, lego_points  # noqa: E402

DT = math.sqrt(3) * 3.0 / 1024


def ref_sigma(field, enc):
    return torch.exp(field.density_net(enc)[:, 0])


def main():
    from ingp.model import NGPField
    aabb = torch.tensor([-1.5] * 3 + [1.5] * 3, device="cuda")
    field = NGPField(aabb).cuda()
    field.load_state_dict(torch.load("outputs/p2/ref_lego/field.pt", map_location="cuda"))
    field.encoding.backend = "cuda"
    xl, _ = lego_points()
    w1, w2 = field.density_net[0].weight, field.density_net[2].weight
    with torch.no_grad():
        enc = field.encoding(xl)
        s_ref = ref_sigma(field, enc)
        s_cu = ext().density_mlp_fwd(enc, w1, w2)
    rel = ((s_cu - s_ref).abs() / s_ref.abs().clamp(min=1e-6)).max().item()
    print(f"Lego marched points N={xl.shape[0]}: sigma max rel err {rel:.2e}, max sigma {s_ref.max().item():.1f}")
    assert rel < 1e-4

    # The prune decision T > 1e-4 depends on the cumulative sum of sigma along each ray. The points come from
    # one march over 48,872 rays sorted by ray, so rebuild the ray ids from the march to compare the decisions.
    from ingp.data import BlenderScene
    from ingp.render import OccupancyGrid, march
    grid = OccupancyGrid(aabb)
    field.fused_mlp = False
    with torch.no_grad():
        for _ in range(30):
            grid.update(lambda p: field.density(p), DT)
    torch.manual_seed(0)
    o, d, _ = BlenderScene("data/nerf_synthetic/lego", "train", "cuda").sample_batch(48872)
    r, t = march(o, d, aabb, grid, DT, True, backend="cuda")
    pts = o[r] + t[:, None] * d[r]
    with torch.no_grad():
        field.fused_mlp = False
        k_ref = transmittance(field.density(pts), DT, r, o.shape[0])[0] > 1e-4
        field.fused_mlp = True
        k_cu = transmittance(field.density(pts), DT, r, o.shape[0])[0] > 1e-4
    print(f"prune: {r.numel()} candidates, kept {k_ref.sum().item()} (ref) / {k_cu.sum().item()} (fused), "
          f"{(k_ref != k_cu).sum().item()} decisions differ")

    with torch.no_grad():
        t_ref = bench(lambda: ref_sigma(field, enc))
        t_cu = bench(lambda: ext().density_mlp_fwd(enc, w1, w2))
    print(f"density MLP on {enc.shape[0]} points: torch {t_ref:.3f} ms   fused {t_cu:.3f} ms   ({t_ref / t_cu:.1f}x)")


if __name__ == "__main__":
    main()
