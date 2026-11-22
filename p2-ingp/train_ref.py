"""Train the pure-PyTorch Instant-NGP reference on a NeRF-synthetic scene.

  python p2-ingp/train_ref.py --data data/nerf_synthetic/lego --iters 30000 --out outputs/p2/ref_lego
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
from ingp.model import NGPField  # noqa: E402
from ingp.render import OccupancyGrid, render_rays  # noqa: E402


def psnr(mse):
    return -10.0 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def evaluate(field, grid, scene, dt, stride=1, chunk=1 << 14, save_dir=None, march_backend="torch"):
    field.eval()
    white = torch.ones(3, device=scene.device)
    psnrs = []
    for i in range(0, scene.n, stride):
        o, d, rgba = scene.image_rays(i)
        gt = rgba[:, :3] * rgba[:, 3:] + (1 - rgba[:, 3:])
        out = torch.cat([render_rays(field, grid, oc, dc, dt, white, march_backend=march_backend)[0] for oc, dc in zip(o.split(chunk), d.split(chunk))])
        psnrs.append(psnr(((out.clamp(0, 1) - gt) ** 2).mean().item()))
        if save_dir is not None and i == 0:
            import cv2
            im = (out.clamp(0, 1).reshape(scene.h, scene.w, 3).cpu().numpy() * 255).astype("uint8")
            cv2.imwrite(os.path.join(save_dir, f"test_{i:03d}.png"), im[..., ::-1])
    field.train()
    return sum(psnrs) / len(psnrs), psnrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nerf_synthetic/lego")
    ap.add_argument("--out", default="outputs/p2/ref_lego")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--target_samples", type=int, default=1 << 18, help="samples per step; the ray batch adapts to it")
    ap.add_argument("--init_rays", type=int, default=4096)
    ap.add_argument("--max_rays", type=int, default=1 << 16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--log2_table", type=int, default=19)
    ap.add_argument("--grid_every", type=int, default=16)
    ap.add_argument("--test_stride", type=int, default=1, help="evaluate every k-th test image")
    ap.add_argument("--eval_every", type=int, default=0, help="quick eval (stride 20) every k iters; 0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoding", default="torch", choices=["torch", "cuda"], help="hash encoding backend")
    ap.add_argument("--march", default="torch", choices=["torch", "cuda"], help="ray marching backend")
    ap.add_argument("--fused_mlp", action="store_true", help="csrc/mlp.cu for the no-grad sigma queries")
    ap.add_argument("--fused_adam", action="store_true", help="torch.optim.Adam(fused=True)")
    ap.add_argument("--fused_train_mlp", action="store_true", help="csrc/mlp_train.cu for the field forward and backward")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=dev)
    dt = math.sqrt(3) * 3.0 / 1024  # 1024 steps across the AABB diagonal, as in instant-ngp

    train = BlenderScene(args.data, "train", dev)
    test = BlenderScene(args.data, "test", dev)
    field = NGPField(aabb, log2_table=args.log2_table).to(dev)
    field.encoding.backend = args.encoding
    field.fused_mlp = args.fused_mlp
    field.fused_train_mlp = args.fused_train_mlp
    grid = OccupancyGrid(aabb, device=dev)

    opt = torch.optim.Adam([
        {"params": field.encoding.parameters(), "weight_decay": 0.0},
        {"params": list(field.density_net.parameters()) + list(field.color_net.parameters()), "weight_decay": 1e-6},
    ], lr=args.lr, betas=(0.9, 0.99), eps=1e-15, fused=args.fused_adam)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.iters // 2, args.iters * 3 // 4, args.iters * 9 // 10], gamma=0.33),
    ], milestones=[100])

    n_rays = args.init_rays
    log, t_train = [], 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        if it % args.grid_every == 0:
            grid.update(lambda p: field.density(p), dt)
        o, d, rgba = train.sample_batch(n_rays)
        bg = torch.rand(n_rays, 3, device=dev)  # random background so the model cannot paint it into the volume
        gt = rgba[:, :3] * rgba[:, 3:] + bg * (1 - rgba[:, 3:])
        color, _, stats = render_rays(field, grid, o, d, dt, bg, jitter=True, march_backend=args.march)
        loss = torch.nn.functional.mse_loss(color, gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if stats["samples"] > 0:
            n_rays = int(min(args.max_rays, max(256, n_rays * args.target_samples / stats["samples"])))
        if it % 500 == 0 or it == 1:
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            row = {"iter": it, "loss": loss.item(), "train_psnr": psnr(loss.item()), "rays": n_rays,
                   "marched": stats["marched"], "samples": stats["samples"], "occupancy": grid.occupancy, "elapsed_s": el}
            log.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every and it % args.eval_every == 0:
            torch.cuda.synchronize()
            t_train += time.perf_counter() - t0
            p, _ = evaluate(field, grid, test, dt, stride=20, march_backend=args.march)
            print(json.dumps({"iter": it, "quick_test_psnr": p}), flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    torch.cuda.synchronize()
    t_train += time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    t1 = time.perf_counter()
    p, ps = evaluate(field, grid, test, dt, stride=args.test_stride, save_dir=args.out, march_backend=args.march)
    t_eval = time.perf_counter() - t1
    res = {"iters": args.iters, "train_s": t_train, "it_per_s": args.iters / t_train, "test_psnr": p,
           "n_test": len(ps), "eval_s": t_eval, "peak_mem_mib": peak, "args": vars(args), "log": log}
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "args")}))
    torch.save(field.state_dict(), os.path.join(args.out, "field.pt"))


if __name__ == "__main__":
    main()
