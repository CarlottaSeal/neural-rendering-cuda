"""Benchmark a trained 3DGS model with whichever diff_gaussian_rasterization build is installed.

Reports, over the test cameras:
  * PSNR / SSIM against ground truth
  * forward render time (ms, median over views x repeats) and FPS
  * forward+backward time (ms) for an L1 loss, i.e. the rasterizer's training cost without the optimizer
  * peak CUDA memory allocated

Usage (from repo root):
  .venv\\Scripts\\python.exe p1-3dgs\\bench.py -m outputs\\baseline\\train [--repeats 5] [--json out.json]
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, "third_party", "gaussian-splatting")
sys.path.insert(0, GS)

import torch  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from scene import Scene  # noqa: E402
from gaussian_renderer import render, GaussianModel  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402

import diff_gaussian_rasterization as _dgr  # noqa: E402
# dc/rest passed separately when the build supports it (OPT_SEPARATE_SH or upstream's 3dgs_accel build)
SEPARATE_SH = getattr(_dgr, "SEPARATE_SH", False) or hasattr(_dgr, "SparseGaussianAdam")


def timed(fn, repeats):
    out = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--json", default=None)
    parser.add_argument("--label", default="")
    args = get_combined_args(parser)
    # cfg_args stores the source path relative to the cwd used at training time (third_party/gaussian-splatting)
    if not os.path.isabs(args.source_path):
        args.source_path = os.path.normpath(os.path.join(GS, args.source_path))

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    views = scene.getTestCameras()
    assert len(views) > 0, "model was trained without --eval; no test cameras"

    def fwd(v):
        return render(v, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SEPARATE_SH)["render"]

    # warmup
    with torch.no_grad():
        for v in views[:3]:
            fwd(v)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    psnrs, ssims, fwd_ms = [], [], []
    with torch.no_grad():
        for v in views:
            img = fwd(v).clamp(0, 1)
            gt = v.original_image[0:3].cuda()
            psnrs.append(psnr(img, gt).mean().item())
            ssims.append(ssim(img, gt).item())
            fwd_ms += timed(lambda: fwd(v), args.repeats)
    peak_fwd = torch.cuda.max_memory_allocated() / 2**20

    torch.cuda.reset_peak_memory_stats()
    fb_ms = []
    for v in views:
        gt = v.original_image[0:3].cuda()

        def fb():
            loss = (fwd(v) - gt).abs().mean()
            loss.backward()

        fb_ms += timed(fb, args.repeats)
        for p in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                  gaussians._opacity, gaussians._scaling, gaussians._rotation):
            p.grad = None
    peak_fb = torch.cuda.max_memory_allocated() / 2**20

    med = statistics.median(fwd_ms)
    res = {
        "label": args.label,
        "model": args.model_path,
        "num_gaussians": gaussians.get_xyz.shape[0],
        "num_test_views": len(views),
        "resolution": [views[0].image_width, views[0].image_height],
        "psnr": statistics.fmean(psnrs),
        "ssim": statistics.fmean(ssims),
        "fwd_ms_median": med,
        "fps": 1000.0 / med,
        "fwd_bwd_ms_median": statistics.median(fb_ms),
        "peak_mem_fwd_mib": peak_fwd,
        "peak_mem_fwd_bwd_mib": peak_fb,
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
