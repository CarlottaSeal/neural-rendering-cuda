"""Minimal driver for Nsight: N iterations of fwd+bwd on one test view, NVTX-annotated.

  ncu  --nvtx --nvtx-include "step/" --set full -o outputs\\prof\\upstream ^
       .venv\\Scripts\\python.exe p1-3dgs\\profile_step.py -m outputs\\baseline\\train
  nsys profile -t cuda,nvtx -o outputs\\prof\\upstream .venv\\Scripts\\python.exe p1-3dgs\\profile_step.py -m ...
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
from gaussian_renderer import render as _render, GaussianModel  # noqa: E402
import diff_gaussian_rasterization as _dgr  # noqa: E402

SEPARATE_SH = getattr(_dgr, "SEPARATE_SH", False)


def render(v, g, pipe, bg):
    return _render(v, g, pipe, bg, separate_sh=SEPARATE_SH)


parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--view", default=0, type=int)
parser.add_argument("--steps", default=2, type=int)
args = get_combined_args(parser)
if not os.path.isabs(args.source_path):
    args.source_path = os.path.normpath(os.path.join(GS, args.source_path))
dataset, pipe = model.extract(args), pipeline.extract(args)
g = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
v = scene.getTestCameras()[args.view]
gt = v.original_image[0:3].cuda()
bg = torch.zeros(3, device="cuda")

for _ in range(2):  # warmup, outside the NVTX range
    (render(v, g, pipe, bg)["render"] - gt).abs().mean().backward()
torch.cuda.synchronize()
for i in range(args.steps):
    torch.cuda.nvtx.range_push(f"step/{i}")
    torch.cuda.nvtx.range_push("forward")
    img = render(v, g, pipe, bg)["render"]
    loss = (img - gt).abs().mean()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("done")
