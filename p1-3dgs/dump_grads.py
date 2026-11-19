"""Dump forward images + parameter gradients from the installed rasterizer, or compare two dumps.

  dump:    python p1-3dgs\\dump_grads.py -m outputs\\baseline\\train --out outputs\\grads\\upstream.pt
  compare: python p1-3dgs\\dump_grads.py --compare outputs\\grads\\upstream.pt outputs\\grads\\opt.pt

Uses L1 + an inverse-depth term so the depth-gradient path is exercised too.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, "third_party", "gaussian-splatting")
sys.path.insert(0, GS)

import torch  # noqa: E402

PARAMS = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    worst = 0.0
    for k in a:
        x, y = a[k].double(), b[k].double()
        scale = x.abs().max().item() or 1.0
        err = (x - y).abs().max().item() / scale
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
        worst = max(worst, err)
        print(f"{k:28s} max|diff|/max|ref| = {err:.3e}   cos = {cos:.9f}")
    print(f"worst relative error: {worst:.3e}")
    return worst


def dump(argv):
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    import diff_gaussian_rasterization as dgr
    separate_sh = getattr(dgr, "SEPARATE_SH", False)
    print("separate_sh =", separate_sh)

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--views", default=4, type=int)
    parser.add_argument("--out", required=True)
    sys.argv = [sys.argv[0]] + argv
    args = get_combined_args(parser)
    if not os.path.isabs(args.source_path):
        args.source_path = os.path.normpath(os.path.join(GS, args.source_path))
    dataset, pipe = model.extract(args), pipeline.extract(args)
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    bg = torch.zeros(3, device="cuda")
    out = {}
    for i, v in enumerate(scene.getTestCameras()[: args.views]):
        for p in PARAMS:
            getattr(g, p).grad = None
        pkg = render(v, g, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh)
        gt = v.original_image[0:3].cuda()
        loss = (pkg["render"] - gt).abs().mean() + 0.1 * pkg["depth"].mean()
        loss.backward()
        out[f"view{i}/render"] = pkg["render"].detach().cpu()
        out[f"view{i}/depth"] = pkg["depth"].detach().cpu()
        for p in PARAMS:
            out[f"view{i}/grad{p}"] = getattr(g, p).grad.detach().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print("saved", args.out, len(out), "tensors")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        dump(sys.argv[1:])
