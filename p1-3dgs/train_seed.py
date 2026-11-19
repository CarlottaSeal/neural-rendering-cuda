"""Run upstream train.py with a chosen RNG seed and record wall time / logged PSNR to JSON.

upstream's safe_state() hard-codes seed 0. This wrapper re-seeds right after it, without editing train.py:

  python p1-3dgs\\train_seed.py --seed 1 --json outputs\\seeds\\all3_s1\\train_run.json -- \\
         -s data\\tandt\\train -m outputs\\seeds\\all3_s1 --eval --disable_viewer

The arguments after `--` go to train.py unchanged. Even with a fixed seed, runs are not bit-reproducible:
float atomics in the rasterizer backward accumulate in a nondeterministic order.
"""
import json
import os
import random
import re
import runpy
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, "third_party", "gaussian-splatting")


def main():
    argv = sys.argv[1:]
    split = argv.index("--")
    own, train_args = argv[:split], argv[split + 1:]
    seed = int(own[own.index("--seed") + 1])
    out_json = own[own.index("--json") + 1]
    # train.py paths are given relative to the repo root; make them absolute before chdir
    for flag in ("-s", "--source_path", "-m", "--model_path"):
        if flag in train_args:
            i = train_args.index(flag) + 1
            train_args[i] = os.path.abspath(train_args[i])
    out_json = os.path.abspath(out_json)

    os.chdir(GS)
    sys.path.insert(0, GS)
    import numpy as np
    import torch
    import utils.general_utils as gu

    upstream_safe_state = gu.safe_state

    def safe_state(silent):
        upstream_safe_state(silent)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    gu.safe_state = safe_state  # train.py does `from utils.general_utils import safe_state` at import time

    # tee stdout so the PSNR lines train.py prints can be parsed afterwards
    # (safe_state wraps whatever sys.stdout is at that point, i.e. this object)
    captured = []
    real_stdout = sys.stdout

    class Tee:
        def write(self, s):
            captured.append(s)
            return real_stdout.write(s)

        def flush(self):
            real_stdout.flush()

        def __getattr__(self, name):
            return getattr(real_stdout, name)

    sys.stdout = Tee()

    import diff_gaussian_rasterization as dgr
    sys.argv = ["train.py"] + train_args
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runpy.run_path("train.py", run_name="__main__")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sys.stdout = real_stdout

    text = "".join(captured)
    evals = {}
    # values are printed as tensors: "... L1 tensor(0.04, device='cuda:0', dtype=torch.float64) PSNR tensor(22.1, ...)"
    for it, split_name, psnr in re.findall(r"\[ITER (\d+)\] Evaluating (test|train): L1 .*? PSNR (?:tensor\()?([\d.]+)", text):
        evals[f"{split_name}_psnr_{it}"] = float(psnr)
    res = {"seed": seed, "wall_s": wall, "separate_sh_build": bool(getattr(dgr, "SEPARATE_SH", False)),
           "train_args": train_args, **evals}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
