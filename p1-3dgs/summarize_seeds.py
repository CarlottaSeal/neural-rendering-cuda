"""Mean ± std over seeds for each build in outputs/seeds/<scene>/<build>_s<seed>/train_run.json.

  python p1-3dgs\\summarize_seeds.py outputs\\seeds\\train
"""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


def num_gaussians(model_dir):
    plys = glob.glob(os.path.join(model_dir, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not plys:
        return None
    ply = max(plys, key=lambda p: int(re.search(r"iteration_(\d+)", p).group(1)))
    with open(ply, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                return None


def ms(xs, fmt):
    if len(xs) == 1:
        return fmt.format(xs[0])
    return (fmt + " ± " + fmt).format(statistics.fmean(xs), statistics.stdev(xs))


def main(root):
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, "*_s*", "train_run.json"))):
        build = re.match(r"(.+)_s\d+$", os.path.basename(os.path.dirname(p))).group(1)
        with open(p) as f:
            r = json.load(f)
        r["n"] = num_gaussians(os.path.dirname(p))
        runs[build].append(r)
    if not runs:
        sys.exit(f"no runs under {root}")
    base = statistics.fmean(r["wall_s"] for r in runs["upstream"]) if "upstream" in runs else None
    print("| build | seeds | wall s | speedup | test PSNR | train PSNR | #Gaussians |")
    print("|---|---|---|---|---|---|---|")
    for build, rs in runs.items():
        wall = [r["wall_s"] for r in rs]
        sp = f"{base / statistics.fmean(wall):.2f}×" if base else "–"
        test = [r["test_psnr_30000"] for r in rs if "test_psnr_30000" in r]
        train = [r["train_psnr_30000"] for r in rs if "train_psnr_30000" in r]
        n = [r["n"] for r in rs if r["n"]]
        print(f"| {build} | {','.join(str(r['seed']) for r in rs)} | {ms(wall, '{:.1f}')} | {sp} "
              f"| {ms(test, '{:.3f}')} | {ms(train, '{:.3f}')} | {ms(n, '{:,.0f}')} |")


if __name__ == "__main__":
    main(sys.argv[1])
