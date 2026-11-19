"""Print a markdown table from bench.py JSON files. Speedups are relative to the 'upstream' row."""
import glob
import json
import sys

rows = []
for pat in sys.argv[1:]:
    for p in sorted(glob.glob(pat)):
        with open(p) as f:
            rows.append(json.load(f))
base = next((r for r in rows if r["label"] == "upstream"), rows[0])
print("| variant | PSNR | ΔPSNR | fwd ms | FPS | fwd speedup | fwd+bwd ms | fwd+bwd speedup | peak MiB (fwd+bwd) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['label']} | {r['psnr']:.3f} | {r['psnr'] - base['psnr']:+.3f} | {r['fwd_ms_median']:.2f} | {r['fps']:.0f} "
          f"| {base['fwd_ms_median'] / r['fwd_ms_median']:.2f}x | {r['fwd_bwd_ms_median']:.2f} "
          f"| {base['fwd_bwd_ms_median'] / r['fwd_bwd_ms_median']:.2f}x | {r['peak_mem_fwd_bwd_mib']:.0f} |")
