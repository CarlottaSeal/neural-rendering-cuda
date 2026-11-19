"""Per-kernel GPU time inside the NVTX `step/*` ranges of an nsys report (from profile_step.py).

  python p1-3dgs\\kernel_breakdown.py outputs\\prof\\upstream_nsys.nsys-rep [--top 15]

Exports the report to SQLite with `nsys export` (once) and joins kernels to their NVTX range via the
CUDA runtime launch correlation, so warmup/scene-loading kernels are excluded.
"""
import argparse
import os
import re
import sqlite3
import subprocess

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.1\target-windows-x64\nsys.exe"


def short(name):
    # PyTorch kernels: keep the functor/op name, which is what identifies them
    m = re.search(r"at::native::(?:<unnamed>::)?(\w+(?:_kernel\w*|Functor\w*))", name)
    if m and not name.startswith("void at::native::reduce_kernel"):
        inner = re.findall(r"at::native::(?:<unnamed>::)?(\w+)", name)
        return "torch:" + "/".join(dict.fromkeys(inner[1:4] or inner[:1]))[:70]
    m = re.search(r"(\w+)\s*(?:<[^()]*>)?\s*\(", name)
    return (m.group(1) if m else name)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    db = os.path.splitext(a.rep)[0] + ".sqlite"
    if not os.path.exists(db):
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, a.rep], check=True)
    con = sqlite3.connect(db)
    q = """
    WITH steps AS (
      SELECT n.start, n.end, n.globalTid FROM NVTX_EVENTS n
      LEFT JOIN StringIds s ON s.id = n.textId
      WHERE COALESCE(n.text, s.value) LIKE 'step/%'
    ), phases AS (
      SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS phase FROM NVTX_EVENTS n
      LEFT JOIN StringIds s ON s.id = n.textId
      WHERE COALESCE(n.text, s.value) IN ('forward', 'backward')
    )
    SELECT ks.value AS kname, ph.phase, k.end - k.start AS dur
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
    -- match on time only: backward kernels are launched from autograd's worker thread, not the NVTX thread
    JOIN steps st ON r.start BETWEEN st.start AND st.end
    LEFT JOIN phases ph ON r.start BETWEEN ph.start AND ph.end
    JOIN StringIds ks ON ks.id = k.demangledName
    """
    rows = con.execute(q).fetchall()
    n_steps = con.execute("""SELECT COUNT(*) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
                             WHERE COALESCE(n.text, s.value) LIKE 'step/%'""").fetchone()[0]
    agg = {}
    for kname, phase, dur in rows:
        key = (short(kname), phase or "?")
        t, c = agg.get(key, (0, 0))
        agg[key] = (t + dur, c + 1)
    total = sum(t for t, _ in agg.values())
    print(f"{n_steps} steps, total GPU kernel time per step: {total / n_steps / 1e6:.3f} ms\n")
    print(f"| kernel | phase | ms/step | % | launches/step |\n|---|---|---|---|---|")
    for (k, ph), (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"| `{k}` | {ph} | {t / n_steps / 1e6:.3f} | {100 * t / total:.1f} | {c / n_steps:g} |")


if __name__ == "__main__":
    main()
