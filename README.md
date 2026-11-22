# neural-rendering-cuda

GPU kernel engineering for neural rendering, done as three projects. See [DEVPLAN.md](DEVPLAN.md) for the schedule.

| Project | Directory | Status |
|---|---|---|
| 1. 3D Gaussian Splatting: study and optimize the official CUDA rasterizer | [`p1-3dgs/`](p1-3dgs/README.md) | baseline reproduced; 4 profile-driven optimizations: 1.72× end-to-end training on T&T train with the first 3 (3 seeds), test PSNR within 0.03 dB; heaviest-first tile order adds another 7%; with all 4, truck / playroom / drjohnson train 1.60× / 2.19× / 1.99× faster (1 seed), test PSNR within 0.07 dB |
| 2. Instant-NGP from scratch: hash encoding, fused WMMA MLP, volume rendering | [`p2-ingp/`](p2-ingp/README.md) | CUDA hash encoding (v1), ray marching (bitwise equal to the PyTorch reference) and fused training MLP: Lego 35.13 dB at 30k iterations in 256 s, 17.0× the reference per step |
| 3. Differentiable soft rasterizer for mesh fitting | `p3-diffrast/` | not started |

## Environment

Windows 11, Visual Studio 2022, CUDA 13.2, RTX 5070 Ti (sm_120, 16 GB), Python 3.12 via `uv`.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe ninja plyfile tqdm opencv-python joblib numpy wheel
git submodule update --init          # glm for p1-3dgs/rasterizer
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
git -C third_party/gaussian-splatting checkout 54c035f
git -C third_party/gaussian-splatting apply ../../patches/gaussian-splatting-train.patch
scripts\build_3dgs_ext.bat
# data: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -> data/
```

`third_party/`, `data/`, `outputs/` and `.venv/` are not tracked. The one local change to upstream
`train.py` is kept in [`patches/`](patches/): it lets the separate-SH build skip `torch.cat`.
