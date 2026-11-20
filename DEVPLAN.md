# Development plan

Neural rendering GPU kernel engineering, three projects over 10–12 weeks.
Stack: C++17, CUDA 13.2, PyTorch 2.14 (cu130), Windows 11 + MSVC 2022, RTX 5070 Ti (sm_120, 16 GB).

> The original plan said CUDA 11.8+. Blackwell (sm_120) needs CUDA ≥ 12.8, and the local nvcc is
> 13.2, so PyTorch here is the cu130 build. That keeps the extension builds on the same major version as torch.

## Project 1 — 3D Gaussian Splatting (weeks 1–4) · `p1-3dgs/`

Pipeline: preprocess (EWA projection to 2D) → tile binning (16×16) → CUB radix sort by (tile, depth)
→ per-tile front-to-back alpha blending (one block per tile) → backward.

| Phase | Work | Status |
|---|---|---|
| 1.1 | Build the official rasterizer on Windows/sm_120 and train a baseline on T&T train/truck and DB playroom/drjohnson | done: train 22.16 dB; truck 25.44, playroom 30.35, drjohnson 29.47 dB (upstream test PSNR) |
| 1.2 | Nsight Compute profile of `preprocessCUDA`, `renderCUDA`, `renderCUDA` backward, `duplicateWithKeys`; annotated source notes | done: nsys per-kernel breakdown (`kernel_breakdown.py`), ncu on `renderCUDA` fwd/bwd, tile-load histogram (`tile_load.py`); findings in the P1 README |
| 1.3 | 2–3 optimizations | done (4): smem feature cache, warp-reduced grad atomics, separate SH dc/rest (no `torch.cat`), heaviest-first tile order. Rejected: FP16 (pipes not saturated). Not done: splitting the heaviest tiles |
| 1.4 | Benchmark: PSNR within 0.1 dB of baseline; report FPS, it/s, peak memory | train scene: all4 fwd 259 FPS, fwd+bwd 1.07× over all3; all3 end-to-end training 1.72× (3 seeds/build), test PSNR −0.03 dB (within noise). all4 vs upstream, 1 seed: truck 1.60×, playroom 2.19×, drjohnson 1.99×, test PSNR within 0.07 dB |

## Project 2 — Instant-NGP from scratch (weeks 5–9) · `p2-ingp/`

- Multi-resolution hash encoding, forward and backward (backward uses atomicAdd)
- Fused MLP with Tensor Core WMMA, 64-wide layers
- Volume rendering, forward and backward
- Validation: compare against PyTorch references (tol ~1e-5) and run `gradcheck`
- Targets: Lego PSNR > 34 dB, 30k iters < 20 min, kernels within 2–3× of tiny-cuda-nn

## Project 3 — Differentiable rasterizer (weeks 10–11) · `p3-diffrast/`

- Soft rasterization (sigmoid coverage + softmax depth weighting), forward and backward in CUDA
- Demo: recover a sphere mesh from noisy vertices using multi-view silhouettes/images plus a Laplacian regularizer
- Optional: nvdiffrast-style edge gradients, texture optimization, PBR

## Week 12 — write-ups

Each project gets a README covering motivation, design, measured benchmarks, and lessons learned.
