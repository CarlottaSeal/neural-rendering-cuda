# Project 1 — 3D Gaussian Splatting rasterizer: study and optimization

The starting point is the official [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(commit `54c035f`), built on Windows 11 / MSVC 2022 / CUDA 13.2 / PyTorch 2.14 cu130 for an RTX 5070 Ti (sm_120).
`rasterizer/` is a copy of its `diff-gaussian-rasterization` submodule, with optimizations behind
compile-time switches (`cuda_rasterizer/config.h`). With every switch off, the code is upstream unchanged.

Upstream code is under the Inria non-commercial research license (`rasterizer/LICENSE.md`).

## Pipeline (forward)

```
preprocessCUDA (1 thread / Gaussian)      EWA projection: 3D cov -> 2D conic, SH -> RGB, radius, tiles touched
   │ InclusiveSum(tiles_touched)          offsets for duplication
duplicateWithKeys                          one (tile_id << 32 | depth) key per Gaussian x touched 16x16 tile
   │ cub::DeviceRadixSort::SortPairs       sort by tile, then depth
identifyTileRanges                         [start,end) of each tile in the sorted list
renderCUDA (1 block / tile, 1 thread / px) batch-fetch 256 Gaussians to smem, front-to-back alpha blend,
                                           block-wide early exit when every pixel has T < 1e-4
```

The backward runs `renderCUDA` back to front: it recovers T from `final_T` and atomically accumulates
gradients for color, mean2D, conic, and opacity. Then `preprocessCUDA`/`computeCov2DCUDA` propagate those
gradients to 3D mean, scale, rotation, and SH.

## Optimizations

| Switch | Kernel | Idea |
|---|---|---|
| `OPT_SHMEM_FEATURES` | forward `renderCUDA` | Stage RGB and 1/depth into shared memory during the batch fetch. Upstream instead does a dependent global gather `features[collected_id[j]*3+ch]` inside the per-pixel loop. (The backward already does this; the forward did not.) |
| `OPT_WARP_REDUCE_GRADS` | backward `renderCUDA` | All 32 lanes of a warp process the same Gaussian `j` at the same time. Keep them in lockstep, put each lane's gradient contribution in registers, `__shfl_down` reduce the 9–10 values, and have one lane issue the atomics. Upstream issues up to 32× that many same-address atomics. When only one lane contributes, the shuffles are skipped. |
| `OPT_SEPARATE_SH` | `preprocessCUDA` fwd/bwd, bindings | Take `features_dc` [P,1,3] and `features_rest` [P,15,3] as two inputs and return two gradients. Upstream's renderer calls `GaussianModel.get_features`, a `torch.cat` that copies all 48 floats per Gaussian (≈210 MB at 1.1M Gaussians) on every render. The SH code addresses coefficients through one pointer helper (`shPointers`, `auxiliary.h`), so every `sh[k]` / `dL_dsh[k]` line stays as upstream wrote it. Python sees `diff_gaussian_rasterization.SEPARATE_SH`. |

## Results

`train` scene (Tanks&Temples), 30k iterations, `--eval` split (38 test views), 1,096,280 Gaussians, 980×545.
Each variant renders the **same trained model** (`bench.py`). The installed build reports whether it takes dc/rest
separately, and the scripts pass `separate_sh` to match. Times are the median over 38 views × 5 repeats,
measured with CUDA events around the full `render()` call (including Python and preprocess).

| variant | PSNR | fwd ms | FPS | fwd+bwd ms | fwd+bwd speedup | peak MiB |
|---|---|---|---|---|---|---|
| upstream | 22.157 | 6.17 | 162 | 22.69 | 1.00× | 3897 |
| opt, all switches off | 22.157 | 6.27 | 160 | 22.70 | 1.00× | 3897 |
| `OPT_SHMEM_FEATURES` | 22.157 | 5.91 | 169 | 22.67 | 1.00× | 3897 |
| `OPT_WARP_REDUCE_GRADS` | 22.157 | 6.27 | 160 | 14.28 | 1.59× | 3897 |
| both | 22.157 | 5.92 | 169 | 13.92 | **1.63×** | 3897 |

`OPT_SEPARATE_SH` was added later. To keep the comparison fair, `both` and `both + OPT_SEPARATE_SH` were rebuilt
back to back and benchmarked twice each in the same session (the table shows the mean of the two runs):

| variant | PSNR | fwd ms | FPS | fwd+bwd ms | vs upstream | peak MiB |
|---|---|---|---|---|---|---|
| both (rerun) | 22.157 | 5.86 | 171 | 13.76 | 1.65× | 3897 |
| both + `OPT_SEPARATE_SH` | 22.157 | **4.36** | **229** | **12.39** | **1.83×** | **3697** |

The forward gain (−1.50 ms) is slightly larger than the `CatArrayBatchedCopy` kernel time (1.39–1.42 ms/step
in nsys). The rest is plausibly the 210 MB allocation that no longer happens per render, but that has not been measured separately.

**Gradient check** (`dump_grads.py`, 4 views, L1 + inverse-depth loss): max |Δgrad| / max |grad|
against the upstream dump. For reference, two upstream runs differ from each other by 3.9e-3,
because float atomics accumulate in a nondeterministic order.

| variant | worst relative error vs upstream |
|---|---|
| upstream rerun | 3.9e-3 |
| opt, all off | 4.7e-3 |
| `OPT_SHMEM_FEATURES` | 3.5e-3 |
| `OPT_WARP_REDUCE_GRADS` | 5.2e-3 |
| both | 2.4e-3 |
| both + `OPT_SEPARATE_SH` | 4.4e-3 (renders and depths bit-identical) |

All variants are within upstream's own run-to-run noise.

**End-to-end training**: a full 30k-iteration `train.py` run per build, one run each.

| build | wall time | speedup | test PSNR | train PSNR | final #Gaussians |
|---|---|---|---|---|---|
| upstream | 924.7 s | 1.00× | 22.157 | 27.135 | 1,096,280 |
| both switches | 573.7 s | **1.61×** | 22.212 (+0.055) | 26.799 (−0.336) | 1,089,628 |

One run per build cannot tell whether the 0.34 dB lower train-view PSNR is a real effect, so both builds
were retrained with **3 seeds each** (`scripts\train_seeds.ps1`, `train_seed.py`, mean ± std).
This time all three switches are on. Wall time here covers only `train.py` itself, without Python startup
and imports, so it is not comparable to the single-run table above:

| build | seeds | wall time | speedup | test PSNR | train PSNR | final #Gaussians |
|---|---|---|---|---|---|---|
| upstream | 0, 1, 2 | 886.0 ± 26.5 s | 1.00× | 22.207 ± 0.055 | 26.565 ± 0.711 | 1,092,202 ± 4,929 |
| all 3 switches | 0, 1, 2 | 514.9 ± 7.8 s | **1.72×** | 22.182 ± 0.055 | 26.480 ± 0.852 | 1,092,647 ± 4,940 |

The seed-to-seed spread in train PSNR (0.7–0.85 dB std) is much larger than the gap between the build means
(0.085 dB). The earlier −0.34 dB was therefore run-to-run noise, not an effect of the optimizations.
Test PSNR is within 0.03 dB, which meets the ≤ 0.1 dB criterion.

## Profile (Nsight Systems)

`profile_step.py` runs 5 NVTX-marked fwd+bwd steps on one test view, and `kernel_breakdown.py` sums the GPU
time of each kernel inside those ranges. Backward kernels are matched by time, not by thread, because
autograd launches them from its own worker thread. ms per step:

| kernel | upstream | both | both + `OPT_SEPARATE_SH` |
|---|---|---|---|
| `renderCUDA` backward | 16.16 | 6.33 | 6.77 |
| `renderCUDA` forward | 4.09 | 3.14 | 3.29 |
| `CatArrayBatchedCopy` (`get_features`) | 1.39 | 1.42 | — |
| `preprocessCUDA` backward | 1.17 | 0.98 | 0.95 |
| CUB radix sort (7 launches) | 1.04 | 0.93 | 0.92 |
| `CUDAFunctor_add` (backward) | 0.82 | 0.83 | 0.93 |
| **total GPU kernel time** | **26.63** | **15.64** | **14.77** |

Each column comes from a separate 5-step capture. The `renderCUDA` code is the same in the last two columns,
so their 0.15–0.45 ms difference is run-to-run variation. After these changes, backward `renderCUDA` is still
46% of GPU time and is the next target.

## Reproduce

```bat
scripts\build_3dgs_ext.bat                         :: upstream rasterizer + simple-knn + fused-ssim
cd third_party\gaussian-splatting
..\..\.venv\Scripts\python.exe train.py -s ..\..\data\tandt\train -m ..\..\outputs\baseline\train --eval --disable_viewer
cd ..\..
pwsh scripts\sweep_variants.ps1                    :: build each variant -> bench + grad check -> table
pwsh scripts\train_seeds.ps1                       :: 30k-iteration training, upstream vs all3, 3 seeds each
```

## Notes / lessons

- CUDA 13's CCCL rejects MSVC's traditional preprocessor. Setting `CL=/Zc:preprocessor` fixes it for
  both cl.exe and nvcc's host passes without patching any setup.py.
- PyTorch has to match nvcc's CUDA **major** version for extension builds, so this project uses cu130 wheels.
- A warp-level early exit idea in the forward turned out to be a no-op: the loop condition `!done && …`
  already takes saturated lanes out of the loop, so there is no wasted spinning to remove.
- The third optimization was chosen from the profile, not from the kernel list in the plan: a plain PyTorch
  `torch.cat` was 9% of step time, more than the whole forward `preprocessCUDA`.
- `third_party/gaussian-splatting/train.py` has one local patch: it sets `separate_sh` from
  `diff_gaussian_rasterization.SEPARATE_SH` instead of from whether `SparseGaussianAdam` imports. With the
  upstream rasterizer installed, the attribute does not exist, so its behaviour is unchanged.
- Next: Nsight Compute on backward `renderCUDA` (stall reasons, atomics throughput) to decide between FP16
  evaluation and load balancing.
