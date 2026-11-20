# Project 1 — 3D Gaussian Splatting rasterizer: study and optimization

The starting point is the official [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(commit `54c035f`), built on Windows 11 / MSVC 2022 / CUDA 13.2 / PyTorch 2.14 cu130 for an RTX 5070 Ti (sm_120).
`rasterizer/` is a copy of its `diff-gaussian-rasterization` submodule, with optimizations behind
compile-time switches (`cuda_rasterizer/opt_switches.h`). With every switch off, the code is upstream unchanged.

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
| `OPT_TILE_ORDER` | forward + backward `renderCUDA` | Launch tiles heaviest-first. After `identifyTileRanges`, the 2170 tiles are radix-sorted by range length, and block `b` renders tile `tile_order[b]` instead of tile `b`. The order is computed once in the forward and stored in the image buffer, so the backward reuses it. Chosen from the ncu profile below. |

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

`OPT_TILE_ORDER` was added after the ncu analysis. Same procedure: all3 and all4 were rebuilt back to back,
2 runs each, mean shown:

| variant | PSNR | fwd ms | FPS | fwd+bwd ms | vs all3 | peak MiB |
|---|---|---|---|---|---|---|
| all3 (rerun) | 22.157 | 4.15 | 241 | 11.88 | 1.00× | 3697 |
| all3 + `OPT_TILE_ORDER` (**all4**) | 22.157 | **3.86** | **259** | **11.15** | **1.07×** (fwd 1.08×) | 3709 |
| `OPT_TILE_ORDER` alone | 22.157 | 5.53 | 181 | 21.11 | – | 3910 |

Gradient check for all4: worst relative error 4.4e-3, with renders and depths bit-identical to upstream. This
all3 rerun is about 4% faster than the earlier all3 measurements (12.35–12.49 ms), which is drift
between sessions. Only same-session pairs are compared here. The extra 12 MiB holds the per-tile key,
permutation, and sort buffers.

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

The other three scenes were trained once each (seed 0), upstream against all four switches. Wall time is
measured the same way as in the 3-seed table:

| scene | upstream wall | all4 wall | speedup | test PSNR upstream → all4 | train PSNR upstream → all4 | final #Gaussians (all4) |
|---|---|---|---|---|---|---|
| T&T truck | 1061.8 s | 664.0 s | **1.60×** | 25.443 → 25.450 (+0.007) | 27.661 → 27.691 | 2,059,347 |
| DB playroom | 1447.9 s | 662.6 s | **2.19×** | 30.351 → 30.326 (−0.025) | 36.551 → 36.399 | 1,846,303 |
| DB drjohnson | 1978.8 s | 994.6 s | **1.99×** | 29.471 → 29.541 (+0.070) | 37.179 → 37.448 | 3,101,085 |

On every scene, test PSNR is within 0.07 dB of upstream. With one seed per scene, these differences are within the seed spread
measured on train. The speedup varies with the scene: the Deep Blending scenes gain more than truck. This
README does not break that difference down per kernel.

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

### Nsight Compute: `renderCUDA` with all three switches

One fwd+bwd step on test view 0, `--set full`. ncu locks clocks, so its durations run longer than nsys's.

| metric | forward | backward |
|---|---|---|
| duration (ncu) | 3.54 ms | 8.05 ms |
| compute / memory throughput | 44.6% / 22.7% | 50.1% / 50.1% (L1 72%, DRAM 4.8%) |
| registers/thread → theoretical occupancy | 39 → 100% | 60 → 66.7% |
| achieved occupancy | 87.8% | 62.5% |
| **SM active cycles min / avg / max** | 4.29M / 5.13M / **8.70M** | 11.87M / 12.94M / **18.21M** |
| **max / avg SM active** | **1.70×** | **1.41×** |
| top stalls (cycles per issued instruction) | not_selected 4.5, wait 2.7, barrier 1.8 | not_selected 2.2, barrier 2.0, wait 2.0, mio_throttle 1.6 |
| shared-memory bank conflicts | 0.13% of wavefronts | 0.09% of wavefronts |
| global reductions (atomics) | – | 38.3M requests |

Neither kernel is bound by bandwidth or math, and no single stall reason dominates. The clearest signal is
the gap between the busiest SM and the average SM. The kernel ends when its slowest SM finishes, and that
SM runs 41% (backward) or 70% (forward) longer than average. This is load imbalance across tiles.

`tile_load.py` reproduces the binning in PyTorch to count the Gaussians in each tile's range (view 0, 2170 tiles,
5.03M tile/Gaussian pairs):

| per-tile Gaussians | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| view 0 | 2,320 | 1,472 | 4,452 | 15,422 | **27,189 (11.7× mean)** |

The heaviest 13% of tiles (280) hold 43.7% of the work. The upstream kernel launches tiles in row-major
order, so whichever heavy tiles are launched last make up the tail. With 4 resident blocks per SM in the
backward, the heaviest tile alone is 1.51× the work of a perfectly balanced slot. That means reordering tiles
heaviest-first cannot fully close the gap, and the top tiles would also need to be split. These counts ignore
early termination, so they overstate saturated tiles. FP16 evaluation is not pursued, because the math and
memory pipes are not the limit.

**After `OPT_TILE_ORDER`** (same ncu setup):

| | SM active max / avg (fwd) | SM active max / avg (bwd) | bwd avg active / elapsed | bwd achieved occupancy |
|---|---|---|---|---|
| all3 | 8.70M / 5.13M = 1.70× | 18.21M / 12.94M = 1.41× | 69% | 62.5% |
| all4 | 7.34M / 4.84M = 1.52× | 14.80M / 12.49M = 1.19× | 85% | 65.6% |

The backward's slowest SM dropped from 18.2M to 14.8M cycles, and the average SM workload barely changed.
The gain comes from removing the tail, as intended. What remains matches the estimate above: in the forward,
the heaviest tile alone is 2.27× a balanced slot, so reordering cannot remove it. **Not done (yet):**
splitting the heaviest tiles across several blocks. That needs a per-tile partial blend plus a merge step in
the forward, and a matching split of the per-pixel backward state, which is much more invasive than a
launch-order permutation.

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
- The fourth optimization also came from the profile. Stall reasons were spread out and no pipe was
  saturated, so FP16 had nothing to fix. The SM active-cycle spread (`sm__cycles_active.max` vs `.avg`)
  pointed straight at load imbalance.
- Measure only back to back: the same build drifted ~4% between sessions, which is the same size as
  some of the gains.
- `config.h` defines `NUM_CHANNELS` as a macro, and a CUB template parameter has the same name. Including it
  before `<cub/cub.cuh>` breaks the build, so the switches live in their own `opt_switches.h`.
