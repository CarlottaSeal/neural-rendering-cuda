# Project 2 — Instant-NGP from scratch

Plan: multiresolution hash encoding, fused MLP and volume rendering as CUDA kernels. Each is checked against
a pure-PyTorch reference and against gradcheck before it is optimized.

## Layout

| Path | Contents |
|---|---|
| `ingp/encoding.py` | hash encoding (PyTorch reference, `backend="cuda"` switches to the kernel), SH degree-4 direction encoding |
| `ingp/model.py` | NGP field: hash grid → 1×64 density MLP → σ (trunc-exp) + 15 features; SH(dir) + features → 2×64 color MLP |
| `ingp/render.py` | 128³ occupancy grid, packed ray marching (dt = √3·3/1024), packed volume rendering with fp64 cumsum |
| `ingp/data.py` | NeRF-synthetic loader; images stay on the GPU as uint8 |
| `csrc/hash_encoding.cu` | CUDA hash encoding. v0: one thread per (point, level), level-major grid, scalar `atomicAdd`. v1 (default): level-fastest threads, paired x-corner loads, vector atomics |
| `csrc/march.cu` | CUDA ray marching v0: one thread per ray, count → scan → write packed samples; same rounding as the reference |
| `csrc/mlp.cu` | fused no-grad density MLP for the prune pass and the occupancy update: sigma only, weights in `__constant__` memory |
| `csrc/mlp_train.cu` | fused training MLP (density + color, fwd/bwd/weight grads), fp32 SIMT with shared-memory weights |
| `tests/test_mlp_train.py` | fused training MLP vs autograd (both against fp64), plus timing |
| `csrc/bindings.cpp` | pybind module for the kernels |
| `tests/test_march.py` | CUDA march vs reference (bitwise), plus timing |
| `tests/test_mlp.py` | fused density MLP vs the density net: sigma error, prune decisions, timing |
| `train_ref.py` | training loop: Adam 1e-2 (eps 1e-15), 100-step warm-up, ×0.33 at 50/75/90 %, about 2^18 samples per step, random background |
| `tests/test_ref.py` | reference vs naive loops + gradcheck |
| `tests/test_cuda.py` | CUDA kernel vs reference, plus timing |
| `profile_step.py` | steady-state step breakdown from a trained checkpoint: plain wall time, synced per-phase wall time, NVTX phases for nsys |

Data: `data/nerf_synthetic/lego` (100 train, 200 test, 800×800).

## Results so far

**Reference correctness** (`tests/test_ref.py`):

- The hash encoding matches a per-point, per-corner naive loop to within 3.3e-16 (fp64; the run covers both dense and hashed levels).
- Packed compositing matches a per-ray `cumprod` loop to within 2.4e-7, including a ray with no samples.
- gradcheck passes for both.

**PyTorch reference on Lego**, 30k iterations, seed 0:

| test PSNR (200 images) | train time | it/s | peak memory |
|---|---|---|---|
| 35.04 dB | 4226 s | 7.1 | 7786 MiB |

The timing is not clean. A game (Wingspan) was open during the run; its GPU use was about 2 % at idle.
The ray batch grows to about 47k rays per step once the occupancy grid is about 3.6 % full. The dense
march cost grows with it: iterations 500–1000 took 56 s, while later 5k-iteration blocks took 630–720 s.

**CUDA hash encoding v0 vs reference**: 2^18 points, 16 levels, F = 2, T = 2^19, median of 20 runs.

| implementation | fwd | fwd+bwd |
|---|---|---|
| PyTorch reference | 11.996 ms | 34.876 ms |
| CUDA v0 | 0.546 ms | 2.768 ms |

The forward pass agrees to 3.0e-7 (absolute). The backward pass agrees to 5.1e-7 (relative to max |grad|).

**Step breakdown** (`profile_step.py`, Lego checkpoint, idle GPU, 48 steps). The step runs in steady state: 48.9k rays,
1.10 M marched candidates and 262k samples after the transmittance cut, which matches the end of training. GPU time per
phase comes from nsys (`scripts\p2_nsys.bat`, then `p1-3dgs\kernel_breakdown.py`).

| encoding | wall ms/step | it/s | GPU kernel ms/step |
|---|---|---|---|
| PyTorch reference | 148.5 | 6.7 | 131.9 |
| CUDA v0 | 43.6 | 22.9 | 36.0 |

The clean reference rate is 6.7 it/s at steady state. The 7.1 it/s from training is a whole-run average, and it
includes the early steps, which use fewer rays.

| phase (GPU ms/step) | PyTorch encoding | CUDA encoding |
|---|---|---|
| march | 18.8 | 18.7 |
| prune (no-grad density on all marched candidates) | 67.0 | 7.3 |
| backward | 26.2 | 5.3 |
| optimizer | 1.6 | 1.6 |
| density + color MLP | 1.5 | 1.5 |
| occupancy grid update (amortized, every 16 steps) | 7.0 | 0.8 |
| compositing + loss | 0.1 | 0.1 |

With the CUDA encoding, the march is 52 % of GPU time (226 launches per step). It lays out every step across the
AABB, which is 27.5 M candidates per step (562 per ray on average), and keeps the 3.5 % that land in occupied cells.
It also runs in 4 chunks of 2^23, each with `.item()` syncs. The two MLPs take 1.5 ms together, so a fused MLP can
save at most that much. In the prune phase, 5.0 ms of the 7.3 ms is the hash forward on 1.1 M points.

**CUDA ray marching v0** (`tests/test_march.py`). Pass 1 counts the samples that land in occupied cells for each
ray. A cumsum gives the offsets, and its last element is the only host sync. Pass 2 repeats the steps and writes
packed (ray, t). Each float op uses an explicit rounding intrinsic (`__fmul_rn`, `__fadd_rn`, …) in the
reference's order, and the scalar division by dt is a multiply by the reciprocal, as torch does it. The jitter
comes from the same RNG call. So the output is compared with `torch.equal`, not a tolerance:

- random 30 % occupancy, 4096 rays (axis-parallel, starting inside the box, missing it), with and without jitter: equal
- Lego rays (48,872) against the trained grid (3.5 % occupied), 1.12 M samples: equal

| march, 48,872 rays | ms |
|---|---|
| PyTorch reference | 21.937 |
| CUDA v0 | 0.837 (26.2×) |

The full step with CUDA encoding and CUDA march takes 20.3 ms (49.2 it/s). The march phase is now 0.9 ms, and
the largest phases are prune (7.8 ms) and backward (6.1 ms).

**Lego, 30k iterations, seed 0, CUDA encoding + CUDA march** (same settings as the reference run):

| run | test PSNR (200 images) | train time | it/s | peak memory |
|---|---|---|---|---|
| PyTorch reference | 35.04 dB | 4226 s | 7.1 | 7786 MiB |
| CUDA encoding + march | 35.00 dB | 596 s | 50.4 | 4449 MiB |

This is one seed per run. The CUDA backward accumulates with float atomics, so runs are not bitwise
reproducible; the 0.04 dB gap is within the seed-to-seed spread seen in P1. The quick test PSNR (every 20th
image) at 5k / 10k / 20k / 30k is 34.05 / 34.57 / 35.40 / 35.50 dB, against 34.24 / 34.65 / 35.44 / 35.54 for
the reference. The train time ratio (7.1×) compares against the reference run, which had a game open. The
clean steady-state ratio is 148.5 / 20.3 = 7.3× per step. The 30k-in-under-20-minutes target is met (9.9 min).
The memory peak drops because the march no longer materialises every candidate.

**CUDA hash encoding v1** (`tests/test_cuda.py`). With the CUDA march in place, nsys put the encoding at about
8 of the step's 17.8 GPU ms. Most of it, 4.9 ms, was the prune-pass forward on 1.1 M points. v0 maps
`blockIdx.y` to the level. So a warp holds 32 points of one level, and each thread writes its 8 output bytes
128 bytes away from its neighbour's. v1 makes the level the fastest index: a warp is 2 points × 16 levels,
reads each point's x as a broadcast, and writes 256 contiguous bytes. The two corners that differ only in x
are loaded as one `float4` when their indices are an aligned pair. That holds for dense levels with an even
index and for hashed levels with even cx, because the x prime is 1. The backward uses `float4`/`float2`
`atomicAdd` (sm_90+) for those pairs. The corners are accumulated in v0's order. Forward error against the
reference is 3.0e-7 for both versions, and backward error is 3.9e-7 for v1 (5.6e-7 for v0).

| hash encoding | random 262k, fwd | random 262k, fwd+bwd | Lego marched 1.12 M, fwd | Lego marched 1.12 M, fwd+bwd |
|---|---|---|---|---|
| v0 | 0.561 ms | 2.744 ms | 5.039 ms | 13.749 ms |
| v1 | 0.449 ms | 2.444 ms | 0.901 ms | 4.536 ms |

The gain is small on random points, which are bound by the scattered table gathers. It is 5.6× on the real
prune input. Those points are coherent along rays, and the uncoalesced output writes were what limited v0
there. With v1, the full step takes 14.5 ms (68.8 it/s): prune drops from 7.8 to 3.7 ms and backward from
6.1 to 5.0 ms.

**Lego, 30k iterations, seed 0** (all runs with the same settings):

| run | test PSNR (200 images) | train time | it/s | peak memory |
|---|---|---|---|---|
| PyTorch reference | 35.04 dB | 4226 s | 7.1 | 7786 MiB |
| CUDA encoding v0 + march | 35.00 dB | 596 s | 50.4 | 4449 MiB |
| CUDA encoding v1 + march | 35.06 dB | 440 s | 68.1 | 4449 MiB |

The quick test PSNR for v1 at 5k / 10k / 20k / 30k is 34.21 / 34.69 / 35.38 / 35.48 dB.

**Fused no-grad density MLP and fused Adam.** With hash v1 the GPU time is 12.0 ms per step, and nsys
attributes about 6.3 ms of it to the two MLPs: the GEMMs, the ReLU write-backs (`clamp`/`threshold`) and
cuBLAS's `Kernel2`. The prune pass and the occupancy update need only sigma and no gradient, but the
reference runs the whole 32→64→16 net and writes a [1.1 M, 64] hidden tensor. `csrc/mlp.cu` computes
sigma = exp(relu(enc·W1ᵀ)·W2[0]) with one thread per point. The weights sit in `__constant__` memory, copied
device to device on the compute stream before each launch. All threads read the same weight, so the constant
cache broadcasts it, and the unrolled FFMAs take it as an operand. On the 1.12 M Lego prune points the max
relative sigma error is 9.1e-7. One of the 1.12 M keep/drop decisions differs (265,164 kept against
265,163). The density MLP goes from 1.828 ms to 0.218 ms.

The optimizer is the other large item: 1.9 ms for 16.8 M parameters with the default foreach Adam.
`torch.optim.Adam(fused=True)` is PyTorch's own fused kernel, not one written here.

| step (CUDA march + hash v1) | wall ms/step | it/s | prune | optimizer |
|---|---|---|---|---|
| baseline | 14.48 | 69.1 | 3.72 | 1.88 |
| + fused density MLP (`--fused_mlp`) | 12.57 | 79.6 | 2.06 | 1.86 |
| + `Adam(fused=True)` (`--fused_adam`) | 11.47 | 87.1 | 2.01 | 0.75 |

Lego 30k with both: 35.04 dB in 340 s (88.2 it/s). The quick test PSNR at 5k / 10k / 20k / 30k is 34.24 /
34.74 / 35.39 / 35.53 dB.

**Fused training MLP** (`csrc/mlp_train.cu`, `--fused_train_mlp`). It replaces the density net, the
feature/SH concat, the color net and the sigmoid, forward and backward, with three fp32 SIMT kernels.

- `mlp_fwd` runs one thread per point. All five weight matrices (37.9 KB) sit in shared memory, and every
  thread reads the same weight at the same time, so each read is a broadcast. SH is computed in the kernel.
  For the backward, it saves each layer's input (240 floats per point).
- `mlp_bwd` also runs one thread per point. It goes back through the layers with the ReLU masks, writes
  d(enc), and writes each layer's output gradient (212 floats per point).
- `mlp_wgrad` computes dW = Σ Gᵀ X for all five layers in one launch (`blockIdx.y` = layer). Each block
  takes 2048 points, stages 32-point tiles in shared memory, accumulates a 4×4 register tile per thread,
  and adds it to dW with one atomic per element per block.

`tests/test_mlp_train.py` runs 262k Lego points from the trained field with random directions and upstream
gradients. It compares both the fp32 autograd path and the fused kernels against fp64. The error is the max
error over the max |fp64 value|:

| | sigma | rgb | d enc | dWd1 | dWd2 | dWc1 | dWc2 | dWc3 |
|---|---|---|---|---|---|---|---|---|
| torch fp32 | 4.5e-7 | 8.4e-7 | 8.0e-7 | 5.9e-7 | 3.2e-7 | 6.4e-4 | 1.9e-3 | 8.1e-7 |
| fused | 6.3e-7 | 8.4e-7 | 1.2e-6 | 1.7e-6 | 6.4e-7 | 6.4e-4 | 1.9e-3 | 1.2e-6 |

The dWc1 and dWc2 errors are the same for both paths. They come from hidden units whose pre-activation is
so close to 0 that fp32 and fp64 put it on different sides of the ReLU. On those 262k points, fwd+bwd goes
from 4.94 ms to 2.19 ms (2.3×), and the no-grad forward from 1.71 to 0.45 ms. The three kernels take 0.57
(fwd), 0.96 (bwd) and 0.67 ms (wgrad). Each of them moves about 250–500 MB of saved activations and
gradients, so they are bound by memory bandwidth, not by math.

| step (CUDA march + hash v1, fused density, fused Adam) | wall ms/step | it/s | field fwd | backward |
|---|---|---|---|---|
| torch MLPs | 11.52 | 86.8 | 2.04 | 4.96 |
| fused training MLP | 8.71 | 114.8 | 0.71 | 3.38 |

Lego 30k: 35.13 dB in 256 s (117.0 it/s). The peak memory goes from 4449 to 5494 MiB. That is the saved
[N, 240] activations plus the [N, 212] gradient buffer, along with what the early large-batch steps keep alive.

## History

- The first 30k run ran out of memory at iteration ~2500. `march` materialised [rays × 1024] candidates,
  and the batch had grown to 41k rays. Candidates are now generated packed (only inside each ray's AABB
  interval) and in chunks of at most 2^23.

## Next

1. Cut the fused MLP's memory traffic: recompute activations in the backward instead of saving them, and
   reduce the weight gradients inside `mlp_bwd`, so the [N, 212] buffer is never written. WMMA for the layers
   comes after that, because the kernels are bandwidth-bound now.
2. Fuse the hash forward into the prune pass so the [1.1 M, 32] encoding is never written.
3. Fused compositing / transmittance kernels.
4. Comparison with tiny-cuda-nn.
