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
| `csrc/hash_encoding.cu` | CUDA hash encoding v0: one thread per (point, level), backward with float `atomicAdd` |
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

## History

- The first 30k run ran out of memory at iteration ~2500. `march` materialised [rays × 1024] candidates,
  and the batch had grown to 41k rays. Candidates are now generated packed (only inside each ray's AABB
  interval) and in chunks of at most 2^23.

## Next

1. CUDA ray marching: one thread per ray, skip empty cells, count then scan then write packed samples.
2. Full training run with the CUDA encoding (PSNR unchanged, end-to-end time).
3. Prune cost (hash forward on all candidates), then compositing; the fused MLP moves down the list.
