/*
 * Optimization switches for project 1 (neural-rendering-cuda). Not part of upstream.
 *
 * Kept apart from config.h because config.h defines NUM_CHANNELS / BLOCK_X / BLOCK_Y as macros,
 * which break CUB headers (a CUB template parameter is also named NUM_CHANNELS) when config.h is
 * included before them. This header defines only OPT_* macros, so it is safe to include anywhere.
 *
 * All switches off reproduces upstream exactly.
 * Override at build time: scripts\build_rasterizer.bat opt "OPT_SHMEM_FEATURES=1 ..."
 */
#ifndef CUDA_RASTERIZER_OPT_SWITCHES_H_INCLUDED
#define CUDA_RASTERIZER_OPT_SWITCHES_H_INCLUDED

// OPT_SHMEM_FEATURES: stage per-Gaussian RGB (and depth) into shared memory alongside xy/conic
//   during the batch fetch, instead of a dependent global gather inside the per-pixel blend loop.
#ifndef OPT_SHMEM_FEATURES
#define OPT_SHMEM_FEATURES 0
#endif
// OPT_WARP_REDUCE_GRADS: in the backward blend kernel, sum each Gaussian's per-pixel gradient
//   contributions across the warp with shuffles, then issue one set of atomics per warp instead
//   of one per contributing pixel. Same math; only the floating-point summation order changes.
#ifndef OPT_WARP_REDUCE_GRADS
#define OPT_WARP_REDUCE_GRADS 0
#endif
// OPT_SEPARATE_SH: accept the SH DC term and the rest as two tensors (features_dc [P,1,3],
//   features_rest [P,M-1,3]) and return their gradients separately, so the renderer can skip the
//   torch.cat in GaussianModel.get_features (and the copy its backward implies). Exposed to Python
//   as diff_gaussian_rasterization.SEPARATE_SH; callers that pass no dc get the upstream path.
#ifndef OPT_SEPARATE_SH
#define OPT_SEPARATE_SH 0
#endif
// OPT_TILE_ORDER: launch render blocks heaviest tile first. After identifyTileRanges, the tiles are
//   sorted by range length (Gaussians binned to the tile), and block b renders tile tile_order[b]
//   instead of tile b. The order is computed once in the forward and reused by the backward. Tile
//   loads vary ~12x around the mean, and in row-major launch order the heavy tiles that happen
//   to run last form the tail.
#ifndef OPT_TILE_ORDER
#define OPT_TILE_ORDER 0
#endif

#endif
