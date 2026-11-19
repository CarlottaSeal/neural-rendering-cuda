/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

// ---- Optimization switches (project 1, phase 1.3). All off reproduces upstream exactly. ----
// Override at build time: RAST_DEFINES="OPT_SHMEM_FEATURES=1" scripts\build_rasterizer.bat
//
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

#endif