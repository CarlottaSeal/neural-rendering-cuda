// Ray marching against the occupancy grid. v0: one thread per ray, the plain port of render.march.
// Pass 1 counts the samples in occupied cells per ray, an exclusive scan gives each ray its output offset, and
// pass 2 recomputes the same steps and writes packed (ray_idx, t), sorted by ray then t.
// Every float op is written with an explicit rounding intrinsic and in the reference's order (no FMA contraction),
// so the sample set matches render.march bit for bit.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {

struct Params {
  float lo[3], hi[3], ext[3];  // aabb min, max, max - min
  float dt, inv_dt, fres;
  int res;
};

__device__ __forceinline__ void ray_box(const float* o, const float* d, const Params& p, float& tmin, float& tmax) {
  tmin = -INFINITY;
  tmax = INFINITY;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float dk = fabsf(d[k]) < 1e-9f ? 1e-9f : d[k];
    const float inv = __frcp_rn(dk);
    const float t0 = __fmul_rn(__fsub_rn(p.lo[k], o[k]), inv);
    const float t1 = __fmul_rn(__fsub_rn(p.hi[k], o[k]), inv);
    tmin = fmaxf(tmin, fminf(t0, t1));
    tmax = fminf(tmax, fmaxf(t0, t1));
  }
  tmin = fmaxf(tmin, 0.f);
}

__device__ __forceinline__ bool occupied(const bool* __restrict__ occ, const float* o, const float* d, float t, const Params& p) {
  int c[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float x = __fadd_rn(o[k], __fmul_rn(t, d[k]));
    const float f = __fmul_rn(__fdiv_rn(__fsub_rn(x, p.lo[k]), p.ext[k]), p.fres);
    c[k] = min(max((int)f, 0), p.res - 1);  // (int) truncates toward zero, like .long()
  }
  return occ[(c[0] * p.res + c[1]) * p.res + c[2]];
}

template <bool WRITE>
__global__ void march_kernel(const float* __restrict__ rays_o, const float* __restrict__ rays_d, const float* __restrict__ u,
                             const bool* __restrict__ occ, Params p, int R, int* __restrict__ counts,
                             const int64_t* __restrict__ offsets, int64_t* __restrict__ out_ray, float* __restrict__ out_t) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float o[3] = {rays_o[r * 3], rays_o[r * 3 + 1], rays_o[r * 3 + 2]};
  const float d[3] = {rays_d[r * 3], rays_d[r * 3 + 1], rays_d[r * 3 + 2]};
  const float ur = u[r];
  float tmin, tmax;
  ray_box(o, d, p, tmin, tmax);
  // n = ceil((tmax - tmin) / dt - u); torch divides by a scalar as a multiply by its reciprocal
  const int n = max((int)ceilf(__fsub_rn(__fmul_rn(__fsub_rn(tmax, tmin), p.inv_dt), ur)), 0);
  int64_t w = WRITE ? offsets[r] : 0;
  int cnt = 0;
  for (int k = 0; k < n; ++k) {
    const float t = __fadd_rn(tmin, __fmul_rn(__fadd_rn((float)k, ur), p.dt));
    if (!occupied(occ, o, d, t, p)) continue;
    if (WRITE) {
      out_ray[w] = r;
      out_t[w] = t;
      ++w;
    } else {
      ++cnt;
    }
  }
  if (!WRITE) counts[r] = cnt;
}

}  // namespace

std::vector<torch::Tensor> march_rays(torch::Tensor o, torch::Tensor d, torch::Tensor u, torch::Tensor occ,
                                      std::vector<double> aabb, int64_t res, double dt) {
  for (auto* t : {&o, &d, &u})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32, "o, d, u must be contiguous float32 CUDA");
  TORCH_CHECK(occ.is_cuda() && occ.is_contiguous() && occ.scalar_type() == torch::kBool && occ.numel() == res * res * res,
              "occ must be a contiguous bool CUDA tensor of res^3 cells");
  TORCH_CHECK(aabb.size() == 6, "aabb must have 6 values");
  Params p;
  for (int k = 0; k < 3; ++k) {
    p.lo[k] = (float)aabb[k];
    p.hi[k] = (float)aabb[k + 3];
    p.ext[k] = (float)aabb[k + 3] - (float)aabb[k];
  }
  p.dt = (float)dt;
  p.inv_dt = 1.0f / (float)dt;
  p.fres = (float)res;
  p.res = (int)res;

  const int R = o.size(0);
  auto opt_i64 = o.options().dtype(torch::kInt64);
  if (R == 0) return {torch::empty({0}, opt_i64), torch::empty({0}, o.options())};
  auto s = c10::cuda::getCurrentCUDAStream();
  const int block = 128, grid = (R + block - 1) / block;
  auto counts = torch::empty({R}, o.options().dtype(torch::kInt32));
  march_kernel<false><<<grid, block, 0, s>>>(o.data_ptr<float>(), d.data_ptr<float>(), u.data_ptr<float>(), occ.data_ptr<bool>(),
                                             p, R, counts.data_ptr<int>(), nullptr, nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto ends = torch::cumsum(counts, 0, torch::kInt64);
  const int64_t total = ends[R - 1].item<int64_t>();  // the one host sync: the output size
  auto out_ray = torch::empty({total}, opt_i64);
  auto out_t = torch::empty({total}, o.options());
  if (total) {
    auto offsets = ends - counts;
    march_kernel<true><<<grid, block, 0, s>>>(o.data_ptr<float>(), d.data_ptr<float>(), u.data_ptr<float>(), occ.data_ptr<bool>(),
                                              p, R, nullptr, offsets.data_ptr<int64_t>(), out_ray.data_ptr<int64_t>(),
                                              out_t.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out_ray, out_t};
}
