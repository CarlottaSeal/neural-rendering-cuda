// Multiresolution hash encoding, forward and backward. v0: one thread per (point, level), the plain port of
// ingp/encoding.py. The layout matches the PyTorch reference: x [N,3] in [0,1], table [L,T,F], out [N,L*F].
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {

constexpr uint32_t P1 = 2654435761u, P2 = 805459861u;

template <int F>
struct Feat { float v[F]; };

__device__ __forceinline__ uint32_t grid_index(uint32_t cx, uint32_t cy, uint32_t cz, uint32_t res, uint32_t table_size) {
  const uint32_t s = res + 1;
  // dense if (res+1)^3 fits; compute in 64-bit to avoid overflow for fine levels
  if ((uint64_t)s * s * s <= table_size) return cx + cy * s + cz * s * s;
  return (cx ^ (cy * P1) ^ (cz * P2)) & (table_size - 1);  // uint32 wrap-around == reference's & (T-1)
}

template <int F>
__global__ void hash_fwd(const float* __restrict__ x, const float* __restrict__ table, const int* __restrict__ res,
                         float* __restrict__ out, int N, int L, uint32_t T) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = blockIdx.y;
  if (n >= N) return;
  const uint32_t R = res[l];
  float p[3], w[3];
  uint32_t c[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float xv = fminf(fmaxf(x[n * 3 + k], 0.f), 1.f - 1e-6f);
    p[k] = xv * R;
    const float f = floorf(p[k]);
    c[k] = (uint32_t)f;
    w[k] = p[k] - f;
  }
  const float* tl = table + (size_t)l * T * F;
  float acc[F] = {};
#pragma unroll
  for (int cor = 0; cor < 8; ++cor) {
    float wt = 1.f;
    uint32_t v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const bool hi = (cor >> k) & 1;
      v[k] = c[k] + hi;
      wt *= hi ? w[k] : 1.f - w[k];
    }
    const Feat<F> e = *reinterpret_cast<const Feat<F>*>(tl + (size_t)grid_index(v[0], v[1], v[2], R, T) * F);
#pragma unroll
    for (int f = 0; f < F; ++f) acc[f] += wt * e.v[f];
  }
#pragma unroll
  for (int f = 0; f < F; ++f) out[(size_t)n * L * F + l * F + f] = acc[f];
}

template <int F>
__global__ void hash_bwd(const float* __restrict__ x, const float* __restrict__ grad_out, const int* __restrict__ res,
                         float* __restrict__ grad_table, int N, int L, uint32_t T) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = blockIdx.y;
  if (n >= N) return;
  const uint32_t R = res[l];
  float w[3];
  uint32_t c[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float xv = fminf(fmaxf(x[n * 3 + k], 0.f), 1.f - 1e-6f);
    const float p = xv * R;
    const float f = floorf(p);
    c[k] = (uint32_t)f;
    w[k] = p - f;
  }
  float g[F];
#pragma unroll
  for (int f = 0; f < F; ++f) g[f] = grad_out[(size_t)n * L * F + l * F + f];
  float* gl = grad_table + (size_t)l * T * F;
#pragma unroll
  for (int cor = 0; cor < 8; ++cor) {
    float wt = 1.f;
    uint32_t v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const bool hi = (cor >> k) & 1;
      v[k] = c[k] + hi;
      wt *= hi ? w[k] : 1.f - w[k];
    }
    float* dst = gl + (size_t)grid_index(v[0], v[1], v[2], R, T) * F;
#pragma unroll
    for (int f = 0; f < F; ++f) atomicAdd(dst + f, wt * g[f]);
  }
}

// v1: the level varies fastest, so a warp covers 2 points x 16 levels. The point's x is a broadcast read and a
// warp's outputs are 256 contiguous bytes. The two x-neighbour corners are one float4 load when their indices form
// an aligned pair, which is always the case for dense levels with even index and for hashed levels with even cx
// (the x prime is 1, so cx and cx+1 differ only in bit 0). Corners are accumulated in the same order as v0.
__device__ __forceinline__ void corner_setup(const float* __restrict__ x, int n, uint32_t R, uint32_t c[3], float w[3]) {
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float xv = fminf(fmaxf(x[n * 3 + k], 0.f), 1.f - 1e-6f);
    const float p = xv * R;
    const float f = floorf(p);
    c[k] = (uint32_t)f;
    w[k] = p - f;
  }
}

template <int L>
__global__ void hash_fwd_v1(const float* __restrict__ x, const float* __restrict__ table, const int* __restrict__ res,
                            float* __restrict__ out, int N, uint32_t T) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = tid / L, l = tid % L;
  if (n >= N) return;
  const uint32_t R = res[l];
  uint32_t c[3];
  float w[3];
  corner_setup(x, n, R, c, w);
  const float2* tl = reinterpret_cast<const float2*>(table) + (size_t)l * T;
  float2 acc = make_float2(0.f, 0.f);
#pragma unroll
  for (int yz = 0; yz < 4; ++yz) {
    const uint32_t cy = c[1] + (yz & 1), cz = c[2] + (yz >> 1);
    const uint32_t i0 = grid_index(c[0], cy, cz, R, T), i1 = grid_index(c[0] + 1, cy, cz, R, T);
    float2 e0, e1;
    if ((i0 ^ i1) == 1) {
      const float4 q = *reinterpret_cast<const float4*>(tl + (i0 & ~1u));
      e0 = (i0 & 1) ? make_float2(q.z, q.w) : make_float2(q.x, q.y);
      e1 = (i0 & 1) ? make_float2(q.x, q.y) : make_float2(q.z, q.w);
    } else {
      e0 = tl[i0];
      e1 = tl[i1];
    }
    // same products as v0: w_x * w_y * w_z, x first
    const float w0 = (1.f - w[0]) * ((yz & 1) ? w[1] : 1.f - w[1]) * ((yz >> 1) ? w[2] : 1.f - w[2]);
    const float w1 = w[0] * ((yz & 1) ? w[1] : 1.f - w[1]) * ((yz >> 1) ? w[2] : 1.f - w[2]);
    acc.x += w0 * e0.x;
    acc.y += w0 * e0.y;
    acc.x += w1 * e1.x;
    acc.y += w1 * e1.y;
  }
  reinterpret_cast<float2*>(out)[(size_t)n * L + l] = acc;
}

template <int L>
__global__ void hash_bwd_v1(const float* __restrict__ x, const float* __restrict__ grad_out, const int* __restrict__ res,
                            float* __restrict__ grad_table, int N, uint32_t T) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = tid / L, l = tid % L;
  if (n >= N) return;
  const uint32_t R = res[l];
  uint32_t c[3];
  float w[3];
  corner_setup(x, n, R, c, w);
  const float2 g = reinterpret_cast<const float2*>(grad_out)[(size_t)n * L + l];
  float2* gl = reinterpret_cast<float2*>(grad_table) + (size_t)l * T;
#pragma unroll
  for (int yz = 0; yz < 4; ++yz) {
    const uint32_t cy = c[1] + (yz & 1), cz = c[2] + (yz >> 1);
    const float w0 = (1.f - w[0]) * ((yz & 1) ? w[1] : 1.f - w[1]) * ((yz >> 1) ? w[2] : 1.f - w[2]);
    const float w1 = w[0] * ((yz & 1) ? w[1] : 1.f - w[1]) * ((yz >> 1) ? w[2] : 1.f - w[2]);
    const uint32_t i0 = grid_index(c[0], cy, cz, R, T), i1 = grid_index(c[0] + 1, cy, cz, R, T);
    const float2 g0 = make_float2(w0 * g.x, w0 * g.y), g1 = make_float2(w1 * g.x, w1 * g.y);
#if __CUDA_ARCH__ >= 900
    if ((i0 ^ i1) == 1) {
      const float4 q = (i0 & 1) ? make_float4(g1.x, g1.y, g0.x, g0.y) : make_float4(g0.x, g0.y, g1.x, g1.y);
      atomicAdd(reinterpret_cast<float4*>(gl + (i0 & ~1u)), q);
    } else {
      atomicAdd(gl + i0, g0);
      atomicAdd(gl + i1, g1);
    }
#else
    atomicAdd(&gl[i0].x, g0.x);
    atomicAdd(&gl[i0].y, g0.y);
    atomicAdd(&gl[i1].x, g1.x);
    atomicAdd(&gl[i1].y, g1.y);
#endif
  }
}

void check(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, name, " must be contiguous float32 CUDA");
}

}  // namespace

torch::Tensor hash_encode_fwd(torch::Tensor x, torch::Tensor table, torch::Tensor res, int64_t version) {
  check(x, "x");
  check(table, "table");
  TORCH_CHECK(res.is_cuda() && res.scalar_type() == torch::kInt32, "res must be int32 CUDA");
  const int N = x.size(0), L = table.size(0), F = table.size(2);
  const uint32_t T = table.size(1);
  TORCH_CHECK((T & (T - 1)) == 0, "table size must be a power of two");
  auto out = torch::empty({N, L * F}, x.options());
  if (N == 0) return out;
  auto s = c10::cuda::getCurrentCUDAStream();
  TORCH_CHECK(F == 2, "only F=2 is compiled");
  if (version == 1) {
    TORCH_CHECK(L == 16, "v1 is compiled for 16 levels");
    const int64_t threads = (int64_t)N * 16;
    hash_fwd_v1<16><<<(threads + 127) / 128, 128, 0, s>>>(x.data_ptr<float>(), table.data_ptr<float>(), res.data_ptr<int>(), out.data_ptr<float>(), N, T);
  } else {
    const dim3 block(128), grid((N + 127) / 128, L);
    hash_fwd<2><<<grid, block, 0, s>>>(x.data_ptr<float>(), table.data_ptr<float>(), res.data_ptr<int>(), out.data_ptr<float>(), N, L, T);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor hash_encode_bwd(torch::Tensor x, torch::Tensor grad_out, torch::Tensor res, int64_t T, int64_t F, int64_t version) {
  check(x, "x");
  check(grad_out, "grad_out");
  const int N = x.size(0), L = res.size(0);
  auto grad_table = torch::zeros({L, T, F}, x.options());
  if (N == 0) return grad_table;
  auto s = c10::cuda::getCurrentCUDAStream();
  TORCH_CHECK(F == 2, "only F=2 is compiled");
  if (version == 1) {
    TORCH_CHECK(L == 16, "v1 is compiled for 16 levels");
    const int64_t threads = (int64_t)N * 16;
    hash_bwd_v1<16><<<(threads + 127) / 128, 128, 0, s>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), res.data_ptr<int>(), grad_table.data_ptr<float>(), N, (uint32_t)T);
  } else {
    const dim3 block(128), grid((N + 127) / 128, L);
    hash_bwd<2><<<grid, block, 0, s>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), res.data_ptr<int>(), grad_table.data_ptr<float>(), N, L, (uint32_t)T);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_table;
}
