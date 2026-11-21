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

void check(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, name, " must be contiguous float32 CUDA");
}

}  // namespace

torch::Tensor hash_encode_fwd(torch::Tensor x, torch::Tensor table, torch::Tensor res) {
  check(x, "x");
  check(table, "table");
  TORCH_CHECK(res.is_cuda() && res.scalar_type() == torch::kInt32, "res must be int32 CUDA");
  const int N = x.size(0), L = table.size(0), F = table.size(2);
  const uint32_t T = table.size(1);
  TORCH_CHECK((T & (T - 1)) == 0, "table size must be a power of two");
  auto out = torch::empty({N, L * F}, x.options());
  if (N == 0) return out;
  const dim3 block(128), grid((N + 127) / 128, L);
  auto s = c10::cuda::getCurrentCUDAStream();
  TORCH_CHECK(F == 2, "only F=2 is compiled");
  hash_fwd<2><<<grid, block, 0, s>>>(x.data_ptr<float>(), table.data_ptr<float>(), res.data_ptr<int>(), out.data_ptr<float>(), N, L, T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor hash_encode_bwd(torch::Tensor x, torch::Tensor grad_out, torch::Tensor res, int64_t T, int64_t F) {
  check(x, "x");
  check(grad_out, "grad_out");
  const int N = x.size(0), L = res.size(0);
  auto grad_table = torch::zeros({L, T, F}, x.options());
  if (N == 0) return grad_table;
  const dim3 block(128), grid((N + 127) / 128, L);
  auto s = c10::cuda::getCurrentCUDAStream();
  TORCH_CHECK(F == 2, "only F=2 is compiled");
  hash_bwd<2><<<grid, block, 0, s>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), res.data_ptr<int>(), grad_table.data_ptr<float>(), N, L, (uint32_t)T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_table;
}
