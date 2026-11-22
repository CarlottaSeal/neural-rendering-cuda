// Fused density MLP for passes that need sigma only and no gradient (the prune pass, the occupancy update):
// sigma = exp(relu(enc @ W1^T) . W2[0]). The reference runs the whole 32->64->16 net and writes a [N,64] hidden
// tensor plus 16 outputs, 15 of which these passes drop.
// One thread per point, the weights in __constant__ memory. Every thread reads the same weight at the same
// time, so the constant cache broadcasts it, and with the loops unrolled the FFMAs take it as an operand.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int N_IN = 32, N_HID = 64;
__constant__ float cW1[N_HID * N_IN];  // nn.Linear layout: [out, in]
__constant__ float cW2[N_HID];       // row 0 of the output layer: the sigma logit

__global__ void density_fwd_kernel(const float* __restrict__ enc, float* __restrict__ sigma, int N) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float x[N_IN];
  const float4* row = reinterpret_cast<const float4*>(enc + (size_t)n * N_IN);
#pragma unroll
  for (int i = 0; i < N_IN / 4; ++i) {
    const float4 v = row[i];
    x[4 * i] = v.x;
    x[4 * i + 1] = v.y;
    x[4 * i + 2] = v.z;
    x[4 * i + 3] = v.w;
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < N_HID; ++j) {
    float h = 0.f;
#pragma unroll
    for (int i = 0; i < N_IN; ++i) h = fmaf(x[i], cW1[j * N_IN + i], h);
    s = fmaf(fmaxf(h, 0.f), cW2[j], s);
  }
  sigma[n] = expf(s);
}

}  // namespace

torch::Tensor density_mlp_fwd(torch::Tensor enc, torch::Tensor w1, torch::Tensor w2) {
  TORCH_CHECK(enc.is_cuda() && enc.is_contiguous() && enc.scalar_type() == torch::kFloat32 && enc.size(1) == N_IN,
              "enc must be contiguous float32 CUDA [N,32]");
  TORCH_CHECK(w1.is_cuda() && w1.is_contiguous() && w1.numel() == N_HID * N_IN, "w1 must be contiguous [64,32]");
  TORCH_CHECK(w2.is_cuda() && w2.size(1) == N_HID, "w2 must be [*,64]");
  const int N = enc.size(0);
  auto sigma = torch::empty({N}, enc.options());
  if (N == 0) return sigma;
  auto s = c10::cuda::getCurrentCUDAStream();
  auto w2row = w2[0].contiguous();
  // device-to-device copies on the compute stream, so they are ordered after the optimizer step that wrote them
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(cW1, w1.data_ptr<float>(), sizeof(float) * N_HID * N_IN, 0, cudaMemcpyDeviceToDevice, s));
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(cW2, w2row.data_ptr<float>(), sizeof(float) * N_HID, 0, cudaMemcpyDeviceToDevice, s));
  density_fwd_kernel<<<(N + 127) / 128, 128, 0, s>>>(enc.data_ptr<float>(), sigma.data_ptr<float>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sigma;
}
