// Fused NGP MLP for training, fp32 SIMT, forward and backward.
//   density: enc[32] -> relu(Wd1) [64] -> Wd2 [16];  sigma = exp(hd[0])
//   color:   c = [hd[1..15], SH4(dir)] [31] -> relu(Wc1) [64] -> relu(Wc2) [64] -> Wc3 [3];  rgb = sigmoid
// mlp_fwd: one thread per point, all five weight matrices in shared memory (all threads read the same weight,
//   so it is a broadcast). Writes sigma, rgb and, for the backward, the layer inputs (act, 240 floats per point).
// mlp_bwd: one thread per point, back through the layers. Writes d(enc) and the per-layer output gradients
//   (grads, 212 floats per point).
// mlp_wgrad: dW = sum_p G[p]^T X[p] for the five layers, split over chunks of points; each block keeps a 64x64
//   tile in registers (4x4 per thread) and adds it to dW with atomics once per chunk.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

// shared-memory weight layout (floats); Wc1 is padded to 32 inputs and Wc3 to 4 rows, both with zeros
constexpr int oWd1 = 0, oWd2 = oWd1 + 64 * 32, oWc1 = oWd2 + 16 * 64, oWc2 = oWc1 + 64 * 32, oWc3 = oWc2 + 64 * 64;
constexpr int W_FLOATS = oWc3 + 4 * 64;
// per-point layer inputs: h1 | hd | c | a1 | a2
constexpr int aH1 = 0, aHd = 64, aC = 80, aA1 = 112, aA2 = 176, ACT = 240;
// per-point output gradients: go | gA2 | gA1 | gHd | gH1
constexpr int gGo = 0, gA2 = 4, gA1 = 68, gHd = 132, gH1 = 148, GRD = 212;

struct Weights {
  const float *wd1, *wd2, *wc1, *wc2, *wc3;
};

__device__ void load_weights(float* Ws, const Weights& w) {
  for (int k = threadIdx.x; k < W_FLOATS; k += blockDim.x) {
    float v;
    if (k < oWd2) v = w.wd1[k];
    else if (k < oWc1) v = w.wd2[k - oWd2];
    else if (k < oWc2) { const int r = (k - oWc1) / 32, c = (k - oWc1) % 32; v = c < 31 ? w.wc1[r * 31 + c] : 0.f; }
    else if (k < oWc3) v = w.wc2[k - oWc2];
    else { const int r = (k - oWc3) / 64; v = r < 3 ? w.wc3[k - oWc3] : 0.f; }
    Ws[k] = v;
  }
}

template <int K>
__device__ __forceinline__ float dot(const float* a, const float* w) {
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < K / 4; ++k) {
    const float4 q = w4[k];
    s = fmaf(a[4 * k], q.x, s);
    s = fmaf(a[4 * k + 1], q.y, s);
    s = fmaf(a[4 * k + 2], q.z, s);
    s = fmaf(a[4 * k + 3], q.w, s);
  }
  return s;
}

// acc[0..K) += g * w[0..K)
template <int K>
__device__ __forceinline__ void axpy(float* acc, float g, const float* w) {
  const float4* w4 = reinterpret_cast<const float4*>(w);
#pragma unroll
  for (int k = 0; k < K / 4; ++k) {
    const float4 q = w4[k];
    acc[4 * k] = fmaf(g, q.x, acc[4 * k]);
    acc[4 * k + 1] = fmaf(g, q.y, acc[4 * k + 1]);
    acc[4 * k + 2] = fmaf(g, q.z, acc[4 * k + 2]);
    acc[4 * k + 3] = fmaf(g, q.w, acc[4 * k + 3]);
  }
}

template <int K>
__device__ __forceinline__ void load_row(float* r, const float* src) {
  const float4* s4 = reinterpret_cast<const float4*>(src);
#pragma unroll
  for (int k = 0; k < K / 4; ++k) {
    const float4 q = s4[k];
    r[4 * k] = q.x;
    r[4 * k + 1] = q.y;
    r[4 * k + 2] = q.z;
    r[4 * k + 3] = q.w;
  }
}

template <int K>
__device__ __forceinline__ void store_row(float* dst, const float* r) {
  float4* d4 = reinterpret_cast<float4*>(dst);
#pragma unroll
  for (int k = 0; k < K / 4; ++k) d4[k] = make_float4(r[4 * k], r[4 * k + 1], r[4 * k + 2], r[4 * k + 3]);
}

// real SH up to degree 3, same constants and order as encoding.sh_encoding_deg4
__device__ __forceinline__ void sh4(float x, float y, float z, float* o) {
  const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
  o[0] = 0.28209479177387814f;
  o[1] = -0.48860251190291987f * y;
  o[2] = 0.48860251190291987f * z;
  o[3] = -0.48860251190291987f * x;
  o[4] = 1.0925484305920792f * xy;
  o[5] = -1.0925484305920792f * yz;
  o[6] = 0.94617469575755997f * zz - 0.31539156525251999f;
  o[7] = -1.0925484305920792f * xz;
  o[8] = 0.54627421529603959f * xx - 0.54627421529603959f * yy;
  o[9] = 0.59004358992664352f * y * (-3.0f * xx + yy);
  o[10] = 2.8906114426405538f * xy * z;
  o[11] = 0.45704579946446572f * y * (1.0f - 5.0f * zz);
  o[12] = 0.3731763325901154f * z * (5.0f * zz - 3.0f);
  o[13] = 0.45704579946446572f * x * (1.0f - 5.0f * zz);
  o[14] = 1.4453057213202769f * z * (xx - yy);
  o[15] = 0.59004358992664352f * x * (-xx + 3.0f * yy);
}

__global__ void __launch_bounds__(128) mlp_fwd_kernel(const float* __restrict__ enc, const float* __restrict__ dirs,
                                                      Weights w, float* __restrict__ sigma, float* __restrict__ rgb,
                                                      float* __restrict__ act, int N) {
  extern __shared__ float4 smem4[];
  float* Ws = reinterpret_cast<float*>(smem4);
  load_weights(Ws, w);
  __syncthreads();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float* arow = act ? act + (size_t)n * ACT : nullptr;

  float x[32], h1[64], hd[16];
  load_row<32>(x, enc + (size_t)n * 32);
#pragma unroll
  for (int j = 0; j < 64; ++j) h1[j] = fmaxf(dot<32>(x, Ws + oWd1 + j * 32), 0.f);
#pragma unroll
  for (int j = 0; j < 16; ++j) hd[j] = dot<64>(h1, Ws + oWd2 + j * 64);
  if (arow) {
    store_row<64>(arow + aH1, h1);
    store_row<16>(arow + aHd, hd);
  }
  sigma[n] = expf(hd[0]);

  float c[32], a1[64];
#pragma unroll
  for (int i = 0; i < 15; ++i) c[i] = hd[i + 1];
  sh4(dirs[n * 3], dirs[n * 3 + 1], dirs[n * 3 + 2], c + 15);
  c[31] = 0.f;
  if (arow) store_row<32>(arow + aC, c);
#pragma unroll
  for (int j = 0; j < 64; ++j) a1[j] = fmaxf(dot<32>(c, Ws + oWc1 + j * 32), 0.f);
  if (arow) store_row<64>(arow + aA1, a1);
  float a2[64];
#pragma unroll
  for (int j = 0; j < 64; ++j) a2[j] = fmaxf(dot<64>(a1, Ws + oWc2 + j * 64), 0.f);
  if (arow) store_row<64>(arow + aA2, a2);
#pragma unroll
  for (int k = 0; k < 3; ++k) rgb[n * 3 + k] = 1.f / (1.f + expf(-dot<64>(a2, Ws + oWc3 + k * 64)));
}

__global__ void __launch_bounds__(128) mlp_bwd_kernel(const float* __restrict__ act, const float* __restrict__ rgb,
                                                      const float* __restrict__ g_sigma, const float* __restrict__ g_rgb,
                                                      Weights w, float* __restrict__ g_enc, float* __restrict__ grads, int N) {
  extern __shared__ float4 smem4[];
  float* Ws = reinterpret_cast<float*>(smem4);
  load_weights(Ws, w);
  __syncthreads();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* arow = act + (size_t)n * ACT;
  float* grow = grads + (size_t)n * GRD;

  // sigmoid backward, then layer c3: gA2 = mask(a2) * Wc3^T go
  float go[4];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float y = rgb[n * 3 + k];
    go[k] = g_rgb[n * 3 + k] * (y * (1.f - y));
  }
  go[3] = 0.f;
  store_row<4>(grow + gGo, go);
  float ga2[64];
#pragma unroll
  for (int j = 0; j < 64; ++j) ga2[j] = 0.f;
#pragma unroll
  for (int k = 0; k < 3; ++k) axpy<64>(ga2, go[k], Ws + oWc3 + k * 64);
  {
    float a2[64];
    load_row<64>(a2, arow + aA2);
#pragma unroll
    for (int j = 0; j < 64; ++j) ga2[j] = a2[j] > 0.f ? ga2[j] : 0.f;
  }
  store_row<64>(grow + gA2, ga2);

  // layer c2: gA1 = mask(a1) * Wc2^T gA2
  float ga1[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) ga1[i] = 0.f;
#pragma unroll
  for (int j = 0; j < 64; ++j) axpy<64>(ga1, ga2[j], Ws + oWc2 + j * 64);
  {
    float a1[64];
    load_row<64>(a1, arow + aA1);
#pragma unroll
    for (int i = 0; i < 64; ++i) ga1[i] = a1[i] > 0.f ? ga1[i] : 0.f;
  }
  store_row<64>(grow + gA1, ga1);

  // layer c1: gradient w.r.t. the 15 density features (the SH part has none); first 16 columns of Wc1
  float gc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) gc[i] = 0.f;
#pragma unroll
  for (int j = 0; j < 64; ++j) axpy<16>(gc, ga1[j], Ws + oWc1 + j * 32);

  // density output: gHd = [d sigma * exp(min(hd0, 15)), gc[0..14]] (TruncExp)
  float ghd[16];
  ghd[0] = g_sigma[n] * expf(fminf(arow[aHd], 15.f));
#pragma unroll
  for (int i = 0; i < 15; ++i) ghd[i + 1] = gc[i];
  store_row<16>(grow + gHd, ghd);

  // layer d2: gH1 = mask(h1) * Wd2^T gHd
  float gh1[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) gh1[i] = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) axpy<64>(gh1, ghd[j], Ws + oWd2 + j * 64);
  {
    float h1[64];
    load_row<64>(h1, arow + aH1);
#pragma unroll
    for (int i = 0; i < 64; ++i) gh1[i] = h1[i] > 0.f ? gh1[i] : 0.f;
  }
  store_row<64>(grow + gH1, gh1);

  // layer d1: d enc = Wd1^T gH1
  float gx[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) gx[i] = 0.f;
#pragma unroll
  for (int j = 0; j < 64; ++j) axpy<32>(gx, gh1[j], Ws + oWd1 + j * 32);
  store_row<32>(g_enc + (size_t)n * 32, gx);
}

struct WLayer {
  const float* G;  // output gradient, row stride gs, the layer's columns start at G
  const float* X;  // layer input, row stride xs
  float* dW;       // [O, I] row-major
  int gs, xs, O, I;
};
struct WLayers {
  WLayer l[5];
};

constexpr int WG_SUB = 32;  // points per shared-memory sub-tile

__global__ void __launch_bounds__(256) mlp_wgrad_kernel(WLayers L, int N, int chunk) {
  __shared__ __align__(16) float Gs[WG_SUB][64];
  __shared__ __align__(16) float Xs[WG_SUB][64];
  const WLayer ly = L.l[blockIdx.y];
  const int t = threadIdx.x;
  const int o0 = (t / 16) * 4, i0 = (t % 16) * 4;
  const int Gw = (ly.O + 3) & ~3, Xw = (ly.I + 3) & ~3;  // padded widths actually loaded
  const bool active = o0 < Gw && i0 < Xw;
  float acc[4][4] = {};
  const int p_begin = blockIdx.x * chunk, p_end = min(N, p_begin + chunk);
  for (int p0 = p_begin; p0 < p_end; p0 += WG_SUB) {
    const int np = min(WG_SUB, p_end - p0);
    __syncthreads();
    for (int k = t; k < WG_SUB * 16; k += blockDim.x) {
      const int p = k / 16, c = (k % 16) * 4;
      float4 g = make_float4(0.f, 0.f, 0.f, 0.f), x = g;
      if (p < np) {
        if (c < Gw) g = *reinterpret_cast<const float4*>(ly.G + (size_t)(p0 + p) * ly.gs + c);
        if (c < Xw) x = *reinterpret_cast<const float4*>(ly.X + (size_t)(p0 + p) * ly.xs + c);
      }
      *reinterpret_cast<float4*>(&Gs[p][c]) = g;
      *reinterpret_cast<float4*>(&Xs[p][c]) = x;
    }
    __syncthreads();
    if (active) {
#pragma unroll 8
      for (int p = 0; p < WG_SUB; ++p) {
        const float4 g = *reinterpret_cast<const float4*>(&Gs[p][o0]);
        const float4 x = *reinterpret_cast<const float4*>(&Xs[p][i0]);
        const float gv[4] = {g.x, g.y, g.z, g.w}, xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
          for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(gv[a], xv[b], acc[a][b]);
      }
    }
  }
  if (!active) return;
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
      if (o0 + a < ly.O && i0 + b < ly.I) atomicAdd(ly.dW + (o0 + a) * ly.I + i0 + b, acc[a][b]);
}

void check_f32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, name, " must be contiguous float32 CUDA");
}

Weights weights_of(const std::vector<torch::Tensor>& ws) {
  TORCH_CHECK(ws.size() == 5, "expected 5 weight matrices");
  const int64_t shapes[5][2] = {{64, 32}, {16, 64}, {64, 31}, {64, 64}, {3, 64}};
  for (int i = 0; i < 5; ++i) {
    check_f32(ws[i], "weight");
    TORCH_CHECK(ws[i].size(0) == shapes[i][0] && ws[i].size(1) == shapes[i][1], "unexpected weight shape");
  }
  return {ws[0].data_ptr<float>(), ws[1].data_ptr<float>(), ws[2].data_ptr<float>(), ws[3].data_ptr<float>(),
          ws[4].data_ptr<float>()};
}

}  // namespace

std::vector<torch::Tensor> mlp_fwd(torch::Tensor enc, torch::Tensor dirs, std::vector<torch::Tensor> ws, bool save) {
  check_f32(enc, "enc");
  check_f32(dirs, "dirs");
  TORCH_CHECK(enc.size(1) == 32 && dirs.size(1) == 3 && enc.size(0) == dirs.size(0), "enc [N,32], dirs [N,3]");
  const Weights w = weights_of(ws);
  const int N = enc.size(0);
  auto sigma = torch::empty({N}, enc.options());
  auto rgb = torch::empty({N, 3}, enc.options());
  auto act = save ? torch::empty({N, ACT}, enc.options()) : torch::empty({0}, enc.options());
  if (N == 0) return {sigma, rgb, act};
  const size_t smem = W_FLOATS * sizeof(float);
  C10_CUDA_CHECK(cudaFuncSetAttribute(mlp_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  mlp_fwd_kernel<<<(N + 127) / 128, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
      enc.data_ptr<float>(), dirs.data_ptr<float>(), w, sigma.data_ptr<float>(), rgb.data_ptr<float>(),
      save ? act.data_ptr<float>() : nullptr, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sigma, rgb, act};
}

// returns {d enc, dWd1, dWd2, dWc1, dWc2, dWc3}
std::vector<torch::Tensor> mlp_bwd(torch::Tensor enc, torch::Tensor act, torch::Tensor rgb, torch::Tensor g_sigma,
                                   torch::Tensor g_rgb, std::vector<torch::Tensor> ws) {
  for (auto* t : {&enc, &act, &rgb, &g_sigma, &g_rgb}) check_f32(*t, "mlp_bwd input");
  const Weights w = weights_of(ws);
  const int N = enc.size(0);
  TORCH_CHECK(act.numel() == (int64_t)N * ACT, "act was not saved");
  auto g_enc = torch::empty({N, 32}, enc.options());
  auto dw = torch::zeros({64 * 32 + 16 * 64 + 64 * 31 + 64 * 64 + 3 * 64}, enc.options());
  std::vector<torch::Tensor> dws;
  int64_t off = 0;
  const int64_t shapes[5][2] = {{64, 32}, {16, 64}, {64, 31}, {64, 64}, {3, 64}};
  for (auto& s : shapes) {
    dws.push_back(dw.narrow(0, off, s[0] * s[1]).view({s[0], s[1]}));
    off += s[0] * s[1];
  }
  if (N > 0) {
    auto grads = torch::empty({N, GRD}, enc.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    const size_t smem = W_FLOATS * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(mlp_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    mlp_bwd_kernel<<<(N + 127) / 128, 128, smem, stream>>>(act.data_ptr<float>(), rgb.data_ptr<float>(),
                                                          g_sigma.data_ptr<float>(), g_rgb.data_ptr<float>(), w,
                                                          g_enc.data_ptr<float>(), grads.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const float* A = act.data_ptr<float>();
    const float* G = grads.data_ptr<float>();
    WLayers L;
    L.l[0] = {G + gH1, enc.data_ptr<float>(), dws[0].data_ptr<float>(), GRD, 32, 64, 32};  // Wd1
    L.l[1] = {G + gHd, A + aH1, dws[1].data_ptr<float>(), GRD, ACT, 16, 64};              // Wd2
    L.l[2] = {G + gA1, A + aC, dws[2].data_ptr<float>(), GRD, ACT, 64, 31};               // Wc1
    L.l[3] = {G + gA2, A + aA1, dws[3].data_ptr<float>(), GRD, ACT, 64, 64};              // Wc2
    L.l[4] = {G + gGo, A + aA2, dws[4].data_ptr<float>(), GRD, ACT, 3, 64};               // Wc3
    const int chunk = 2048;
    mlp_wgrad_kernel<<<dim3((N + chunk - 1) / chunk, 5), 256, 0, stream>>>(L, N, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {g_enc, dws[0], dws[1], dws[2], dws[3], dws[4]};
}
