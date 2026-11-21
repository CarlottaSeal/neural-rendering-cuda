#include <torch/extension.h>

torch::Tensor hash_encode_fwd(torch::Tensor x, torch::Tensor table, torch::Tensor res, int64_t version);
torch::Tensor hash_encode_bwd(torch::Tensor x, torch::Tensor grad_out, torch::Tensor res, int64_t T, int64_t F, int64_t version);
std::vector<torch::Tensor> march_rays(torch::Tensor o, torch::Tensor d, torch::Tensor u, torch::Tensor occ,
                                      std::vector<double> aabb, int64_t res, double dt);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_encode_fwd", &hash_encode_fwd);
  m.def("hash_encode_bwd", &hash_encode_bwd);
  m.def("march_rays", &march_rays);
}
