"""Instant-NGP NeRF network: hash grid -> density MLP -> (sigma, 15 geo features); SH(dir) + geo -> color MLP."""
import torch
import torch.nn as nn

from .encoding import HashEncoding, sh_encoding_deg4


class TruncExp(torch.autograd.Function):
    """exp() whose gradient is clamped (exp(min(x, 15))), as in torch-ngp / nerfacc."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.exp(x.clamp(max=15))


def mlp(n_in, n_hidden, n_layers, n_out):
    layers, d = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, n_hidden, bias=False), nn.ReLU(inplace=True)]
        d = n_hidden
    layers.append(nn.Linear(d, n_out, bias=False))
    return nn.Sequential(*layers)


class NGPField(nn.Module):
    def __init__(self, aabb, log2_table=19, n_levels=16, n_features=2, n_min=16, n_max=2048, width=64):
        super().__init__()
        self.register_buffer("aabb", torch.as_tensor(aabb, dtype=torch.float32))  # [6] min xyz, max xyz
        self.encoding = HashEncoding(n_levels, n_features, log2_table, n_min, n_max)
        self.density_net = mlp(self.encoding.out_dim, width, 1, 16)
        self.color_net = mlp(15 + 16, width, 2, 3)

    def normalize(self, x):
        lo, hi = self.aabb[:3], self.aabb[3:]
        return (x - lo) / (hi - lo)

    def density(self, x, return_feat=False):
        h = self.density_net(self.encoding(self.normalize(x)))
        sigma = TruncExp.apply(h[:, 0])
        return (sigma, h[:, 1:]) if return_feat else sigma

    def forward(self, x, d):
        sigma, feat = self.density(x, return_feat=True)
        rgb = torch.sigmoid(self.color_net(torch.cat([feat, sh_encoding_deg4(d)], -1)))
        return sigma, rgb
