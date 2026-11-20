"""JIT-built CUDA kernels for P2 (torch.utils.cpp_extension.load). Run inside a VS dev env: scripts\\p2_run.bat."""
import os

import torch
from torch.utils.cpp_extension import load

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc")
_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="ingp_cuda",
            sources=[os.path.join(_SRC, "hash_encoding.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
            verbose=False,
        )
    return _ext


class HashEncodeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, res):
        x = x.contiguous()
        ctx.save_for_backward(x, res)
        ctx.shape = table.shape
        return ext().hash_encode_fwd(x, table.contiguous(), res)

    @staticmethod
    def backward(ctx, g):
        x, res = ctx.saved_tensors
        L, T, F = ctx.shape
        return None, ext().hash_encode_bwd(x, g.contiguous(), res, T, F), None
