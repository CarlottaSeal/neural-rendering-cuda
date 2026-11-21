"""JIT-built CUDA kernels for P2 (torch.utils.cpp_extension.load). Run inside a VS dev env: scripts\\p2_run.bat."""
import os

import torch
from torch.utils.cpp_extension import load

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc")
_ext = None


def ext():
    global _ext
    if _ext is None and os.environ.get("INGP_PREBUILT"):
        # under nsys, torch's `where cl` lookup fails and the JIT relinks with a bad linker path; load the cached build
        from torch.utils.cpp_extension import _get_build_directory, _import_module_from_library
        _ext = _import_module_from_library("ingp_cuda", _get_build_directory("ingp_cuda", False), True)
    if _ext is None:
        _ext = load(
            name="ingp_cuda",
            sources=[os.path.join(_SRC, f) for f in ("bindings.cpp", "hash_encoding.cu", "march.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
            verbose=False,
        )
    return _ext


class HashEncodeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, res, version):
        x = x.contiguous()
        ctx.save_for_backward(x, res)
        ctx.shape, ctx.version = table.shape, version
        return ext().hash_encode_fwd(x, table.contiguous(), res, version)

    @staticmethod
    def backward(ctx, g):
        x, res = ctx.saved_tensors
        L, T, F = ctx.shape
        return None, ext().hash_encode_bwd(x, g.contiguous(), res, T, F, ctx.version), None, None
