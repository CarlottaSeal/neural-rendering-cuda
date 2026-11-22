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
            sources=[os.path.join(_SRC, f) for f in ("bindings.cpp", "hash_encoding.cu", "march.cu", "mlp.cu", "mlp_train.cu")],
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


class FusedMLPFn(torch.autograd.Function):
    """Density + color MLP in one kernel each way. Weights: Wd1 [64,32], Wd2 [16,64], Wc1 [64,31], Wc2 [64,64], Wc3 [3,64]."""

    @staticmethod
    def forward(ctx, enc, dirs, version, *ws):
        """version 0: fp32 SIMT, 1: TF32 tensor cores"""
        enc, dirs = enc.contiguous(), dirs.contiguous()
        ws = [w.contiguous() for w in ws]
        sigma, rgb, act = ext().mlp_fwd(enc, dirs, ws, any(ctx.needs_input_grad), version)
        ctx.save_for_backward(enc, act, rgb, *ws)
        ctx.version = version
        return sigma, rgb

    @staticmethod
    def backward(ctx, g_sigma, g_rgb):
        enc, act, rgb, *ws = ctx.saved_tensors
        g_sigma = torch.zeros_like(rgb[:, 0]) if g_sigma is None else g_sigma.contiguous()
        g_rgb = torch.zeros_like(rgb) if g_rgb is None else g_rgb.contiguous()
        g_enc, *dws = ext().mlp_bwd(enc, act, rgb, g_sigma, g_rgb, ws, ctx.version)
        return (g_enc, None, None, *dws)
