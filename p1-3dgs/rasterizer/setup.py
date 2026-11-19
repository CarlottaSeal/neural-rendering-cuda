#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# Optimization switches (see cuda_rasterizer/config.h), e.g. RAST_DEFINES="OPT_SHMEM_FEATURES=1 OPT_WARP_EXIT=1".
# Unset means every switch keeps its default from config.h.
_defines = ["-D" + d for d in os.environ.get("RAST_DEFINES", "").split()]
if _defines:
    print("diff_gaussian_rasterization build defines:", _defines)

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": _defines,
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                         "-lineinfo"] + _defines})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
