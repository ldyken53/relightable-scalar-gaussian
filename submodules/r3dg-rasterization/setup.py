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

setup(
    name="r3dg_rasterization", # package 名称， import 时使用
    packages=['r3dg_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="r3dg_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "render_equation.cu",
                "reduced_3dgs/redundancy_score.cu",
                "reduced_3dgs/kmeans.cu",
                "reduced_3dgs.cu",
                "ext.cpp"], # 源文件，可以是多个，可以是cpp或cu文件，但是cu文件会被nvcc编译，cpp文件会被g++编译，所以cu文件中不能包含cpp代码，否则会报错，这里的文件路径是相对于setup.py的
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                         "-O3"],
                "cxx": ["-O3"]}) #-03 is for gcc complier, '/w' is for MSVC complier
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
