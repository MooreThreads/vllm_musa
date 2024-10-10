import os
from setuptools import setup, find_packages
from torch_musa.utils.simple_porting import SimplePorting
from torch_musa.utils.musa_extension import MUSAExtension

SimplePorting(cuda_dir_path="./csrc", mapping_rule={
    "x.device().is_cuda()": "true",
    "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
    "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
    "#include <ATen/cuda/Exceptions.h>": "#include \"torch_musa/csrc/core/MUSAException.h\"",
    "#include <c10/cuda/CUDAStream.h>": "#include \"torch_musa/csrc/core/MUSAStream.h\"",
    "at::kCUDA": "at::musa::kMUSA",
    "at::cuda::getCurrentCUDAStream()": "at::musa::getCurrentMUSAStream()",
    "__nv_bfloat16": "__mt_bfloat16",
    "at::cuda::OptionalCUDAGuard": "at::musa::OptionalMUSAGuard",
    "at::cuda::getCurrentCUDABlasHandle()": "at::musa::getCurrentMUSABlasHandle()",
    "ATen/cuda/CUDATensorMethods.cuh": "ATen/musa/MUSA_PORT_TensorMethods.muh",
    "#include \"attention_generic.cuh\"": "#include \"attention_generic.muh\"",
    "#include \"reduction_utils.cuh\"": "#include \"reduction_utils.muh\"",
    "#include <THC/THCAtomics.cuh>": "#include <THC/THCAtomics.muh>",
    "#include \"dtype_float16.cuh\"": "#include \"dtype_float16.muh\"",
    "#include \"dtype_float32.cuh\"": "#include \"dtype_float32.muh\"",
    "#include \"custom_all_reduce.cuh\"": "#include \"custom_all_reduce.muh\"",
    "#include \"dtype_bfloat16.cuh\"": "#include \"dtype_bfloat16.muh\"",
    "#include \"dtype_fp8.cuh\"": "#include \"dtype_fp8.muh\"",
    "#include \"attention_utils.cuh\"": "#include \"attention_utils.muh\"",
    "cuPointerGetAttribute": "muPointerGetAttribute",
    "CUdeviceptr": "MUdeviceptr",
    "CUDA_SUCCESS": "MUSA_SUCCESS",
    "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR": "MU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
    "c10::cuda": "c10::musa",
    "cudaStreamIsCapturing": "at::musa::musaStreamIsCapturing",
    "AT_CUDA_CHECK": "C10_MUSA_CHECK",
    "nv_bfloat16": "mt_bfloat16",
    "struct __align__(16) RankData { const void *__restrict__ ptrs[8]; };":"struct __align__(16) RankData { const void *__restrict__ ptrs[8]; RankData& operator=(const RankData& ){return *this;} };"
    }).run()