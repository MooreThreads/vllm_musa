cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

option(VLLM_TARGET_DEVICE "Target device backend for vLLM" "musa")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures.
# set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported MUSA architectures.
set(MUSA_SUPPORTED_ARCHS "220")

# Supported AMD GPU architectures.
# set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.2.0")
set(TORCH_SUPPORTED_VERSION_ROCM_5X "2.0.1")
set(TORCH_SUPPORTED_VERSION_ROCM_6X "2.1.1")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

include(/opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa/share/cmake/utils.cmake)

add_definitions(-DTORCH_MUSA_ARCH=220)
set(MUSA_CSRCS)
set(CMAKE_MODULE_PATH /opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa/share/cmake/modules)
set(DEPENDENT_LIBRARIES "")
set(DEPENDENT_INCLUDE_DIRS "")
find_package(MUDNN)

if(MUDNN_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MUDNN_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MUDNN_LIBRARIES})
else()
    message(WARNING " The environment variable MUSA_HOME may be not specified." 
    "Using default MUDNN PATH: /usr/local/musa")
    
    list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include")
    list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
    set(MUDNN_PATH "/usr/local/musa")
    set(MUDNN_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
endif()

find_package(MUSAToolkits)

if(MUSAToolkits_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MUSAToolkits_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MUSAToolkits_LIBRARIES})
else()
    message(WARNING " The environment variable MUSA_HOME may be not specified." 
    "Using default MUSATOOLKITS PATH: /usr/local/musa")
    
    list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include/")
    list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmusart.so")
    set(ENV{MUSA_HOME} "/usr/local/musa")
    set(MUSATOOLKITS_PATH "/usr/local/musa")
    set(MUSAToolkits_LIBRARIES "/usr/local/musa/lib/")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
    include_directories(${PYTHON_INCLUDE_DIR})
else()
    message(FATAL_ERROR, "Cannot find installed Python head file directory")
endif()

list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
# Normally `torch.utils.cpp_extension.CUDAExtension` would add
# `libtorch_python.so` for linking against an extension. Torch's cmake
# configuration does not include this library (presumably since the cmake
# config is used for standalone C++ binaries that link against torch).
# The `libtorch_python.so` library defines some of the glue code between
# torch/python via pybind and is required by VLLM extensions for this
# reason. So, add it by manually with `find_library` using torch's
# installed library path.
#
find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "musa" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
    if (VLLM_TARGET_DEVICE STREQUAL "cpu")
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
        message(FATAL_ERROR "Unsupported vLLM target device: ${VLLM_TARGET_DEVICE}")
    endif()
    return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND MUSA_FOUND)
  set(VLLM_GPU_LANG "MUSA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.x
  if (ROCM_VERSION_DEV_MAJOR EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM_5X})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_ROCM_5X} "
      "expected for ROCMm 5.x build, saw ${Torch_VERSION} instead.")
  endif()

  # ROCm 6.x
  if (ROCM_VERSION_DEV_MAJOR EQUAL 6 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM_6X})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_ROCM_6X} "
      "expected for ROCMm 6.x build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `VLLM_GPU_ARCHES`.
#
# override_gpu_arches(VLLM_GPU_ARCHES
#   ${VLLM_GPU_LANG}
#   "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Define extension targets
#

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc_musa/cache_kernels.mu"
  "csrc_musa/attention/attention_kernels.mu"
  "csrc_musa/pos_encoding_kernels.mu"
  "csrc_musa/activation_kernels.mu"
  "csrc_musa/layernorm_kernels.mu"
  "csrc_musa/quantization/squeezellm/quant_cuda_kernel.mu"
  "csrc_musa/quantization/gptq/q_gemm.mu"
  "csrc_musa/quantization/fp8/fp8_cuda_kernels.mu"
  "csrc_musa/musa_utils_kernels.mu"
  "csrc_musa/moe_align_block_size_kernels.mu"
  "csrc_musa/pybind.cpp")

if(VLLM_GPU_LANG STREQUAL "MUSA")
  list(APPEND VLLM_EXT_SRC
    "csrc_musa/quantization/aqlm/gemm_kernels.mu"
    "csrc_musa/quantization/awq/gemm_kernels.mu"
    "csrc_musa/quantization/marlin/marlin_cuda_kernel.mu"
    "csrc_musa/quantization/gptq_marlin/gptq_marlin.mu"
    "csrc_musa/quantization/gptq_marlin/gptq_marlin_repack.mu"
    "csrc_musa/custom_all_reduce.mu")
endif()

string(APPEND MUSA_MCC_FLAGS 

)
string(APPEND MUSA_MCC_FLAGS " -U__CUDA__")

set(MUSA_VERBOSE_BUILD ON)

                
musa_include_directories(
/opt/conda/envs/py39/include/python3.9
/usr/local/musa/include
/opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa/share/generated_cuda_compatible/aten/src
/opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa/share/generated_cuda_compatible/include
/opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa/share/generated_cuda_compatible/include/torch/csrc/api/include
/opt/conda/envs/py39/lib/python3.9/site-packages
/opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa
)

musa_add_library(vllm_C SHARED ${VLLM_EXT_SRC})
set(INSTALL_BIN_DIR "bin")
set(INSTALL_LIB_DIR "lib64")
set(INSTALL_INC_DIR "include")
set(INSTALL_SHARE_DIR "share")
set(INSTALL_DOC_DIR "docs")

define_gpu_extension_target(
  vllm_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  WITH_SOABI)

target_link_libraries(vllm_C ${DEPENDENT_LIBRARIES})
target_link_libraries(vllm_C "/opt/conda/envs/py39/lib/python3.9/site-packages/torch_musa/lib/libmusa_python.so")
#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc_musa/moe/moe_ops.cpp"
  "csrc_musa/moe/topk_softmax_kernels.mu")

define_gpu_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  WITH_SOABI)

#
# _punica_C extension
#

set(VLLM_PUNICA_EXT_SRC
  "csrc_musa/punica/bgmv/bgmv_bf16_bf16_bf16.mu"
  "csrc_musa/punica/bgmv/bgmv_bf16_fp32_bf16.mu"
  "csrc_musa/punica/bgmv/bgmv_fp16_fp16_fp16.mu"
  "csrc_musa/punica/bgmv/bgmv_fp16_fp32_fp16.mu"
  "csrc_musa/punica/bgmv/bgmv_fp32_bf16_bf16.mu"
  "csrc_musa/punica/bgmv/bgmv_fp32_fp16_fp16.mu"
  "csrc_musa/punica/punica_ops.cc")

#
# Copy GPU compilation flags+update for punica
#
set(VLLM_PUNICA_GPU_FLAGS ${VLLM_GPU_FLAGS})
list(REMOVE_ITEM VLLM_PUNICA_GPU_FLAGS
  "-D__MUSA_NO_HALF_OPERATORS__"
  "-D__MUSA_NO_HALF_CONVERSIONS__"
  "-D__MUSA_NO_BFLOAT16_CONVERSIONS__"
  "-D__MUSA_NO_HALF2_OPERATORS__")

#
# Filter out CUDA architectures < 8.0 for punica.
#
# if (${VLLM_GPU_LANG} STREQUAL "CUDA")
#   set(VLLM_PUNICA_GPU_ARCHES)
#   foreach(ARCH ${VLLM_GPU_ARCHES})
#     string_to_ver(CODE_VER ${ARCH})
#     if (CODE_VER GREATER_EQUAL 8.0)
#       list(APPEND VLLM_PUNICA_GPU_ARCHES ${ARCH})
#     endif()
#   endforeach()
#   message(STATUS "Punica target arches: ${VLLM_PUNICA_GPU_ARCHES}")
# endif()

if (VLLM_PUNICA_GPU_ARCHES)
  define_gpu_extension_target(
    _punica_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_PUNICA_EXT_SRC}
    COMPILE_FLAGS ${VLLM_PUNICA_GPU_FLAGS}
    ARCHITECTURES ${VLLM_PUNICA_GPU_ARCHES}
    WITH_SOABI)
else()
  message(WARNING "Unable to create _punica_C target because none of the "
    "requested architectures (${VLLM_GPU_ARCHES}) are supported, i.e. >= 8.0")
endif()

#
# Add the `default` target which detects which extensions should be
# built based on platform/architecture.  This is the same logic that
# setup.py uses to select which extensions should be built and should
# be kept in sync.
#
# The `default` target makes direct use of cmake easier since knowledge
# of which extensions are supported has been factored in, e.g.
#
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../vllm ..
# cmake --build . --target default
#
add_custom_target(default)

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "MUSA" OR VLLM_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling C extension.")
  add_dependencies(default _C)
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "MUSA")
  message(STATUS "Enabling moe extension.")
  add_dependencies(default _moe_C)

  # Enable punica if -DVLLM_INSTALL_PUNICA_KERNELS=ON or
  # VLLM_INSTALL_PUNICA_KERNELS is set in the environment and
  # there are supported target arches.
  if (VLLM_PUNICA_GPU_ARCHES AND
      (ENV{VLLM_INSTALL_PUNICA_KERNELS} OR VLLM_INSTALL_PUNICA_KERNELS))
    message(STATUS "Enabling punica extension.")
    add_dependencies(default _punica_C)
  endif()
endif()
