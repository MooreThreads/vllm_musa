# This file is a pure Python wrapper for the MCCL library.
# The main purpose is to use MCCL combined with MUSA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls MCCL correctly, but `cupy` itself
#  often gets stuck when initializing the MCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential musa APIs, that are not allowed during
#  capturing the MUSA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-musagraph-with-mccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for MCCL. It is usually
# doable, but we often encounter issues related with mccl versions, and need
# to switch between different versions of MCCL. See
# https://github.com/NVIDIA/mccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of MCCL by
# changing the environment variable `VLLM_MCCL_SO_PATH`, or the `so_file`
# variable in the code.

import ctypes
import platform
from typing import Optional, Union

# ===================== import region =====================
import torch
import torch_musa
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.distributed.parallel_state import get_cpu_world_group, get_local_rank
from vllm.logger import init_logger
from vllm.utils import find_mccl_library, mccl_integrity_check

logger = init_logger(__name__)

so_file = find_mccl_library()

try:
    # load the library in another process.
    # if it core dumps, it will not crash the current process
    mccl_integrity_check(so_file)
    mccl = ctypes.CDLL(so_file)
except Exception as e:
    logger.error(
        "Failed to load MCCL library from %s ."
        "It is expected if you are not running on NVIDIA/AMD GPUs."
        "Otherwise, the mccl library might not exist, be corrupted "
        "or it does not support the current platform %s."
        "One solution is to download libmccl2 version 2.18 from "
        "https://developer.download.nvidia.com/compute/musa/repos/ "
        "and extract the libmccl.so.2 file. If you already have the "
        "library, please set the environment variable VLLM_MCCL_SO_PATH"
        " to point to the correct mccl library path.", so_file,
        platform.platform())
    raise e

# === export types and functions from mccl to Python ===
# for the original mccl definition, please check
# https://github.com/NVIDIA/mccl/blob/master/src/mccl.h.in

mcclResult_t = ctypes.c_int

_c_mcclGetErrorString = mccl.mcclGetErrorString
_c_mcclGetErrorString.restype = ctypes.c_char_p
_c_mcclGetErrorString.argtypes = [mcclResult_t]


def MCCL_CHECK(result: mcclResult_t) -> None:
    if result != 0:
        error_str = _c_mcclGetErrorString(result)
        error_str = error_str.decode("utf-8")
        raise RuntimeError(f"MCCL error: {error_str}")


# equivalent to c declaration:
# mcclResult_t  mcclGetVersion(int *version);
_c_mcclGetVersion = mccl.mcclGetVersion
_c_mcclGetVersion.restype = ctypes.c_int
_c_mcclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]


def mcclGetVersion() -> str:
    version = ctypes.c_int()
    MCCL_CHECK(_c_mcclGetVersion(ctypes.byref(version)))
    version_str = str(version.value)
    return version_str


class McclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# equivalent to c declaration:
# mcclResult_t mcclGetUniqueId(mcclUniqueId* uniqueId);
_c_mcclGetUniqueId = mccl.mcclGetUniqueId
_c_mcclGetUniqueId.restype = ctypes.c_int
_c_mcclGetUniqueId.argtypes = [ctypes.POINTER(McclUniqueId)]


def mcclGetUniqueId() -> McclUniqueId:
    unique_id = McclUniqueId()
    MCCL_CHECK(_c_mcclGetUniqueId(ctypes.byref(unique_id)))
    return unique_id


# equivalent to c declaration:
# mcclResult_t  mcclCommInitRank(
#   mcclComm_t* comm, int nranks, mcclUniqueId commId, int rank);
# note that mcclComm_t is a pointer type, so the first argument
# is a pointer to a pointer
_c_mcclCommInitRank = mccl.mcclCommInitRank
_c_mcclCommInitRank.restype = ctypes.c_int
_c_mcclCommInitRank.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, McclUniqueId, ctypes.c_int
]

mcclDataType_t = ctypes.c_int


class mcclDataTypeEnum:
    mcclInt8 = 0
    mcclChar = 0
    mcclUint8 = 1
    mcclInt32 = 2
    mcclInt = 2
    mcclUint32 = 3
    mcclInt64 = 4
    mcclUint64 = 5
    mcclFloat16 = 6
    mcclHalf = 6
    mcclFloat32 = 7
    mcclFloat = 7
    mcclFloat64 = 8
    mcclDouble = 8
    mcclBfloat16 = 9
    mcclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.mcclInt8
        if dtype == torch.uint8:
            return cls.mcclUint8
        if dtype == torch.int32:
            return cls.mcclInt32
        if dtype == torch.int64:
            return cls.mcclInt64
        if dtype == torch.float16:
            return cls.mcclFloat16
        if dtype == torch.float32:
            return cls.mcclFloat32
        if dtype == torch.float64:
            return cls.mcclFloat64
        if dtype == torch.bfloat16:
            return cls.mcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


mcclRedOp_t = ctypes.c_int


class mcclRedOpTypeEnum:
    mcclSum = 0
    mcclProd = 1
    mcclMax = 2
    mcclMin = 3
    mcclAvg = 4
    mcclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.mcclSum
        if op == ReduceOp.PRODUCT:
            return cls.mcclProd
        if op == ReduceOp.MAX:
            return cls.mcclMax
        if op == ReduceOp.MIN:
            return cls.mcclMin
        if op == ReduceOp.AVG:
            return cls.mcclAvg
        raise ValueError(f"Unsupported op: {op}")


# equivalent to c declaration:
# mcclResult_t  mcclAllReduce(
#   const void* sendbuff, void* recvbuff, size_t count,
#   mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
#   udaStream_t stream);
# note that musaStream_t is a pointer type, so the last argument is a pointer
_c_mcclAllReduce = mccl.mcclAllReduce
_c_mcclAllReduce.restype = ctypes.c_int
_c_mcclAllReduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, mcclRedOp_t,
    mcclDataType_t, ctypes.c_void_p, ctypes.c_void_p
]

# be cautious! this is a collective call, it will block until all
# processes in the communicator have called this function.
# because Python object destruction can happen in random order,
# it is better not to call it at all.
# equivalent to c declaration:
# mcclResult_t  mcclCommDestroy(mcclComm_t comm);
_c_mcclCommDestroy = mccl.mcclCommDestroy
_c_mcclCommDestroy.restype = ctypes.c_int
_c_mcclCommDestroy.argtypes = [ctypes.c_void_p]


class MCCLCommunicator:

    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        device: Optional[Union[int, str, torch.device]] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the MCCLCommunicator to. If None,
                it will be bind to f"musa:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        assert dist.is_initialized()
        group = get_cpu_world_group() if group is None else group
        assert dist.get_backend(group) != dist.Backend.MCCL, (
            "MCCLCommunicator should be attached to a non-MCCL group.")
        self.group = group
        # note: this rank is the rank in the group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.rank == 0:
            self.unique_id = mcclGetUniqueId()
        else:
            self.unique_id = McclUniqueId()
        tensor = torch.ByteTensor(list(self.unique_id.internal))
        ranks = dist.get_process_group_ranks(group)
        # arg `src` in `broadcast` is the global rank
        dist.broadcast(tensor, src=ranks[0], group=group)
        byte_list = tensor.tolist()
        for i, byte in enumerate(byte_list):
            self.unique_id.internal[i] = byte
        self.comm = ctypes.c_void_p()
        if device is None:
            local_rank = get_local_rank()
            device = torch.device(f"musa:{local_rank}")
        elif isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # mccl communicator and stream will use this device
        # `torch.musa.device` is a context manager that changes the
        # current musa device to the specified one
        with torch.musa.device(device):
            MCCL_CHECK(
                _c_mcclCommInitRank(ctypes.byref(self.comm), self.world_size,
                                    self.unique_id, self.rank))
            self.stream = torch.musa.Stream()

    def all_reduce(self,
                   tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None):
        # mccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this mccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self.stream
        MCCL_CHECK(
            _c_mcclAllReduce(ctypes.c_void_p(tensor.data_ptr()),
                             ctypes.c_void_p(tensor.data_ptr()),
                             tensor.numel(),
                             mcclDataTypeEnum.from_torch(tensor.dtype),
                             mcclRedOpTypeEnum.from_torch(op), self.comm,
                             ctypes.c_void_p(stream.musa_stream)))
