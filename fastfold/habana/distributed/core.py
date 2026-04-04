import os
import torch
import torch.distributed as dist

_DATA_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_RANK = None


def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0, \
        '{} is not divisible by {}'.format(numerator, denominator)


def init_dist(tensor_model_parallel_size_=1):
    """Initialize distributed backend for Gaudi HPU (HCCL), no mpi4py."""
    if dist.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size <= 1:
        # Single-device mode: init a trivial process group so
        # dist.get_world_size() returns 1 instead of crashing.
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        import habana_frameworks.torch.distributed.hccl  # noqa: F401
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    ensure_divisibility(world_size, tensor_model_parallel_size_)
    data_parallel_size_ = world_size // tensor_model_parallel_size_

    global _DATA_PARALLEL_GROUP
    for i in range(tensor_model_parallel_size_):
        ranks = list(range(i, world_size, tensor_model_parallel_size_))
        group = dist.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    global _TENSOR_MODEL_PARALLEL_GROUP
    for i in range(data_parallel_size_):
        ranks = list(range(i * tensor_model_parallel_size_,
                           (i + 1) * tensor_model_parallel_size_))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group


def dap_is_initialized():
    return (_TENSOR_MODEL_PARALLEL_GROUP is not None and
            _DATA_PARALLEL_GROUP is not None)


def get_tensor_model_parallel_group():
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    global _TENSOR_MODEL_PARALLEL_RANK
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_data_parallel_world_size():
    return dist.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    return dist.get_rank(group=get_data_parallel_group())


def get_tensor_model_parallel_src_rank():
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
