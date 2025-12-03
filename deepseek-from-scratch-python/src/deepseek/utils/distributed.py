import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from deepseek.utils.logging import get_logger

logger = get_logger(__name__)

# Group handles
_TENSOR_MODEL_PARALLEL_GROUP = None
_EXPERT_MODEL_PARALLEL_GROUP = None

def get_expert_model_parallel_group():
    """Get the expert model parallel group the caller rank belongs to."""
    return _EXPERT_MODEL_PARALLEL_GROUP

def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group."""
    if _EXPERT_MODEL_PARALLEL_GROUP is not None:
        return dist.get_world_size(group=_EXPERT_MODEL_PARALLEL_GROUP)
    return 1

def get_expert_model_parallel_rank():
    """Return my rank for the expert model parallel group."""
    if _EXPERT_MODEL_PARALLEL_GROUP is not None:
        return dist.get_rank(group=_EXPERT_MODEL_PARALLEL_GROUP)
    return 0

def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_GROUP is not None:
        return dist.get_world_size(group=_TENSOR_MODEL_PARALLEL_GROUP)
    return 1

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_GROUP is not None:
        return dist.get_rank(group=_TENSOR_MODEL_PARALLEL_GROUP)
    return 0

def is_dist_avail_and_initialized() -> bool:
    """Check if distributed package is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    """Get total number of processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Get rank of current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_local_rank() -> int:
    """Get local rank (GPU index) of current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0

def setup_distributed(backend: str = "nccl"):
    """
    Initialize distributed training.
    
    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU/GPU
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if torch.cuda.is_available() and backend == "nccl":
            torch.cuda.set_device(local_rank)
        
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        dist.barrier()
        
        # Initialize TP group (default to world size 1 for now if not specified)
        # In real scenario, we would split world into DP and TP groups
        # For now, let's assume TP size = 1 unless configured otherwise
        # We need a way to pass TP size here. 
        # Let's assume we just set it to None (size 1) for now.
        
        logger.info("Distributed initialized", rank=rank, world_size=world_size, backend=backend)
        return True
    else:
        logger.info("Not using distributed mode")
        return False

def cleanup_distributed():
    """Clean up distributed group."""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

def reduce_dict(input_dict: dict, average: bool = True):
    """
    Reduce a dictionary of values across all processes.
    
    Args:
        input_dict: Dictionary of values to reduce
        average: Whether to average (True) or sum (False)
    """
    if not is_dist_avail_and_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
        
    with torch.no_grad():
        names = []
        values = []
        # Sort keys for deterministic order
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            
        values = torch.stack([torch.tensor(v, device="cuda" if torch.cuda.is_available() else "cpu") for v in values])
        dist.all_reduce(values)
        
        if average:
            values /= world_size
            
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
        return reduced_dict
