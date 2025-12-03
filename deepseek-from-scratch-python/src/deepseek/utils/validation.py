import torch
import functools
from typing import Any, Callable
from deepseek.utils.logging import get_logger
from deepseek.utils.errors import TrainingError

logger = get_logger(__name__)

def check_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Check if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any():
        raise TrainingError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise TrainingError(f"Inf detected in {name}")

def validate_inputs(func: Callable):
    """Decorator to validate input tensors for NaN/Inf."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check args
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                check_nan_inf(arg, f"arg_{i}")
        
        # Check kwargs
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                check_nan_inf(v, f"kwarg_{k}")
                
        return func(*args, **kwargs)
    return wrapper

def validate_gradients(model: torch.nn.Module):
    """Check model gradients for NaN/Inf."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            try:
                check_nan_inf(param.grad, f"grad_{name}")
            except TrainingError as e:
                logger.error("Gradient validation failed", param=name, error=str(e))
                raise e
