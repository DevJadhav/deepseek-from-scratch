"""
FSDP (Fully Sharded Data Parallel) Integration for DeepSeek Training.

This module provides comprehensive FSDP wrapping with:
- MixedPrecision policy configuration
- transformer_auto_wrap_policy for DeepSeek layers and experts
- Multiple sharding strategies (FULL_SHARD, HYBRID_SHARD, SHARD_GRAD_OP)
- CPU offloading for optimizer states
- FSDP-compatible gradient clipping
- Proper state dict saving/loading with StateDictType configuration
- FSDP scaling efficiency benchmarking

Reference: DeepSeek-V3 uses FSDP with expert parallelism for large-scale training.
"""

import functools
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Set, Type, Callable, Dict, Any, List, Union
from enum import Enum, auto

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
    LocalStateDictConfig,
    FullOptimStateDictConfig,
    ShardedOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    ModuleWrapPolicy,
)
from torch.distributed.fsdp.api import ShardingStrategy

from deepseek.torch.utils.logging import get_logger

logger = get_logger(__name__)


class FSDPShardingStrategy(str, Enum):
    """FSDP sharding strategies."""
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"
    HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"


@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapper.
    
    Attributes:
        sharding_strategy: How to shard model parameters
        cpu_offload: Whether to offload parameters to CPU
        mixed_precision: Mixed precision configuration
        backward_prefetch: Whether to prefetch next layer during backward
        forward_prefetch: Whether to prefetch next layer during forward
        limit_all_gathers: Limit concurrent all_gathers for memory efficiency
        use_orig_params: Use original parameter shapes (needed for torch.compile)
        sync_module_states: Sync module states from rank 0 at initialization
        auto_wrap_policy: Policy for automatic module wrapping
        min_num_params: Minimum params for size-based wrapping
        wrap_classes: Classes to wrap with transformer policy
    """
    sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD
    cpu_offload: bool = False
    backward_prefetch: bool = True
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    sync_module_states: bool = True
    
    # Mixed precision config
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    buffer_dtype: torch.dtype = torch.bfloat16
    
    # Wrapping config
    auto_wrap_policy: str = "transformer"  # "transformer", "size_based", or "module"
    min_num_params: int = 100_000
    wrap_classes: List[str] = field(default_factory=lambda: ["DeepSeekLayer", "Expert"])
    
    # Checkpoint config
    state_dict_type: str = "full"  # "full", "sharded", or "local"
    offload_to_cpu_for_save: bool = True
    rank0_only: bool = True


def get_sharding_strategy(strategy: FSDPShardingStrategy) -> ShardingStrategy:
    """Convert config enum to PyTorch ShardingStrategy."""
    mapping = {
        FSDPShardingStrategy.FULL_SHARD: ShardingStrategy.FULL_SHARD,
        FSDPShardingStrategy.SHARD_GRAD_OP: ShardingStrategy.SHARD_GRAD_OP,
        FSDPShardingStrategy.NO_SHARD: ShardingStrategy.NO_SHARD,
        FSDPShardingStrategy.HYBRID_SHARD: ShardingStrategy.HYBRID_SHARD,
        FSDPShardingStrategy.HYBRID_SHARD_ZERO2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return mapping.get(strategy, ShardingStrategy.FULL_SHARD)


def create_mixed_precision_policy(config: FSDPConfig) -> MixedPrecision:
    """Create MixedPrecision policy from config.
    
    DeepSeek-V3 uses BF16 for parameters, FP32 for reductions, BF16 for buffers.
    """
    return MixedPrecision(
        param_dtype=config.param_dtype,
        reduce_dtype=config.reduce_dtype,
        buffer_dtype=config.buffer_dtype,
    )


def get_wrap_classes(class_names: List[str]) -> Set[Type[nn.Module]]:
    """Get module classes to wrap from names.
    
    Dynamically imports DeepSeek layer classes.
    """
    classes = set()
    
    # Try to import DeepSeek classes
    try:
        from deepseek.torch.model.transformer import DeepSeekLayer
        if "DeepSeekLayer" in class_names:
            classes.add(DeepSeekLayer)
    except ImportError:
        pass
    
    try:
        from deepseek.torch.model.moe import Expert, DeepSeekMoE
        if "Expert" in class_names:
            classes.add(Expert)
        if "DeepSeekMoE" in class_names:
            classes.add(DeepSeekMoE)
    except ImportError:
        pass
    
    try:
        from deepseek.torch.model.mla import DeepSeekAttention
        if "DeepSeekAttention" in class_names:
            classes.add(DeepSeekAttention)
    except ImportError:
        pass
    
    return classes


def create_auto_wrap_policy(config: FSDPConfig) -> Callable:
    """Create auto wrap policy based on config.
    
    Supports:
    - transformer: Wraps specific transformer layer classes
    - size_based: Wraps modules with params >= min_num_params
    - module: Uses ModuleWrapPolicy with specified classes
    """
    if config.auto_wrap_policy == "transformer":
        wrap_classes = get_wrap_classes(config.wrap_classes)
        if not wrap_classes:
            # Fallback to size-based if no classes found
            logger.warning(
                "No wrap classes found for transformer policy, falling back to size-based"
            )
            return functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=config.min_num_params
            )
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=wrap_classes
        )
    elif config.auto_wrap_policy == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params
        )
    elif config.auto_wrap_policy == "module":
        wrap_classes = get_wrap_classes(config.wrap_classes)
        return ModuleWrapPolicy(wrap_classes)
    else:
        raise ValueError(f"Unknown auto_wrap_policy: {config.auto_wrap_policy}")


def create_cpu_offload(config: FSDPConfig) -> Optional[CPUOffload]:
    """Create CPU offload config."""
    if config.cpu_offload:
        return CPUOffload(offload_params=True)
    return None


def get_backward_prefetch(config: FSDPConfig) -> Optional[BackwardPrefetch]:
    """Get backward prefetch setting."""
    if config.backward_prefetch:
        return BackwardPrefetch.BACKWARD_PRE
    return BackwardPrefetch.BACKWARD_POST


class FSDPWrapper:
    """Wrapper class for FSDP model management.
    
    Provides:
    - Easy wrapping of models with FSDP
    - State dict saving/loading with proper configuration
    - FSDP-compatible gradient clipping
    - Memory tracking and optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[FSDPConfig] = None,
        device_id: Optional[int] = None,
    ):
        """Initialize FSDP wrapper.
        
        Args:
            model: Model to wrap with FSDP
            config: FSDP configuration
            device_id: CUDA device ID for this rank
        """
        self.config = config or FSDPConfig()
        self.device_id = device_id
        
        # Verify distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "FSDP requires distributed to be initialized. "
                "Use setup_distributed() first."
            )
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Wrap model
        self.model = self._wrap_model(model)
        
        logger.info(
            "FSDP wrapper initialized",
            rank=self.rank,
            world_size=self.world_size,
            sharding_strategy=self.config.sharding_strategy.value,
            cpu_offload=self.config.cpu_offload,
            mixed_precision=f"{self.config.param_dtype}",
        )
    
    def _wrap_model(self, model: nn.Module) -> FSDP:
        """Wrap model with FSDP using configuration."""
        # Create policies
        mixed_precision = create_mixed_precision_policy(self.config)
        auto_wrap_policy = create_auto_wrap_policy(self.config)
        cpu_offload = create_cpu_offload(self.config)
        sharding_strategy = get_sharding_strategy(self.config.sharding_strategy)
        backward_prefetch = get_backward_prefetch(self.config)
        
        # FSDP wrap
        wrapped_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=self.device_id,
            sync_module_states=self.config.sync_module_states,
            forward_prefetch=self.config.forward_prefetch,
            backward_prefetch=backward_prefetch,
            limit_all_gathers=self.config.limit_all_gathers,
            use_orig_params=self.config.use_orig_params,
        )
        
        return wrapped_model
    
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Clip gradients for FSDP model.
        
        FSDP requires special handling for gradient clipping since parameters
        are sharded across ranks.
        
        Args:
            max_norm: Maximum gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        return self.model.clip_grad_norm_(max_norm)
    
    def get_state_dict_type(self) -> StateDictType:
        """Get state dict type from config."""
        mapping = {
            "full": StateDictType.FULL_STATE_DICT,
            "sharded": StateDictType.SHARDED_STATE_DICT,
            "local": StateDictType.LOCAL_STATE_DICT,
        }
        return mapping.get(self.config.state_dict_type, StateDictType.FULL_STATE_DICT)
    
    def get_state_dict_config(self):
        """Get state dict config based on type."""
        state_dict_type = self.config.state_dict_type
        
        if state_dict_type == "full":
            return FullStateDictConfig(
                offload_to_cpu=self.config.offload_to_cpu_for_save,
                rank0_only=self.config.rank0_only,
            )
        elif state_dict_type == "sharded":
            return ShardedStateDictConfig(
                offload_to_cpu=self.config.offload_to_cpu_for_save,
            )
        elif state_dict_type == "local":
            return LocalStateDictConfig(
                offload_to_cpu=self.config.offload_to_cpu_for_save,
            )
        else:
            raise ValueError(f"Unknown state_dict_type: {state_dict_type}")
    
    def get_optim_state_dict_config(self):
        """Get optimizer state dict config based on type."""
        state_dict_type = self.config.state_dict_type
        
        if state_dict_type == "full":
            return FullOptimStateDictConfig(
                offload_to_cpu=self.config.offload_to_cpu_for_save,
                rank0_only=self.config.rank0_only,
            )
        elif state_dict_type == "sharded":
            return ShardedOptimStateDictConfig(
                offload_to_cpu=self.config.offload_to_cpu_for_save,
            )
        else:
            # Local state dict for optimizer not commonly used
            return None
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save FSDP checkpoint with proper state dict configuration.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer to save
            extra_state: Additional state to save (e.g., step, lr_scheduler)
        """
        state_dict_type = self.get_state_dict_type()
        state_dict_config = self.get_state_dict_config()
        optim_state_dict_config = self.get_optim_state_dict_config()
        
        with FSDP.state_dict_type(
            self.model,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        ):
            model_state = self.model.state_dict()
            
            checkpoint = {
                "model_state_dict": model_state,
                "fsdp_config": {
                    "sharding_strategy": self.config.sharding_strategy.value,
                    "state_dict_type": self.config.state_dict_type,
                },
            }
            
            if optimizer is not None:
                optim_state = FSDP.optim_state_dict(self.model, optimizer)
                checkpoint["optimizer_state_dict"] = optim_state
            
            if extra_state is not None:
                checkpoint.update(extra_state)
            
            # Only rank 0 saves for full state dict
            if self.config.state_dict_type == "full" and self.config.rank0_only:
                if self.rank == 0:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
            else:
                # All ranks save for sharded/local
                rank_path = f"{checkpoint_path}.rank{self.rank}"
                os.makedirs(os.path.dirname(rank_path), exist_ok=True)
                torch.save(checkpoint, rank_path)
                logger.info(f"Checkpoint saved to {rank_path}")
        
        # Synchronize after save
        dist.barrier()
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load FSDP checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optional optimizer to load state into
            
        Returns:
            Extra state from checkpoint
        """
        state_dict_type = self.get_state_dict_type()
        state_dict_config = self.get_state_dict_config()
        optim_state_dict_config = self.get_optim_state_dict_config()
        
        # Determine actual path based on state dict type
        if self.config.state_dict_type in ["sharded", "local"]:
            actual_path = f"{checkpoint_path}.rank{self.rank}"
        else:
            actual_path = checkpoint_path
        
        checkpoint = torch.load(actual_path, map_location="cpu")
        
        with FSDP.state_dict_type(
            self.model,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        ):
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optim_state = FSDP.optim_state_dict_to_load(
                    self.model,
                    optimizer,
                    checkpoint["optimizer_state_dict"],
                )
                optimizer.load_state_dict(optim_state)
        
        logger.info(f"Checkpoint loaded from {actual_path}")
        
        # Return extra state
        extra_keys = {"model_state_dict", "optimizer_state_dict", "fsdp_config"}
        return {k: v for k, v in checkpoint.items() if k not in extra_keys}
    
    def __call__(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        return self.model(*args, **kwargs)
    
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict)


def wrap_model_fsdp(
    model: nn.Module,
    config: Optional[FSDPConfig] = None,
    device_id: Optional[int] = None,
) -> FSDPWrapper:
    """Convenience function to wrap a model with FSDP.
    
    Args:
        model: Model to wrap
        config: FSDP configuration
        device_id: CUDA device ID
        
    Returns:
        FSDPWrapper containing the wrapped model
    """
    return FSDPWrapper(model, config, device_id)


def benchmark_fsdp_scaling(
    model_fn: Callable[[], nn.Module],
    input_fn: Callable[[int], torch.Tensor],
    world_sizes: List[int] = [1, 2, 4, 8],
    num_steps: int = 100,
    config: Optional[FSDPConfig] = None,
) -> Dict[int, Dict[str, float]]:
    """Benchmark FSDP scaling efficiency.
    
    Note: This needs to be run as a distributed job with varying world sizes.
    This function is a placeholder for integration with job launchers.
    
    Args:
        model_fn: Function to create model
        input_fn: Function to create input given batch size
        world_sizes: World sizes to benchmark
        num_steps: Number of steps per benchmark
        config: FSDP configuration
        
    Returns:
        Dict mapping world_size to metrics
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning empty results")
        return {}
    
    current_world_size = dist.get_world_size()
    results = {}
    
    # Create model and wrap with FSDP
    model = model_fn()
    wrapper = wrap_model_fsdp(model, config, dist.get_rank())
    
    # Create optimizer
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(10):
        x = input_fn(8)
        if torch.cuda.is_available():
            x = x.cuda()
        output = wrapper(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    import time
    start = time.time()
    
    for _ in range(num_steps):
        x = input_fn(8)
        if torch.cuda.is_available():
            x = x.cuda()
        output = wrapper(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    # Calculate metrics
    throughput = num_steps / elapsed
    
    results[current_world_size] = {
        "throughput_steps_per_sec": throughput,
        "elapsed_seconds": elapsed,
        "num_steps": num_steps,
    }
    
    logger.info(
        f"FSDP scaling benchmark",
        world_size=current_world_size,
        throughput=throughput,
        elapsed=elapsed,
    )
    
    return results


# Unit test utilities
def verify_fsdp_sharding_correctness(
    model: FSDP,
    reference_model: nn.Module,
    input_tensor: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Verify FSDP wrapped model produces same output as reference.
    
    Args:
        model: FSDP wrapped model
        reference_model: Unsharded reference model
        input_tensor: Test input
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if outputs match within tolerance
    """
    model.eval()
    reference_model.eval()
    
    with torch.no_grad():
        fsdp_output = model(input_tensor)
        ref_output = reference_model(input_tensor)
    
    # Gather FSDP output if distributed
    if dist.is_initialized() and dist.get_world_size() > 1:
        # All ranks should have same output for inference
        pass
    
    return torch.allclose(fsdp_output, ref_output, rtol=rtol, atol=atol)
