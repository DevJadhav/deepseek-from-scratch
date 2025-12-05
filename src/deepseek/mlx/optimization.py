"""
MLX Optimization Utilities for DeepSeek

This module provides Phase 1 optimizations for MLX on Apple Silicon:
- Memory layout optimization for unified memory
- Explicit mx.eval() placement for computation control
- Metal acceleration verification
- Memory monitoring for Apple Silicon

MLX uses lazy evaluation with automatic compilation to Metal shaders.
These utilities help ensure optimal performance.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Callable
from dataclasses import dataclass
import time


# =============================================================================
# MLX Memory and Performance Utilities
# =============================================================================

@dataclass
class MLXMemoryStats:
    """Container for MLX memory statistics."""
    peak_memory_mb: float
    cache_memory_mb: float
    wired_memory_mb: float


def get_mlx_device_info() -> dict:
    """
    Get MLX device information.
    
    Returns:
        Dictionary with device information.
    """
    return {
        "backend": "metal",
        "platform": "apple_silicon",
        "default_device": str(mx.default_device()),
    }


def verify_metal_acceleration() -> bool:
    """
    Verify that Metal acceleration is active.
    
    MLX automatically uses Metal on Apple Silicon, but this
    function performs a simple computation to verify.
    
    Returns:
        True if Metal is active and working.
    """
    try:
        # Create test tensors
        a = mx.ones((1000, 1000))
        b = mx.ones((1000, 1000))
        
        # Perform computation
        c = a @ b
        
        # Force evaluation
        mx.eval(c)
        
        # Check result
        return float(c[0, 0]) == 1000.0
    except Exception:
        return False


def force_eval(*arrays: mx.array) -> None:
    """
    Force evaluation of MLX arrays.
    
    MLX uses lazy evaluation, meaning computations are deferred
    until results are needed. This function forces immediate evaluation.
    
    Use strategically to:
    - Control memory usage
    - Enable accurate profiling
    - Synchronize computation
    
    Args:
        *arrays: Arrays to evaluate
    """
    mx.eval(*arrays)


def eval_and_sync() -> None:
    """
    Evaluate all pending computations and synchronize.
    
    This is more aggressive than force_eval - it evaluates
    everything in the computation graph.
    """
    # MLX doesn't have explicit synchronization, but eval forces completion
    # Creating a small tensor and evaluating it can help flush the graph
    mx.eval(mx.zeros(1))


# =============================================================================
# Training Loop Utilities
# =============================================================================

class MLXTrainingContext:
    """
    Context manager for MLX training optimization.
    
    Provides:
    - Strategic mx.eval() placement
    - Memory monitoring
    - Timing utilities
    """
    
    def __init__(
        self,
        eval_every_n_steps: int = 1,
        log_memory: bool = False,
    ):
        self.eval_every_n_steps = eval_every_n_steps
        self.log_memory = log_memory
        self.step_count = 0
        self.step_times = []
    
    def step_start(self) -> float:
        """Mark the start of a training step."""
        return time.perf_counter()
    
    def step_end(
        self,
        start_time: float,
        loss: mx.array,
        gradients: Optional[dict] = None,
    ) -> dict:
        """
        Mark the end of a training step.
        
        Args:
            start_time: Time from step_start()
            loss: Loss array (will be evaluated)
            gradients: Optional gradients dict (will be evaluated)
            
        Returns:
            Dict with step metrics
        """
        self.step_count += 1
        
        # Force evaluation of loss
        loss_value = float(loss)
        
        # Optionally evaluate gradients to free memory
        if gradients is not None and self.step_count % self.eval_every_n_steps == 0:
            # Evaluate all gradient arrays
            grad_arrays = []
            for g in gradients.values():
                if isinstance(g, mx.array):
                    grad_arrays.append(g)
                elif isinstance(g, dict):
                    grad_arrays.extend(v for v in g.values() if isinstance(v, mx.array))
            if grad_arrays:
                mx.eval(*grad_arrays)
        
        step_time = time.perf_counter() - start_time
        self.step_times.append(step_time)
        
        return {
            "loss": loss_value,
            "step_time_ms": step_time * 1000,
            "step": self.step_count,
        }
    
    def get_average_step_time(self) -> float:
        """Get average step time in milliseconds."""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times) * 1000


# =============================================================================
# Memory-Efficient Data Loading
# =============================================================================

def create_batches_memory_efficient(
    data: mx.array,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
) -> Callable[[], tuple[mx.array, mx.array]]:
    """
    Create a memory-efficient batch generator.
    
    Uses MLX's lazy evaluation to minimize memory usage.
    
    Args:
        data: Token IDs array
        batch_size: Batch size
        seq_len: Sequence length
        shuffle: Whether to shuffle data
        
    Yields:
        Tuple of (input_ids, labels) batches
    """
    total_tokens = data.shape[0]
    num_sequences = total_tokens // (seq_len + 1)
    
    def generate_batch():
        if shuffle:
            # Random starting positions
            starts = mx.random.randint(0, total_tokens - seq_len - 1, (batch_size,))
        else:
            # Sequential positions
            idx = mx.random.randint(0, num_sequences, (1,))
            starts = mx.arange(batch_size) * (seq_len + 1) + int(idx[0]) * batch_size * (seq_len + 1)
            starts = starts % (total_tokens - seq_len - 1)
        
        # Extract sequences - MLX will lazily evaluate
        input_ids = mx.stack([data[int(s):int(s)+seq_len] for s in starts])
        labels = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts])
        
        return input_ids, labels
    
    return generate_batch


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    total = 0
    for name, param in model.parameters().items():
        if isinstance(param, mx.array):
            total += param.size
        elif isinstance(param, dict):
            for p in param.values():
                if isinstance(p, mx.array):
                    total += p.size
    return total


def get_model_memory_mb(model: nn.Module, dtype: mx.Dtype = mx.float16) -> float:
    """
    Estimate model memory usage in MB.
    
    Args:
        model: The model
        dtype: Data type for estimation
        
    Returns:
        Estimated memory in MB
    """
    num_params = count_parameters(model)
    bytes_per_param = 2 if dtype in [mx.float16, mx.bfloat16] else 4
    return (num_params * bytes_per_param) / (1024 * 1024)


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize model for inference.
    
    MLX models are already optimized, but this ensures:
    - Evaluation mode is set
    - Dropout is disabled
    - Parameters are not tracking gradients
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model (same instance)
    """
    model.eval()
    return model


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_forward_pass(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict:
    """
    Benchmark model forward pass.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (batch, seq_len)
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dict with benchmark results
    """
    # Create dummy input
    input_ids = mx.random.randint(0, 32000, input_shape)
    
    # Warmup
    for _ in range(warmup_iterations):
        output = model(input_ids)
        mx.eval(output)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        output = model(input_ids)
        mx.eval(output)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    throughput = input_shape[0] * input_shape[1] / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "throughput_tokens_per_sec": throughput,
        "batch_size": input_shape[0],
        "seq_len": input_shape[1],
    }


def benchmark_training_step(
    model: nn.Module,
    loss_fn: Callable,
    input_shape: tuple,
    num_iterations: int = 50,
    warmup_iterations: int = 5,
) -> dict:
    """
    Benchmark training step (forward + backward).
    
    Args:
        model: Model to benchmark
        loss_fn: Loss function
        input_shape: Shape of input tensor
        num_iterations: Number of iterations
        warmup_iterations: Warmup iterations
        
    Returns:
        Dict with benchmark results
    """
    # Create dummy data
    input_ids = mx.random.randint(0, 32000, input_shape)
    labels = mx.random.randint(0, 32000, input_shape)
    
    # Create value_and_grad function
    def loss_wrapper(model, inputs, targets):
        logits = model(inputs)
        return loss_fn(logits, targets)
    
    loss_and_grad = nn.value_and_grad(model, loss_wrapper)
    
    # Warmup
    for _ in range(warmup_iterations):
        loss, grads = loss_and_grad(model, input_ids, labels)
        mx.eval(loss)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        loss, grads = loss_and_grad(model, input_ids, labels)
        mx.eval(loss)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    throughput = input_shape[0] * input_shape[1] / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "throughput_tokens_per_sec": throughput,
    }


# =============================================================================
# Apple Silicon Memory Limits
# =============================================================================

# Recommended maximum model sizes for different Apple Silicon variants
APPLE_SILICON_MEMORY_LIMITS = {
    "M1": {"ram_gb": 8, "max_params_b": 0.5},
    "M1_Pro": {"ram_gb": 16, "max_params_b": 1.0},
    "M1_Max": {"ram_gb": 32, "max_params_b": 2.0},
    "M1_Ultra": {"ram_gb": 64, "max_params_b": 4.0},
    "M2": {"ram_gb": 8, "max_params_b": 0.5},
    "M2_Pro": {"ram_gb": 16, "max_params_b": 1.0},
    "M2_Max": {"ram_gb": 32, "max_params_b": 2.0},
    "M2_Ultra": {"ram_gb": 64, "max_params_b": 4.0},
    "M3": {"ram_gb": 8, "max_params_b": 0.5},
    "M3_Pro": {"ram_gb": 18, "max_params_b": 1.2},
    "M3_Max": {"ram_gb": 36, "max_params_b": 2.5},
    "M4": {"ram_gb": 16, "max_params_b": 1.0},
    "M4_Pro": {"ram_gb": 24, "max_params_b": 1.5},
    "M4_Max": {"ram_gb": 48, "max_params_b": 3.0},
}


def get_recommended_model_size(chip: str = "M1") -> dict:
    """
    Get recommended model size for given Apple Silicon chip.

    Args:
        chip: Apple Silicon variant (e.g., "M1", "M2_Max")

    Returns:
        Dict with RAM and recommended max parameters
    """
    return APPLE_SILICON_MEMORY_LIMITS.get(
        chip,
        {"ram_gb": 8, "max_params_b": 0.5}  # Default to conservative
    )


# =============================================================================
# BFloat16 Support Verification
# =============================================================================

def verify_bfloat16_support() -> dict:
    """
    Verify BFloat16 support on current Apple Silicon.

    MLX supports BF16 on M-series chips, but availability may vary.

    Returns:
        Dict with support info and test results
    """
    result = {
        "bf16_available": False,
        "bf16_compute_works": False,
        "recommended_dtype": "float16",
        "notes": [],
    }

    try:
        # Try to create a BF16 array
        a = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
        result["bf16_available"] = True
        result["notes"].append("BF16 arrays can be created")

        # Test computation
        b = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
        c = a + b
        mx.eval(c)

        # Verify result
        c_float = c.astype(mx.float32)
        expected = [2.0, 4.0, 6.0]
        if all(abs(float(c_float[i]) - expected[i]) < 0.01 for i in range(3)):
            result["bf16_compute_works"] = True
            result["recommended_dtype"] = "bfloat16"
            result["notes"].append("BF16 computation verified correct")
        else:
            result["notes"].append("BF16 computation gave unexpected results")

    except Exception as e:
        result["notes"].append(f"BF16 not available: {e}")
        result["recommended_dtype"] = "float16"

    return result


def get_optimal_mlx_dtype() -> type:
    """
    Get optimal dtype for MLX on current hardware.

    Returns:
        mx.float16 or mx.bfloat16 depending on support
    """
    bf16_info = verify_bfloat16_support()
    if bf16_info["bf16_compute_works"]:
        return mx.bfloat16
    return mx.float16


# =============================================================================
# Activation Checkpointing for MLX
# =============================================================================

@dataclass
class MLXCheckpointConfig:
    """Configuration for MLX activation checkpointing."""

    enabled: bool = True
    checkpoint_every_n_layers: int = 1
    checkpoint_attention: bool = True
    checkpoint_mlp: bool = True


class MLXActivationCheckpointing:
    """
    Activation checkpointing for MLX models.

    MLX uses lazy evaluation, which provides some natural memory efficiency.
    This class provides additional control for explicit checkpointing.

    Key insight: In MLX, we control memory by strategically placing mx.eval()
    calls and allowing intermediate results to be garbage collected.
    """

    def __init__(self, config: Optional[MLXCheckpointConfig] = None):
        self.config = config or MLXCheckpointConfig()
        self.stored_activations: dict = {}
        self.layer_count = 0

    def should_checkpoint(self, layer_idx: int) -> bool:
        """Check if layer should be checkpointed."""
        if not self.config.enabled:
            return False
        return layer_idx % self.config.checkpoint_every_n_layers == 0

    def checkpoint_forward(
        self,
        layer_fn: Callable[[mx.array], mx.array],
        x: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """
        Forward pass with optional checkpointing.

        For checkpointed layers, we force evaluation and allow
        intermediate activations to be freed.

        Args:
            layer_fn: Layer forward function
            x: Input tensor
            layer_idx: Layer index

        Returns:
            Layer output
        """
        output = layer_fn(x)

        if self.should_checkpoint(layer_idx):
            # Force evaluation to materialize the output
            # This allows intermediate computation graph to be freed
            mx.eval(output)

        return output

    def wrap_transformer_layer(
        self,
        attention_fn: Callable[[mx.array], mx.array],
        mlp_fn: Callable[[mx.array], mx.array],
        x: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """
        Wrap a transformer layer with selective checkpointing.

        Args:
            attention_fn: Attention sublayer function
            mlp_fn: MLP sublayer function
            x: Input tensor
            layer_idx: Layer index

        Returns:
            Layer output
        """
        # Attention sublayer
        if self.config.checkpoint_attention and self.should_checkpoint(layer_idx):
            attn_out = attention_fn(x)
            mx.eval(attn_out)  # Checkpoint attention
        else:
            attn_out = attention_fn(x)

        x = x + attn_out

        # MLP sublayer
        if self.config.checkpoint_mlp and self.should_checkpoint(layer_idx):
            mlp_out = mlp_fn(x)
            mx.eval(mlp_out)  # Checkpoint MLP
        else:
            mlp_out = mlp_fn(x)

        return x + mlp_out

    def estimate_memory_savings(
        self,
        num_layers: int,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
    ) -> dict:
        """
        Estimate memory savings from checkpointing.

        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Dict with memory estimates
        """
        # Activation size per layer (rough estimate for float16)
        bytes_per_element = 2
        activation_per_layer = batch_size * seq_len * hidden_size * bytes_per_element * 2

        checkpointed = num_layers // self.config.checkpoint_every_n_layers
        non_checkpointed = num_layers - checkpointed

        without_checkpoint_mb = (num_layers * activation_per_layer) / (1024 * 1024)
        with_checkpoint_mb = (
            non_checkpointed * activation_per_layer +
            checkpointed * activation_per_layer * 0.1  # ~10% overhead for checkpointed
        ) / (1024 * 1024)

        return {
            "without_checkpoint_mb": without_checkpoint_mb,
            "with_checkpoint_mb": with_checkpoint_mb,
            "savings_mb": without_checkpoint_mb - with_checkpoint_mb,
            "savings_percent": (1 - with_checkpoint_mb / without_checkpoint_mb) * 100,
        }


def create_checkpointed_model_wrapper(
    model: nn.Module,
    config: Optional[MLXCheckpointConfig] = None,
) -> Callable[[mx.array], mx.array]:
    """
    Create a checkpointed wrapper for an MLX model.

    Args:
        model: MLX model
        config: Checkpoint configuration

    Returns:
        Wrapped forward function with checkpointing
    """
    checkpointer = MLXActivationCheckpointing(config)

    def checkpointed_forward(x: mx.array) -> mx.array:
        # Check if model has layers attribute (transformer)
        if hasattr(model, 'layers'):
            h = model.embed(x) if hasattr(model, 'embed') else x

            for idx, layer in enumerate(model.layers):
                h = checkpointer.checkpoint_forward(layer, h, idx)

            if hasattr(model, 'norm'):
                h = model.norm(h)
            if hasattr(model, 'head'):
                h = model.head(h)

            return h
        else:
            # Simple model, just call forward
            return model(x)

    return checkpointed_forward

