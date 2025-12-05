"""
Training Optimization Utilities for DeepSeek

This module provides Phase 1 core GPU optimizations:
- torch.compile integration with mode selection and fallback
- Mixed precision training with automatic GPU detection (BF16/FP16)
- Memory profiling and monitoring
- NVTX annotations for profiling
- Precision-specific weight initialization
- NaN/Inf validation
- Memory budget management

Usage:
    from deepseek.torch.training.optimization import (
        compile_model,
        get_optimal_precision,
        MixedPrecisionConfig,
        MemoryProfiler,
        NaNInfValidator,
        MemoryBudgetManager,
    )
"""

import contextlib
import functools
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from deepseek.torch.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# torch.compile Integration
# =============================================================================

class CompileMode(Enum):
    """torch.compile mode options."""
    REDUCE_OVERHEAD = "reduce-overhead"  # Good for training
    MAX_AUTOTUNE = "max-autotune"  # Best for inference
    DEFAULT = "default"  # Balanced
    DISABLED = "disabled"  # No compilation


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""
    mode: CompileMode = CompileMode.REDUCE_OVERHEAD
    fullgraph: bool = False  # Try to compile entire model as single graph
    dynamic: bool = True  # Handle dynamic shapes (variable seq lengths)
    backend: str = "inductor"  # Default PyTorch backend
    # Warmup settings
    warmup_steps: int = 5  # Number of steps before compilation kicks in
    # Debugging
    debug_graph_breaks: bool = False  # Enable TORCH_LOGS="graph_breaks"


def compile_model(
    model: nn.Module,
    config: Optional[CompileConfig] = None,
) -> nn.Module:
    """
    Wrap model with torch.compile for optimized execution.
    
    Args:
        model: The model to compile
        config: Compilation configuration
        
    Returns:
        Compiled model (or original if compilation disabled/unavailable)
    """
    config = config or CompileConfig()
    
    # Check if compilation is disabled
    if config.mode == CompileMode.DISABLED:
        logger.info("torch.compile disabled by configuration")
        return model
    
    # Check PyTorch version
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    # Check CUDA availability for optimal performance
    if not torch.cuda.is_available():
        logger.warning("torch.compile works best with CUDA; falling back to eager mode on CPU/MPS")
        return model
    
    # Enable debug logging if requested
    if config.debug_graph_breaks:
        os.environ["TORCH_LOGS"] = "graph_breaks"
        logger.info("Enabled graph break logging (TORCH_LOGS=graph_breaks)")
    
    try:
        compiled_model = torch.compile(
            model,
            mode=config.mode.value,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
            backend=config.backend,
        )
        logger.info(
            "Model compiled successfully",
            mode=config.mode.value,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
            backend=config.backend,
        )
        return compiled_model
    except Exception as e:
        logger.error(f"torch.compile failed, falling back to eager mode: {e}")
        return model


def create_compile_warmup_wrapper(
    model: nn.Module,
    compile_config: CompileConfig,
) -> tuple[nn.Module, Callable[[], None]]:
    """
    Create a model wrapper that delays compilation until warmup is complete.
    
    This is useful to avoid compilation overhead during the first few steps.
    
    Args:
        model: The model to eventually compile
        compile_config: Compilation configuration
        
    Returns:
        Tuple of (model, trigger_compile_fn)
        Call trigger_compile_fn() after warmup to compile the model.
    """
    # Store reference for later compilation
    model_ref = {"model": model, "compiled": False}
    
    def trigger_compile():
        if not model_ref["compiled"]:
            model_ref["model"] = compile_model(model_ref["model"], compile_config)
            model_ref["compiled"] = True
            logger.info("Triggered model compilation after warmup")
    
    return model_ref["model"], trigger_compile


# =============================================================================
# Mixed Precision Training
# =============================================================================

class PrecisionMode(Enum):
    """Training precision modes."""
    FP32 = "fp32"  # Full precision
    FP16 = "fp16"  # Half precision with loss scaling
    BF16 = "bf16"  # Brain float16 (no loss scaling needed)
    AUTO = "auto"  # Automatic selection based on hardware


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    mode: PrecisionMode = PrecisionMode.AUTO
    # GradScaler settings (for FP16)
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    # Loss computation
    compute_loss_in_fp32: bool = True
    # Optimizer states
    optimizer_states_in_fp32: bool = True


def get_cuda_compute_capability() -> tuple[int, int] | None:
    """Get CUDA compute capability of current device."""
    if not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.current_device()
        return torch.cuda.get_device_capability(device)
    except Exception:
        return None


def supports_bfloat16() -> bool:
    """Check if current GPU supports BF16 (SM 80+, Ampere+)."""
    capability = get_cuda_compute_capability()
    if capability is None:
        return False
    major, _ = capability
    return major >= 8


def supports_fp16() -> bool:
    """Check if current GPU supports FP16 efficiently (SM 70+, Volta+)."""
    capability = get_cuda_compute_capability()
    if capability is None:
        return False
    major, _ = capability
    return major >= 7


def get_optimal_precision() -> PrecisionMode:
    """
    Determine optimal precision based on hardware.
    
    Returns:
        Optimal PrecisionMode for current hardware.
        - BF16 for SM 80+ (Ampere, Ada, Hopper)
        - FP16 for SM 70-79 (Volta, Turing)
        - FP32 for older GPUs or CPU
    """
    if not torch.cuda.is_available():
        return PrecisionMode.FP32
    
    if supports_bfloat16():
        return PrecisionMode.BF16
    elif supports_fp16():
        return PrecisionMode.FP16
    else:
        return PrecisionMode.FP32


def get_amp_dtype(mode: PrecisionMode) -> torch.dtype | None:
    """Get torch dtype for autocast based on precision mode."""
    if mode == PrecisionMode.BF16:
        return torch.bfloat16
    elif mode == PrecisionMode.FP16:
        return torch.float16
    else:
        return None


class MixedPrecisionTrainer:
    """
    Mixed precision training context manager and utilities.
    
    Handles:
    - Automatic precision selection based on GPU
    - GradScaler for FP16 training
    - BF16 training without scaling
    - FP32 fallback
    """
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig()
        
        # Determine actual precision mode
        if self.config.mode == PrecisionMode.AUTO:
            self.precision_mode = get_optimal_precision()
        else:
            self.precision_mode = self.config.mode
        
        # Validate mode is supported
        if self.precision_mode == PrecisionMode.BF16 and not supports_bfloat16():
            logger.warning("BF16 not supported on this GPU, falling back to FP16")
            self.precision_mode = PrecisionMode.FP16 if supports_fp16() else PrecisionMode.FP32
        elif self.precision_mode == PrecisionMode.FP16 and not supports_fp16():
            logger.warning("FP16 not efficient on this GPU, falling back to FP32")
            self.precision_mode = PrecisionMode.FP32
        
        # Initialize GradScaler for FP16 (not needed for BF16)
        self.scaler: Optional[GradScaler] = None
        if self.precision_mode == PrecisionMode.FP16:
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
            )
        
        # Get autocast dtype
        self.amp_dtype = get_amp_dtype(self.precision_mode)
        self.enabled = self.amp_dtype is not None
        
        logger.info(
            "Mixed precision initialized",
            mode=self.precision_mode.value,
            amp_dtype=str(self.amp_dtype) if self.amp_dtype else "disabled",
            grad_scaler="enabled" if self.scaler else "disabled",
        )
    
    @contextlib.contextmanager
    def autocast_context(self, device_type: str = "cuda"):
        """
        Context manager for autocast.
        
        Usage:
            with trainer.autocast_context():
                output = model(input)
        """
        if self.enabled and device_type == "cuda":
            with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP16 training."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: Optional[float] = None,
        model: Optional[nn.Module] = None,
    ) -> float:
        """
        Perform optimizer step with proper scaling.
        
        Args:
            optimizer: The optimizer
            clip_grad_norm: Optional gradient clipping value
            model: Model for gradient clipping (required if clip_grad_norm is set)
            
        Returns:
            Gradient norm (if clipping enabled, else 0.0)
        """
        grad_norm = 0.0
        
        # Unscale for gradient operations
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        
        # Gradient clipping (must be done after unscaling)
        if clip_grad_norm is not None and model is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_norm,
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        return grad_norm
    
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        state = {"precision_mode": self.precision_mode.value}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


# =============================================================================
# Memory Profiling and Monitoring
# =============================================================================

@dataclass
class MemoryStats:
    """Container for memory statistics."""
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    active_blocks: int
    inactive_split_blocks: int


class MemoryProfiler:
    """
    Memory profiler for tracking GPU memory usage.
    
    Provides:
    - Current memory statistics
    - Peak memory tracking
    - Memory logging at intervals
    - NVTX annotations for Nsight profiling
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        reset_peak_on_log: bool = True,
    ):
        self.log_interval = log_interval
        self.reset_peak_on_log = reset_peak_on_log
        self.step_count = 0
        
        # Track if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        
        # Try to import NVTX for profiling annotations
        self.nvtx_available = False
        try:
            if self.cuda_available:
                torch.cuda.nvtx.range_push("test")
                torch.cuda.nvtx.range_pop()
                self.nvtx_available = True
        except Exception:
            pass
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        if not self.cuda_available:
            return None
        
        stats = torch.cuda.memory_stats()
        return MemoryStats(
            allocated_mb=stats.get("allocated_bytes.all.current", 0) / 1024 / 1024,
            reserved_mb=stats.get("reserved_bytes.all.current", 0) / 1024 / 1024,
            peak_allocated_mb=stats.get("allocated_bytes.all.peak", 0) / 1024 / 1024,
            peak_reserved_mb=stats.get("reserved_bytes.all.peak", 0) / 1024 / 1024,
            active_blocks=stats.get("active.all.current", 0),
            inactive_split_blocks=stats.get("inactive_split.all.current", 0),
        )
    
    def log_memory(self, prefix: str = "") -> Optional[MemoryStats]:
        """Log current memory statistics."""
        stats = self.get_memory_stats()
        if stats:
            logger.info(
                f"{prefix}Memory stats",
                allocated_mb=f"{stats.allocated_mb:.1f}",
                reserved_mb=f"{stats.reserved_mb:.1f}",
                peak_allocated_mb=f"{stats.peak_allocated_mb:.1f}",
                peak_reserved_mb=f"{stats.peak_reserved_mb:.1f}",
            )
        return stats
    
    def step(self, force_log: bool = False) -> Optional[MemoryStats]:
        """
        Called each training step. Logs memory at intervals.
        
        Args:
            force_log: Force logging regardless of interval
            
        Returns:
            Memory stats if logged, else None
        """
        self.step_count += 1
        
        if force_log or (self.step_count % self.log_interval == 0):
            stats = self.log_memory(prefix=f"[Step {self.step_count}] ")
            
            if self.reset_peak_on_log and self.cuda_available:
                torch.cuda.reset_peak_memory_stats()
            
            return stats
        return None
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
    
    @contextlib.contextmanager
    def profile_region(self, name: str):
        """
        Context manager for profiling a region with NVTX markers.
        
        Usage:
            with profiler.profile_region("forward_pass"):
                output = model(input)
        """
        if self.nvtx_available:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if self.nvtx_available:
                torch.cuda.nvtx.range_pop()


# =============================================================================
# PyTorch Profiler Integration
# =============================================================================

def create_pytorch_profiler(
    output_dir: str = "./profiler_output",
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
    profile_memory: bool = True,
    with_stack: bool = True,
) -> torch.profiler.profile:
    """
    Create a PyTorch profiler with TensorBoard trace export.
    
    Args:
        output_dir: Directory for profiler output
        wait: Steps to wait before profiling
        warmup: Warmup steps
        active: Active profiling steps
        repeat: Number of cycles
        profile_memory: Whether to profile memory
        with_stack: Include stack traces
        
    Returns:
        Configured profiler instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )
    
    def trace_handler(prof):
        # Export Chrome trace
        prof.export_chrome_trace(
            os.path.join(output_dir, f"trace_{prof.step_num}.json")
        )
        # Export TensorBoard trace
        try:
            prof.export_stacks(
                os.path.join(output_dir, f"stacks_{prof.step_num}.txt"),
                "self_cuda_time_total"
            )
        except Exception:
            pass  # Stack export may fail on some systems
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    return torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )


# =============================================================================
# Model Bandwidth Utilization (MBU) Metric
# =============================================================================

def calculate_mbu(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    time_seconds: float,
    device: torch.device,
) -> float:
    """
    Calculate Model Bandwidth Utilization (MBU).
    
    MBU = (Bytes Accessed) / (Peak Memory Bandwidth * Time)
    
    Higher MBU indicates better memory bandwidth utilization.
    
    Args:
        model: The model
        batch_size: Batch size used
        seq_len: Sequence length
        time_seconds: Time for forward+backward pass
        device: Device the model is on
        
    Returns:
        MBU as a percentage (0-100)
    """
    if device.type != "cuda":
        return 0.0
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate bytes accessed (2x for forward+backward, element size)
    bytes_per_param = 4 if next(model.parameters()).dtype == torch.float32 else 2
    bytes_accessed = total_params * bytes_per_param * 2  # Forward + backward
    
    # Get device properties
    props = torch.cuda.get_device_properties(device)
    peak_bandwidth_bytes = props.memory_clock_rate * 1000 * (props.memory_bus_width / 8) * 2
    
    # Calculate MBU
    theoretical_time = bytes_accessed / peak_bandwidth_bytes
    mbu = (theoretical_time / time_seconds) * 100

    return min(mbu, 100.0)  # Cap at 100%


# =============================================================================
# Precision-Specific Weight Initialization
# =============================================================================

def get_init_scale_for_precision(
    precision: PrecisionMode,
    fan_in: int,
    fan_out: int,
) -> float:
    """
    Get initialization scale adjusted for training precision.

    Mixed precision training can affect gradient flow, so we adjust
    initialization scales to compensate.

    Args:
        precision: Training precision mode
        fan_in: Number of input features
        fan_out: Number of output features

    Returns:
        Scale factor for initialization
    """
    # Base scale using Xavier/Glorot
    base_scale = math.sqrt(2.0 / (fan_in + fan_out))

    # Precision-specific adjustments
    if precision == PrecisionMode.FP16:
        # FP16 has limited dynamic range, slightly reduce scale
        return base_scale * 0.95
    elif precision == PrecisionMode.BF16:
        # BF16 has same range as FP32, no adjustment needed
        return base_scale
    else:
        return base_scale


def init_weights_for_precision(
    module: nn.Module,
    precision: PrecisionMode = PrecisionMode.AUTO,
) -> None:
    """
    Initialize module weights with precision-aware scaling.

    Args:
        module: Module to initialize
        precision: Training precision mode
    """
    if precision == PrecisionMode.AUTO:
        precision = get_optimal_precision()

    for name, param in module.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            fan_in, fan_out = param.shape[-2], param.shape[-1]
            scale = get_init_scale_for_precision(precision, fan_in, fan_out)
            nn.init.normal_(param, mean=0.0, std=scale)
        elif 'bias' in name:
            nn.init.zeros_(param)


# =============================================================================
# NaN/Inf Validation
# =============================================================================

@dataclass
class NaNCheckResult:
    """Result of NaN/Inf check."""
    has_nan: bool
    has_inf: bool
    nan_params: list
    inf_params: list
    nan_grads: list
    inf_grads: list


def check_nan_inf(
    model: nn.Module,
    check_gradients: bool = True,
) -> NaNCheckResult:
    """
    Check model parameters and gradients for NaN/Inf values.

    Args:
        model: Model to check
        check_gradients: Whether to check gradients too

    Returns:
        NaNCheckResult with details about any issues found
    """
    nan_params = []
    inf_params = []
    nan_grads = []
    inf_grads = []

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)

        if check_gradients and param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)

    return NaNCheckResult(
        has_nan=len(nan_params) > 0 or len(nan_grads) > 0,
        has_inf=len(inf_params) > 0 or len(inf_grads) > 0,
        nan_params=nan_params,
        inf_params=inf_params,
        nan_grads=nan_grads,
        inf_grads=inf_grads,
    )


class NaNInfValidator:
    """
    Validator for detecting NaN/Inf during training.

    Can be used as a callback to check values at regular intervals.
    """

    def __init__(
        self,
        check_interval: int = 100,
        check_gradients: bool = True,
        raise_on_nan: bool = True,
        raise_on_inf: bool = False,
    ):
        self.check_interval = check_interval
        self.check_gradients = check_gradients
        self.raise_on_nan = raise_on_nan
        self.raise_on_inf = raise_on_inf
        self.step_count = 0

    def step(
        self,
        model: nn.Module,
        loss: Optional[torch.Tensor] = None,
    ) -> Optional[NaNCheckResult]:
        """
        Check for NaN/Inf values.

        Args:
            model: Model to check
            loss: Optional loss tensor to check

        Returns:
            NaNCheckResult if check was performed, None otherwise
        """
        self.step_count += 1

        if self.step_count % self.check_interval != 0:
            return None

        result = check_nan_inf(model, self.check_gradients)

        # Check loss
        if loss is not None:
            if torch.isnan(loss):
                result = NaNCheckResult(
                    has_nan=True,
                    has_inf=result.has_inf,
                    nan_params=result.nan_params + ["loss"],
                    inf_params=result.inf_params,
                    nan_grads=result.nan_grads,
                    inf_grads=result.inf_grads,
                )
            if torch.isinf(loss):
                result = NaNCheckResult(
                    has_nan=result.has_nan,
                    has_inf=True,
                    nan_params=result.nan_params,
                    inf_params=result.inf_params + ["loss"],
                    nan_grads=result.nan_grads,
                    inf_grads=result.inf_grads,
                )

        # Raise if configured
        if result.has_nan and self.raise_on_nan:
            raise ValueError(
                f"NaN detected in: params={result.nan_params}, grads={result.nan_grads}"
            )
        if result.has_inf and self.raise_on_inf:
            raise ValueError(
                f"Inf detected in: params={result.inf_params}, grads={result.inf_grads}"
            )

        return result


# =============================================================================
# Memory Budget and Automatic Batch Size
# =============================================================================

@dataclass
class MemoryBudgetConfig:
    """Configuration for memory budget management."""
    max_memory_mb: Optional[float] = None  # None = use all available
    memory_fraction: float = 0.9  # Use up to 90% of available memory
    min_batch_size: int = 1
    max_batch_size: int = 256
    enable_auto_adjustment: bool = True


class MemoryBudgetManager:
    """
    Manages memory budget and automatically adjusts batch size.

    Monitors GPU memory usage and reduces batch size if memory
    pressure is detected.
    """

    def __init__(self, config: Optional[MemoryBudgetConfig] = None):
        self.config = config or MemoryBudgetConfig()
        self.current_batch_size: Optional[int] = None
        self.oom_count = 0

        # Determine memory budget
        if self.config.max_memory_mb:
            self.memory_budget_mb = self.config.max_memory_mb
        elif torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            total_mb = props.total_memory / 1024 / 1024
            self.memory_budget_mb = total_mb * self.config.memory_fraction
        else:
            self.memory_budget_mb = float('inf')

        logger.info(
            "Memory budget initialized",
            budget_mb=f"{self.memory_budget_mb:.0f}",
            fraction=self.config.memory_fraction,
        )

    def get_recommended_batch_size(
        self,
        model: nn.Module,
        seq_len: int,
        start_batch_size: int = 32,
    ) -> int:
        """
        Find the maximum batch size that fits in memory budget.

        Uses binary search to find optimal batch size.

        Args:
            model: The model
            seq_len: Sequence length
            start_batch_size: Initial batch size to try

        Returns:
            Recommended batch size
        """
        if not torch.cuda.is_available() or not self.config.enable_auto_adjustment:
            return start_batch_size

        device = next(model.parameters()).device
        vocab_size = getattr(model, 'vocab_size', 32000)

        low = self.config.min_batch_size
        high = min(start_batch_size * 2, self.config.max_batch_size)
        best_batch_size = low

        while low <= high:
            mid = (low + high) // 2

            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Try forward pass
                dummy_input = torch.randint(0, vocab_size, (mid, seq_len), device=device)

                with torch.no_grad():
                    _ = model(dummy_input)

                # Check memory usage
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

                if peak_mb < self.memory_budget_mb:
                    best_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1

                del dummy_input
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise

        self.current_batch_size = best_batch_size
        logger.info(
            "Recommended batch size determined",
            batch_size=best_batch_size,
            seq_len=seq_len,
        )

        return best_batch_size

    def handle_oom(self) -> int:
        """
        Handle out-of-memory error by reducing batch size.

        Returns:
            New recommended batch size
        """
        self.oom_count += 1

        if self.current_batch_size is None:
            return self.config.min_batch_size

        # Reduce batch size by 25%
        new_batch_size = max(
            self.config.min_batch_size,
            int(self.current_batch_size * 0.75)
        )

        logger.warning(
            "OOM detected, reducing batch size",
            old_batch_size=self.current_batch_size,
            new_batch_size=new_batch_size,
            oom_count=self.oom_count,
        )

        self.current_batch_size = new_batch_size
        torch.cuda.empty_cache()

        return new_batch_size


# =============================================================================
# Compilation Overhead Documentation
# =============================================================================

def measure_compilation_overhead(
    model: nn.Module,
    sample_input: torch.Tensor,
    compile_config: Optional[CompileConfig] = None,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> dict:
    """
    Measure and document torch.compile overhead vs steady-state speedup.

    Args:
        model: Model to benchmark
        sample_input: Sample input tensor
        compile_config: Compilation configuration
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations

    Returns:
        Dictionary with timing results and analysis
    """
    import time

    device = sample_input.device
    results = {}

    # Benchmark uncompiled model
    model.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Time uncompiled
    times_uncompiled = []
    for _ in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_uncompiled.append(time.perf_counter() - start)

    results['uncompiled_mean_ms'] = sum(times_uncompiled) / len(times_uncompiled) * 1000
    results['uncompiled_std_ms'] = (
        sum((t - results['uncompiled_mean_ms']/1000)**2 for t in times_uncompiled)
        / len(times_uncompiled)
    ) ** 0.5 * 1000

    # Compile model
    config = compile_config or CompileConfig()
    if config.mode != CompileMode.DISABLED:
        # Time compilation
        compile_start = time.perf_counter()
        compiled_model = compile_model(model, config)
        compile_time = time.perf_counter() - compile_start
        results['compilation_time_s'] = compile_time

        # First run (includes remaining compilation)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        first_run_start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        results['first_run_time_ms'] = (time.perf_counter() - first_run_start) * 1000

        # Warmup compiled
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = compiled_model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Time compiled
        times_compiled = []
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_compiled.append(time.perf_counter() - start)

        results['compiled_mean_ms'] = sum(times_compiled) / len(times_compiled) * 1000
        results['compiled_std_ms'] = (
            sum((t - results['compiled_mean_ms']/1000)**2 for t in times_compiled)
            / len(times_compiled)
        ) ** 0.5 * 1000

        # Calculate speedup
        results['speedup'] = results['uncompiled_mean_ms'] / results['compiled_mean_ms']
        results['overhead_amortization_iterations'] = int(
            results['first_run_time_ms'] / (results['uncompiled_mean_ms'] - results['compiled_mean_ms'])
        ) if results['speedup'] > 1 else float('inf')

    return results


# =============================================================================
# Precision-Specific Weight Initialization
# =============================================================================

@dataclass
class PrecisionWeightInitializer:
    """
    Weight initialization with precision-specific scaling.
    
    Different precisions require different initialization scales to maintain
    proper gradient flow and avoid overflow/underflow.
    """
    precision: PrecisionMode = PrecisionMode.BF16
    base_std: float = 0.02
    use_scaled_init: bool = True
    
    def get_init_std(self, fan_in: int, fan_out: int) -> float:
        """
        Get initialization standard deviation scaled for precision.
        
        Args:
            fan_in: Number of input features
            fan_out: Number of output features
            
        Returns:
            Scaled standard deviation for initialization
        """
        # Base: Xavier/Glorot uniform equivalent
        base = self.base_std if not self.use_scaled_init else (2.0 / (fan_in + fan_out)) ** 0.5
        
        # Scale based on precision
        if self.precision == PrecisionMode.FP16:
            # FP16 has smaller dynamic range, use slightly smaller init
            return base * 0.9
        elif self.precision == PrecisionMode.BF16:
            # BF16 has same range as FP32 but less precision
            return base
        else:  # FP32
            return base
    
    def initialize_linear(self, module: nn.Linear) -> None:
        """Initialize a linear layer with precision-aware scaling."""
        fan_in = module.in_features
        fan_out = module.out_features
        std = self.get_init_std(fan_in, fan_out)
        
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def initialize_embedding(self, module: nn.Embedding) -> None:
        """Initialize an embedding layer with precision-aware scaling."""
        std = self.get_init_std(module.embedding_dim, module.embedding_dim)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def initialize_model(self, model: nn.Module) -> None:
        """
        Initialize all layers in a model with precision-aware scaling.
        
        Args:
            model: Model to initialize
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.initialize_linear(module)
                logger.debug(f"Initialized linear layer: {name}")
            elif isinstance(module, nn.Embedding):
                self.initialize_embedding(module)
                logger.debug(f"Initialized embedding layer: {name}")


# =============================================================================
# Numerical Validator (alias for NaNInfValidator with extended features)
# =============================================================================

class NumericalValidator(NaNInfValidator):
    """
    Extended numerical validator with additional validation features.
    
    Alias for NaNInfValidator with extra methods for comprehensive
    numerical stability checking.
    """
    
    def __init__(
        self,
        check_interval: int = 100,
        check_gradients: bool = True,
        raise_on_nan: bool = True,
        raise_on_inf: bool = False,
        check_loss_scale: bool = True,
        max_loss_value: float = 1e4,
    ):
        super().__init__(
            check_interval=check_interval,
            check_gradients=check_gradients,
            raise_on_nan=raise_on_nan,
            raise_on_inf=raise_on_inf,
        )
        self.check_loss_scale = check_loss_scale
        self.max_loss_value = max_loss_value
        self.loss_history: list[float] = []
    
    def validate_loss(self, loss: torch.Tensor) -> bool:
        """
        Validate loss value for numerical issues.
        
        Args:
            loss: Loss tensor to validate
            
        Returns:
            True if loss is valid, False otherwise
        """
        loss_val = loss.item()
        
        if torch.isnan(loss):
            if self.raise_on_nan:
                raise ValueError("Loss is NaN")
            return False
        
        if torch.isinf(loss):
            if self.raise_on_inf:
                raise ValueError("Loss is Inf")
            return False
        
        if self.check_loss_scale and abs(loss_val) > self.max_loss_value:
            logger.warning(
                f"Loss value {loss_val} exceeds max threshold {self.max_loss_value}"
            )
            return False
        
        self.loss_history.append(loss_val)
        return True
    
    def check_gradient_norm(
        self,
        model: nn.Module,
        max_norm: float = 1000.0,
    ) -> tuple[float, bool]:
        """
        Check total gradient norm.
        
        Args:
            model: Model to check
            max_norm: Maximum acceptable gradient norm
            
        Returns:
            Tuple of (gradient_norm, is_valid)
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        is_valid = total_norm <= max_norm and not (
            torch.isnan(torch.tensor(total_norm)) or 
            torch.isinf(torch.tensor(total_norm))
        )
        
        return total_norm, is_valid


# =============================================================================
# CPU Offloading Configuration
# =============================================================================

@dataclass
class CPUOffloadConfig:
    """
    Configuration for CPU offloading of optimizer states.
    
    Enables training larger models by keeping optimizer states on CPU
    and transferring them to GPU only when needed.
    """
    enabled: bool = True
    offload_optimizer: bool = True
    offload_gradients: bool = False
    pin_memory: bool = True
    non_blocking: bool = True
    prefetch_count: int = 2  # Number of layers to prefetch
    
    def create_cpu_optimizer_state(
        self,
        param: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Create CPU-resident optimizer state for a parameter.
        
        Args:
            param: Parameter tensor
            
        Returns:
            Dictionary of optimizer state tensors on CPU
        """
        cpu_state = {
            'exp_avg': torch.zeros_like(param, device='cpu'),
            'exp_avg_sq': torch.zeros_like(param, device='cpu'),
        }
        
        if self.pin_memory:
            for key in cpu_state:
                cpu_state[key] = cpu_state[key].pin_memory()
        
        return cpu_state
    
    def transfer_to_gpu(
        self,
        state: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Transfer optimizer state from CPU to GPU."""
        return {
            key: tensor.to(device, non_blocking=self.non_blocking)
            for key, tensor in state.items()
        }
    
    def transfer_to_cpu(
        self,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Transfer optimizer state from GPU to CPU."""
        cpu_state = {
            key: tensor.to('cpu', non_blocking=self.non_blocking)
            for key, tensor in state.items()
        }
        
        if self.pin_memory:
            cpu_state = {
                key: tensor.pin_memory() if not tensor.is_pinned() else tensor
                for key, tensor in cpu_state.items()
            }
        
        return cpu_state


# =============================================================================
# Gradient Accumulation Configuration
# =============================================================================

@dataclass
class GradientAccumulationConfig:
    """
    Configuration for gradient accumulation.
    
    Enables effective larger batch sizes by accumulating gradients
    over multiple forward passes before updating weights.
    """
    accumulation_steps: int = 1
    normalize_gradients: bool = True
    sync_gradients: bool = True  # For distributed training
    
    @property
    def effective_batch_multiplier(self) -> int:
        """Get the effective batch size multiplier."""
        return self.accumulation_steps
    
    def should_step(self, current_step: int) -> bool:
        """
        Check if optimizer should step at current accumulation step.
        
        Args:
            current_step: Current micro-batch step (0-indexed)
            
        Returns:
            True if optimizer should step
        """
        return (current_step + 1) % self.accumulation_steps == 0
    
    def get_loss_scale(self) -> float:
        """Get loss scale factor for gradient normalization."""
        if self.normalize_gradients:
            return 1.0 / self.accumulation_steps
        return 1.0


class GradientAccumulationContext:
    """
    Context manager for gradient accumulation.
    
    Handles gradient scaling, accumulation counting, and optimizer stepping.
    """
    
    def __init__(
        self,
        config: GradientAccumulationConfig,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
    ):
        self.config = config
        self.optimizer = optimizer
        self.scaler = scaler
        self.micro_step = 0
        self.accumulated_loss = 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset state on exit
        self.micro_step = 0
        self.accumulated_loss = 0.0
        return False
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass with proper scaling.
        
        Args:
            loss: Loss tensor
        """
        scaled_loss = loss * self.config.get_loss_scale()
        
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.micro_step += 1
    
    def step(self) -> tuple[bool, float]:
        """
        Perform optimizer step if accumulation complete.
        
        Returns:
            Tuple of (did_step, accumulated_loss)
        """
        if self.config.should_step(self.micro_step - 1):
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            avg_loss = self.accumulated_loss / self.config.accumulation_steps
            self.accumulated_loss = 0.0
            
            return True, avg_loss
        
        return False, 0.0


def create_gradient_accumulation_context(
    accumulation_steps: int,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler] = None,
    normalize_gradients: bool = True,
) -> GradientAccumulationContext:
    """
    Create a gradient accumulation context.
    
    Args:
        accumulation_steps: Number of steps to accumulate gradients
        optimizer: Optimizer to use
        scaler: Optional GradScaler for mixed precision
        normalize_gradients: Whether to normalize by accumulation steps
        
    Returns:
        GradientAccumulationContext instance
    """
    config = GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        normalize_gradients=normalize_gradients,
    )
    return GradientAccumulationContext(config, optimizer, scaler)
