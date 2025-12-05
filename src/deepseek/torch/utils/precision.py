"""
Automatic Precision Detection and Configuration.

This module provides automatic precision selection based on hardware capabilities:
- H100/H200: FP8 (E4M3 format)
- A100: BF16
- Older/Consumer GPUs: FP16
- CPU fallback: FP32

Usage:
    from deepseek.torch.utils.precision import PrecisionManager, detect_optimal_precision
    
    # Auto-detect best precision
    precision = detect_optimal_precision()
    print(f"Optimal precision: {precision}")
    
    # Create precision manager
    manager = PrecisionManager(mode="auto")
    ctx = manager.get_autocast_context()
    with ctx:
        output = model(input)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any
import logging

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes."""
    AUTO = "auto"
    FP8 = "fp8"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class FP8Format(Enum):
    """FP8 format types."""
    E4M3 = "e4m3"  # 4-bit exponent, 3-bit mantissa (higher precision, lower range)
    E5M2 = "e5m2"  # 5-bit exponent, 2-bit mantissa (lower precision, higher range)


# Compute capability thresholds
COMPUTE_CAPABILITY_FP8 = 9.0    # H100, H200
COMPUTE_CAPABILITY_BF16 = 8.0  # A100, RTX 30xx
COMPUTE_CAPABILITY_FP16 = 6.0  # Pascal (P100, GTX 10xx)


# GPU name to precision mapping
GPU_PRECISION_MAP: dict[str, PrecisionMode] = {
    # H100 series (FP8)
    "H100": PrecisionMode.FP8,
    "H200": PrecisionMode.FP8,
    
    # A100 series (BF16)
    "A100": PrecisionMode.BF16,
    "A10": PrecisionMode.BF16,
    "A10G": PrecisionMode.BF16,
    "A30": PrecisionMode.BF16,
    "A40": PrecisionMode.BF16,
    
    # RTX 40xx (BF16)
    "RTX 4090": PrecisionMode.BF16,
    "RTX 4080": PrecisionMode.BF16,
    "RTX 4070": PrecisionMode.BF16,
    
    # RTX 30xx (BF16)
    "RTX 3090": PrecisionMode.BF16,
    "RTX 3080": PrecisionMode.BF16,
    "RTX 3070": PrecisionMode.BF16,
    
    # Older GPUs (FP16)
    "V100": PrecisionMode.FP16,
    "RTX 2080": PrecisionMode.FP16,
    "RTX 2070": PrecisionMode.FP16,
    "T4": PrecisionMode.FP16,
    "P100": PrecisionMode.FP16,
}


@dataclass
class PrecisionConfig:
    """Configuration for precision settings."""
    mode: PrecisionMode = PrecisionMode.AUTO
    
    # FP8 settings
    fp8_format: FP8Format = FP8Format.E4M3
    fp8_gradient_format: FP8Format = FP8Format.E5M2
    fp8_tile_size: int = 128
    fp8_amax_history_len: int = 1024
    fp8_delayed_scaling: bool = True
    
    # FP16 settings
    fp16_initial_scale: float = 65536.0
    fp16_growth_factor: float = 2.0
    fp16_backoff_factor: float = 0.5
    fp16_growth_interval: int = 2000
    
    # Common settings
    loss_in_fp32: bool = True
    optimizer_in_fp32: bool = True
    gradient_clipping_in_fp32: bool = True
    
    # Fallback behavior
    warn_on_fallback: bool = True
    strict_mode: bool = False  # Fail if requested precision unavailable


@dataclass
class HardwareInfo:
    """Hardware information for precision detection."""
    device_name: str = "unknown"
    compute_capability: tuple[int, int] = (0, 0)
    total_memory_gb: float = 0.0
    supports_fp8: bool = False
    supports_bf16: bool = False
    supports_fp16: bool = True
    is_cuda: bool = False
    is_mps: bool = False  # Apple Silicon


def _try_import_torch():
    """Safely import torch."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def detect_hardware_info() -> HardwareInfo:
    """Detect hardware capabilities for precision selection.
    
    Returns:
        HardwareInfo with device capabilities
    """
    torch = _try_import_torch()
    
    if torch is None:
        return HardwareInfo(device_name="CPU (torch not available)")
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        compute = (props.major, props.minor)
        compute_float = props.major + props.minor / 10.0
        
        return HardwareInfo(
            device_name=props.name,
            compute_capability=compute,
            total_memory_gb=props.total_memory / (1024**3),
            supports_fp8=compute_float >= COMPUTE_CAPABILITY_FP8,
            supports_bf16=compute_float >= COMPUTE_CAPABILITY_BF16,
            supports_fp16=compute_float >= COMPUTE_CAPABILITY_FP16,
            is_cuda=True,
            is_mps=False,
        )
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return HardwareInfo(
            device_name="Apple Silicon (MPS)",
            supports_fp8=False,  # MPS doesn't support FP8
            supports_bf16=True,  # MPS supports BF16
            supports_fp16=True,
            is_cuda=False,
            is_mps=True,
        )
    
    # CPU fallback
    return HardwareInfo(
        device_name="CPU",
        supports_fp8=False,
        supports_bf16=False,
        supports_fp16=False,
        is_cuda=False,
        is_mps=False,
    )


def detect_optimal_precision(
    hardware_info: HardwareInfo | None = None,
    config: PrecisionConfig | None = None,
) -> PrecisionMode:
    """Detect the optimal precision mode for current hardware.
    
    Args:
        hardware_info: Pre-detected hardware info (optional)
        config: Precision configuration (optional)
        
    Returns:
        Optimal PrecisionMode for the hardware
    """
    if hardware_info is None:
        hardware_info = detect_hardware_info()
    
    if config is None:
        config = PrecisionConfig()
    
    # If not AUTO mode, validate and return requested precision
    if config.mode != PrecisionMode.AUTO:
        return _validate_precision_mode(config.mode, hardware_info, config)
    
    # Auto-detect based on hardware
    # Check by GPU name first (more reliable)
    for gpu_prefix, precision in GPU_PRECISION_MAP.items():
        if gpu_prefix.lower() in hardware_info.device_name.lower():
            logger.info(f"Detected {hardware_info.device_name}, using {precision.value}")
            return precision
    
    # Fall back to compute capability
    if hardware_info.supports_fp8:
        logger.info(f"Hardware supports FP8 (compute capability >= {COMPUTE_CAPABILITY_FP8})")
        return PrecisionMode.FP8
    
    if hardware_info.supports_bf16:
        logger.info(f"Hardware supports BF16 (compute capability >= {COMPUTE_CAPABILITY_BF16})")
        return PrecisionMode.BF16
    
    if hardware_info.supports_fp16:
        logger.info(f"Hardware supports FP16 (compute capability >= {COMPUTE_CAPABILITY_FP16})")
        return PrecisionMode.FP16
    
    # MPS (Apple Silicon)
    if hardware_info.is_mps:
        logger.info("Detected Apple Silicon, using FP16 (BF16 available but FP16 is more tested)")
        return PrecisionMode.FP16
    
    # CPU fallback
    logger.warning("No GPU detected, falling back to FP32")
    return PrecisionMode.FP32


def _validate_precision_mode(
    mode: PrecisionMode,
    hardware_info: HardwareInfo,
    config: PrecisionConfig,
) -> PrecisionMode:
    """Validate requested precision mode against hardware capabilities.
    
    Args:
        mode: Requested precision mode
        hardware_info: Detected hardware info
        config: Precision configuration
        
    Returns:
        Validated precision mode (may be fallback)
        
    Raises:
        RuntimeError: If strict_mode and precision not supported
    """
    if mode == PrecisionMode.FP32:
        return mode  # Always supported
    
    # Check FP8
    if mode == PrecisionMode.FP8:
        if not hardware_info.supports_fp8:
            msg = f"FP8 requested but not supported on {hardware_info.device_name}"
            if config.strict_mode:
                raise RuntimeError(msg)
            if config.warn_on_fallback:
                warnings.warn(f"{msg}. Falling back to BF16 or FP16.")
            # Try BF16 fallback
            if hardware_info.supports_bf16:
                return PrecisionMode.BF16
            elif hardware_info.supports_fp16:
                return PrecisionMode.FP16
            return PrecisionMode.FP32
        return mode
    
    # Check BF16
    if mode == PrecisionMode.BF16:
        if not hardware_info.supports_bf16:
            msg = f"BF16 requested but not supported on {hardware_info.device_name}"
            if config.strict_mode:
                raise RuntimeError(msg)
            if config.warn_on_fallback:
                warnings.warn(f"{msg}. Falling back to FP16.")
            if hardware_info.supports_fp16:
                return PrecisionMode.FP16
            return PrecisionMode.FP32
        return mode
    
    # Check FP16
    if mode == PrecisionMode.FP16:
        if not hardware_info.supports_fp16:
            msg = f"FP16 requested but not supported on {hardware_info.device_name}"
            if config.strict_mode:
                raise RuntimeError(msg)
            if config.warn_on_fallback:
                warnings.warn(f"{msg}. Falling back to FP32.")
            return PrecisionMode.FP32
        return mode
    
    return mode


class PrecisionManager:
    """Manager for handling precision settings during training.
    
    Example:
        manager = PrecisionManager(mode="auto")
        
        # Get autocast context
        with manager.get_autocast_context():
            output = model(input)
            loss = criterion(output, target)
        
        # Scale loss for backward pass
        scaled_loss = manager.scale_loss(loss)
        scaled_loss.backward()
        
        # Unscale and step optimizer
        manager.unscale_and_step(optimizer)
    """
    
    def __init__(
        self,
        mode: str | PrecisionMode = "auto",
        config: PrecisionConfig | None = None,
    ):
        """Initialize precision manager.
        
        Args:
            mode: Precision mode ("auto", "fp8", "bf16", "fp16", "fp32")
            config: Optional detailed configuration
        """
        if isinstance(mode, str):
            mode = PrecisionMode(mode)
        
        self.config = config or PrecisionConfig(mode=mode)
        self.config.mode = mode
        
        self.hardware_info = detect_hardware_info()
        self.effective_mode = detect_optimal_precision(self.hardware_info, self.config)
        
        self._scaler = None
        self._setup_scaler()
        
        logger.info(
            f"PrecisionManager initialized: "
            f"requested={mode.value}, effective={self.effective_mode.value}, "
            f"device={self.hardware_info.device_name}"
        )
    
    def _setup_scaler(self):
        """Set up gradient scaler for FP16 training."""
        torch = _try_import_torch()
        if torch is None:
            return
        
        if self.effective_mode == PrecisionMode.FP16 and self.hardware_info.is_cuda:
            self._scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.fp16_initial_scale,
                growth_factor=self.config.fp16_growth_factor,
                backoff_factor=self.config.fp16_backoff_factor,
                growth_interval=self.config.fp16_growth_interval,
                enabled=True,
            )
    
    def get_autocast_context(self):
        """Get autocast context manager for mixed precision.
        
        Returns:
            Context manager for autocast
        """
        torch = _try_import_torch()
        if torch is None:
            import contextlib
            return contextlib.nullcontext()
        
        if self.effective_mode == PrecisionMode.FP32:
            import contextlib
            return contextlib.nullcontext()
        
        if self.hardware_info.is_cuda:
            dtype = self._get_torch_dtype()
            return torch.cuda.amp.autocast(dtype=dtype)
        
        if self.hardware_info.is_mps:
            # MPS autocast
            dtype = self._get_torch_dtype()
            return torch.autocast(device_type='mps', dtype=dtype)
        
        import contextlib
        return contextlib.nullcontext()
    
    def _get_torch_dtype(self):
        """Get torch dtype for effective precision mode."""
        torch = _try_import_torch()
        if torch is None:
            return None
        
        dtype_map = {
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.FP32: torch.float32,
        }
        
        if self.effective_mode == PrecisionMode.FP8:
            # FP8 uses BF16 for compute (FP8 is applied at specific points)
            return torch.bfloat16
        
        return dtype_map.get(self.effective_mode, torch.float32)
    
    def scale_loss(self, loss):
        """Scale loss for mixed precision backward pass.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss (or original if scaling not needed)
        """
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss
    
    def unscale_and_step(self, optimizer, max_grad_norm: float | None = None):
        """Unscale gradients and step optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            max_grad_norm: Optional gradient clipping norm
        """
        torch = _try_import_torch()
        
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
            
            if max_grad_norm is not None and self.config.gradient_clipping_in_fp32:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_grad_norm,
                )
            
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_grad_norm,
                )
            optimizer.step()
    
    def get_precision_info(self) -> dict[str, Any]:
        """Get precision information for logging.
        
        Returns:
            Dictionary with precision details
        """
        return {
            "requested_mode": self.config.mode.value,
            "effective_mode": self.effective_mode.value,
            "device_name": self.hardware_info.device_name,
            "compute_capability": self.hardware_info.compute_capability,
            "supports_fp8": self.hardware_info.supports_fp8,
            "supports_bf16": self.hardware_info.supports_bf16,
            "supports_fp16": self.hardware_info.supports_fp16,
            "using_scaler": self._scaler is not None,
        }


def get_precision_config_for_hardware() -> dict[str, Any]:
    """Get recommended precision configuration for current hardware.
    
    Returns:
        Dictionary with recommended settings
    """
    hardware = detect_hardware_info()
    optimal = detect_optimal_precision(hardware)
    
    return {
        "precision": {
            "mode": optimal.value,
            "fp8": {
                "enabled": optimal == PrecisionMode.FP8,
                "format": "e4m3",
            },
            "bf16": {
                "enabled": optimal == PrecisionMode.BF16,
            },
            "fp16": {
                "enabled": optimal == PrecisionMode.FP16,
                "grad_scaler": True,
            },
        },
        "hardware": {
            "device": hardware.device_name,
            "compute_capability": hardware.compute_capability,
            "memory_gb": hardware.total_memory_gb,
        },
    }


if __name__ == "__main__":
    # Test precision detection
    import json
    
    print("=" * 60)
    print("Precision Detection Test")
    print("=" * 60)
    
    # Detect hardware
    hardware = detect_hardware_info()
    print(f"\nHardware Info:")
    print(f"  Device: {hardware.device_name}")
    print(f"  Compute Capability: {hardware.compute_capability}")
    print(f"  Memory: {hardware.total_memory_gb:.1f} GB")
    print(f"  Supports FP8: {hardware.supports_fp8}")
    print(f"  Supports BF16: {hardware.supports_bf16}")
    print(f"  Supports FP16: {hardware.supports_fp16}")
    
    # Detect optimal precision
    optimal = detect_optimal_precision()
    print(f"\nOptimal Precision: {optimal.value}")
    
    # Test precision manager
    print("\nPrecision Manager Test:")
    manager = PrecisionManager(mode="auto")
    print(json.dumps(manager.get_precision_info(), indent=2))
    
    # Get recommended config
    print("\nRecommended Configuration:")
    config = get_precision_config_for_hardware()
    print(json.dumps(config, indent=2))
