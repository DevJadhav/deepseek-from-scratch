"""
Unified Device Selection for DeepSeek PyTorch

This module provides centralized device selection with CUDA → MPS → CPU priority chain.

Features:
- Automatic device detection with consistent priority (CUDA first)
- Configurable device selection via environment variables
- Binary search auto batch size optimization
- Device information and availability checking

Usage:
    from deepseek.torch.utils.device import get_device, DeviceConfig, AutoBatchSizer
    
    # Get best available device (CUDA → MPS → CPU)
    device = get_device()
    
    # With custom configuration
    config = DeviceConfig.from_env()
    device = get_device_with_config(config)
    
    # Auto batch sizing
    sizer = AutoBatchSizer(AutoBatchConfig(max_memory_mb=8000))
    optimal_batch = sizer.find_optimal_batch_size_binary(memory_per_sample_mb=100)
"""

import os
from dataclasses import dataclass
from enum import Enum

import torch

from deepseek.torch.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Device Priority Enum
# =============================================================================

class DevicePriority(Enum):
    """Device selection priority."""
    CUDA_FIRST = "cuda_first"  # CUDA → MPS → CPU (recommended for training)
    MPS_FIRST = "mps_first"    # MPS → CUDA → CPU (for Apple Silicon optimization)
    CPU_ONLY = "cpu_only"      # Always use CPU


# =============================================================================
# Device Configuration
# =============================================================================

@dataclass
class DeviceConfig:
    """Configuration for device selection.
    
    Attributes:
        priority: Device selection priority order
        cuda_device_id: CUDA device ID for multi-GPU systems
        mps_device_id: MPS device ID (typically 0)
        min_batch_size: Minimum batch size for auto-adjustment
        max_batch_size: Maximum batch size for auto-adjustment
    """
    priority: DevicePriority = DevicePriority.CUDA_FIRST
    cuda_device_id: int = 0
    mps_device_id: int = 0
    min_batch_size: int = 1
    max_batch_size: int = 256
    
    @classmethod
    def from_env(cls) -> "DeviceConfig":
        """Create configuration from environment variables.
        
        Environment variables:
            DEEPSEEK_DEVICE_PRIORITY: "cuda", "mps", or "cpu"
            DEEPSEEK_CUDA_DEVICE: CUDA device ID (default: 0)
            DEEPSEEK_MIN_BATCH_SIZE: Minimum batch size (default: 1)
            DEEPSEEK_MAX_BATCH_SIZE: Maximum batch size (default: 256)
        """
        priority_str = os.environ.get("DEEPSEEK_DEVICE_PRIORITY", "cuda").lower()
        if priority_str == "mps":
            priority = DevicePriority.MPS_FIRST
        elif priority_str == "cpu":
            priority = DevicePriority.CPU_ONLY
        else:
            priority = DevicePriority.CUDA_FIRST
        
        try:
            cuda_device_id = int(os.environ.get("DEEPSEEK_CUDA_DEVICE", "0"))
        except ValueError:
            cuda_device_id = 0
        
        try:
            min_batch_size = int(os.environ.get("DEEPSEEK_MIN_BATCH_SIZE", "1"))
        except ValueError:
            min_batch_size = 1
        
        try:
            max_batch_size = int(os.environ.get("DEEPSEEK_MAX_BATCH_SIZE", "256"))
        except ValueError:
            max_batch_size = 256
        
        return cls(
            priority=priority,
            cuda_device_id=cuda_device_id,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
    
    def get_priority_order(self) -> list[str]:
        """Get the priority order as a list of device type names."""
        if self.priority == DevicePriority.CUDA_FIRST:
            return ["cuda", "mps", "cpu"]
        elif self.priority == DevicePriority.MPS_FIRST:
            return ["mps", "cuda", "cpu"]
        else:
            return ["cpu"]


# =============================================================================
# Auto Batch Size Configuration
# =============================================================================

@dataclass
class AutoBatchConfig:
    """Configuration for automatic batch size adjustment.
    
    Attributes:
        max_memory_mb: Maximum memory budget in MB
        memory_fraction: Fraction of max memory to use (0.0-1.0)
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        auto_adjust: Whether to enable auto-adjustment
    """
    max_memory_mb: float = 8000  # 8 GB default
    memory_fraction: float = 0.9
    min_batch_size: int = 1
    max_batch_size: int = 256
    auto_adjust: bool = True


class AutoBatchSizer:
    """Automatic batch size optimizer using binary search.
    
    Uses binary search to find the optimal batch size that fits within
    the memory budget. Faster convergence than linear adjustment.
    
    Example:
        config = AutoBatchConfig(max_memory_mb=8000)
        sizer = AutoBatchSizer(config)
        optimal = sizer.find_optimal_batch_size_binary(memory_per_sample_mb=100)
    """
    
    def __init__(self, config: AutoBatchConfig | None = None):
        self.config = config or AutoBatchConfig()
    
    def find_optimal_batch_size_binary(self, memory_per_sample_mb: float) -> int:
        """Find optimal batch size using binary search.
        
        Args:
            memory_per_sample_mb: Memory required per sample in MB
            
        Returns:
            The optimal batch size within configured bounds
        """
        if not self.config.auto_adjust:
            return self.config.max_batch_size
        
        budget_mb = self.config.max_memory_mb * self.config.memory_fraction
        
        # Binary search for optimal batch size
        low = self.config.min_batch_size
        high = self.config.max_batch_size
        best = self.config.min_batch_size
        
        while low <= high:
            mid = low + (high - low) // 2
            total_memory = mid * memory_per_sample_mb
            
            if total_memory <= budget_mb:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return max(self.config.min_batch_size, min(best, self.config.max_batch_size))
    
    def find_optimal_batch_size_binary_with_default(
        self,
        memory_per_sample_mb: float,
        default_batch_size: int,
    ) -> int:
        """Find optimal batch size with fallback default.
        
        Args:
            memory_per_sample_mb: Memory required per sample in MB
            default_batch_size: Batch size to return if auto-adjust is disabled
            
        Returns:
            The optimal batch size or default if auto-adjust is disabled
        """
        if not self.config.auto_adjust:
            return default_batch_size
        return self.find_optimal_batch_size_binary(memory_per_sample_mb)


# =============================================================================
# Device Selection Functions
# =============================================================================

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(priority: DevicePriority = DevicePriority.CUDA_FIRST) -> torch.device:
    """Get the best available device with specified priority.
    
    Default priority: CUDA → MPS → CPU
    
    Args:
        priority: Device selection priority
        
    Returns:
        The best available torch.device
    """
    config = DeviceConfig(priority=priority)
    return get_device_with_config(config)


def get_device_with_config(config: DeviceConfig) -> torch.device:
    """Get device with custom configuration.
    
    Args:
        config: Device configuration
        
    Returns:
        The selected torch.device
    """
    if config.priority == DevicePriority.CPU_ONLY:
        logger.info("Using CPU (configured)")
        return torch.device("cpu")
    
    if config.priority == DevicePriority.CUDA_FIRST:
        return _get_cuda_first(config)
    else:
        return _get_mps_first(config)


def _get_cuda_first(config: DeviceConfig) -> torch.device:
    """CUDA → MPS → CPU priority chain."""
    if is_cuda_available():
        logger.info(f"Using CUDA GPU (device {config.cuda_device_id})")
        return torch.device(f"cuda:{config.cuda_device_id}")
    
    if is_mps_available():
        logger.info("Using MPS GPU - CUDA not available")
        return torch.device("mps")
    
    logger.info("Using CPU - no GPU available")
    return torch.device("cpu")


def _get_mps_first(config: DeviceConfig) -> torch.device:
    """MPS → CUDA → CPU priority chain."""
    if is_mps_available():
        logger.info("Using MPS GPU")
        return torch.device("mps")
    
    if is_cuda_available():
        logger.info(f"Using CUDA GPU (device {config.cuda_device_id}) - MPS not available")
        return torch.device(f"cuda:{config.cuda_device_id}")
    
    logger.info("Using CPU - no GPU available")
    return torch.device("cpu")


def get_device_info() -> dict[str, str]:
    """Get device availability information.
    
    Returns:
        Dictionary with device availability and selection info
    """
    cuda_available = is_cuda_available()
    mps_available = is_mps_available()
    
    if cuda_available:
        selected = "cuda"
    elif mps_available:
        selected = "mps"
    else:
        selected = "cpu"
    
    return {
        "cuda_available": str(cuda_available),
        "mps_available": str(mps_available),
        "selected_device": selected,
    }


def device_type_string(device: torch.device) -> str:
    """Get string representation of device type.
    
    Args:
        device: The torch device
        
    Returns:
        Device type as string ("cuda", "mps", or "cpu")
    """
    return device.type


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Enums
    "DevicePriority",
    # Config classes
    "DeviceConfig",
    "AutoBatchConfig",
    # Classes
    "AutoBatchSizer",
    # Functions
    "get_device",
    "get_device_with_config",
    "get_device_info",
    "device_type_string",
    "is_cuda_available",
    "is_mps_available",
]
