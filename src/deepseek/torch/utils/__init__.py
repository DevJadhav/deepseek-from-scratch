"""PyTorch utilities for DeepSeek.

Provides:
- Device selection with CUDA → MPS → CPU priority
- Auto batch size optimization using binary search
- Logging, checkpointing, and other utilities
"""

from deepseek.torch.utils.device import (
    AutoBatchConfig,
    AutoBatchSizer,
    DeviceConfig,
    DevicePriority,
    device_type_string,
    get_device,
    get_device_info,
    get_device_with_config,
    is_cuda_available,
    is_mps_available,
)

__all__ = [
    # Device selection
    "DevicePriority",
    "DeviceConfig",
    "AutoBatchConfig",
    "AutoBatchSizer",
    "get_device",
    "get_device_with_config",
    "get_device_info",
    "device_type_string",
    "is_cuda_available",
    "is_mps_available",
]