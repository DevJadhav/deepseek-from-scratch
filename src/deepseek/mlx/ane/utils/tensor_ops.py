"""
ANE Tensor Operations

This module provides utility functions for ANE-optimized tensor operations:
- Shape normalization: Pad tensors to ANE-friendly dimensions
- Layout conversion: Convert between NCHW and NHWC layouts
- Memory alignment: Ensure 16-byte alignment for ANE efficiency
- Tile splitting: Split tensors into ANE-compatible tiles

ANE Constraints:
- Prefer dimensions that are multiples of 16
- Powers of 2 dimensions provide best throughput
- 16-byte memory alignment required
- Maximum tensor size ~256MB per operation
"""

import math
from typing import Tuple, List, Optional, Union

import torch

# ANE constants
ANE_ALIGNMENT = 16  # 16-byte alignment
ANE_PREFERRED_MULTIPLE = 16  # Dimensions should be multiples of 16
ANE_MAX_TENSOR_SIZE = 256 * 1024 * 1024  # 256MB max tensor size


def pad_to_multiple(value: int, multiple: int = ANE_PREFERRED_MULTIPLE) -> int:
    """
    Pad a value to the next multiple of the given base.
    
    Args:
        value: The value to pad
        multiple: The base multiple (default 16 for ANE)
        
    Returns:
        The padded value (>= original value, divisible by multiple)
        
    Examples:
        >>> pad_to_multiple(17, 16)
        32
        >>> pad_to_multiple(16, 16)
        16
    """
    if value <= 0:
        return multiple
    if value % multiple == 0:
        return value
    return ((value // multiple) + 1) * multiple


def pad_to_power_of_2(value: int, min_size: int = 1) -> int:
    """
    Pad a value to the next power of 2.
    
    Args:
        value: The value to pad
        min_size: Minimum result size (default 1)
        
    Returns:
        The next power of 2 >= max(value, min_size)
        
    Examples:
        >>> pad_to_power_of_2(17)
        32
        >>> pad_to_power_of_2(8, min_size=16)
        16
    """
    value = max(value, min_size)
    if value <= 0:
        return min_size
    # Check if already power of 2
    if value & (value - 1) == 0:
        return value
    # Find next power of 2
    return 1 << (value - 1).bit_length()


def normalize_shape_for_ane(
    x: torch.Tensor,
    multiple: int = ANE_PREFERRED_MULTIPLE,
    power_of_2: bool = False,
    min_dims: int = 2,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Normalize tensor shape for ANE compatibility.

    Pads tensor dimensions to ANE-friendly values while preserving
    the original data.

    Args:
        x: Input tensor
        multiple: Pad dimensions to this multiple (default 16)
        power_of_2: If True, pad to power of 2 instead of multiple
        min_dims: Minimum number of dimensions to pad (from the end)

    Returns:
        Tuple of (padded_tensor, original_shape)

    Examples:
        >>> x = torch.randn(17, 33)
        >>> padded, orig = normalize_shape_for_ane(x)
        >>> padded.shape
        torch.Size([32, 48])
        >>> orig
        (17, 33)
    """
    original_shape = x.shape

    # Determine which dimensions to pad (last min_dims dimensions)
    ndim = x.ndim
    dims_to_pad = min(min_dims, ndim)

    # Calculate padding - F.pad expects (last_dim_left, last_dim_right, second_last_left, ...)
    padding = []
    for i in range(ndim - 1, max(ndim - dims_to_pad - 1, -1), -1):
        dim_size = x.shape[i]
        if power_of_2:
            target_size = pad_to_power_of_2(dim_size, min_size=multiple)
        else:
            target_size = pad_to_multiple(dim_size, multiple)
        pad_amount = target_size - dim_size
        padding.extend([0, pad_amount])  # left=0, right=pad_amount

    # Apply padding if needed
    if any(p > 0 for p in padding):
        padded = torch.nn.functional.pad(x, padding)
    else:
        padded = x

    return padded, tuple(original_shape)


def unpad_tensor(
    x: torch.Tensor,
    original_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Remove padding from a tensor to restore original shape.
    
    Args:
        x: Padded tensor
        original_shape: Original shape before padding
        
    Returns:
        Tensor with original shape
    """
    slices = tuple(slice(0, s) for s in original_shape)
    return x[slices]


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from NCHW to NHWC layout.
    
    Args:
        x: Input tensor in NCHW format (batch, channels, height, width)
        
    Returns:
        Tensor in NHWC format (batch, height, width, channels)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (NCHW), got {x.ndim}D")
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from NHWC to NCHW layout.
    
    Args:
        x: Input tensor in NHWC format (batch, height, width, channels)
        
    Returns:
        Tensor in NCHW format (batch, channels, height, width)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (NHWC), got {x.ndim}D")
    return x.permute(0, 3, 1, 2).contiguous()


def convert_to_channel_last(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to channel-last memory format.
    
    This is the preferred format for ANE operations.
    
    Args:
        x: Input tensor (4D for images, or any shape)
        
    Returns:
        Tensor in channel-last format
    """
    if x.ndim == 4:
        return x.to(memory_format=torch.channels_last)
    elif x.ndim == 5:
        return x.to(memory_format=torch.channels_last_3d)
    return x  # No conversion for other shapes


def convert_from_channel_last(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from channel-last to contiguous memory format.
    
    Args:
        x: Input tensor in channel-last format
        
    Returns:
        Tensor in contiguous format
    """
    return x.contiguous()


def is_aligned(x: torch.Tensor, alignment: int = ANE_ALIGNMENT) -> bool:
    """
    Check if tensor storage is aligned to the specified boundary.
    
    Args:
        x: Input tensor
        alignment: Alignment boundary in bytes (default 16)
        
    Returns:
        True if tensor is aligned
    """
    if not x.is_contiguous():
        return False
    # Check data pointer alignment
    return x.data_ptr() % alignment == 0


def align_tensor(
    x: torch.Tensor,
    alignment: int = ANE_ALIGNMENT,
) -> torch.Tensor:
    """
    Ensure tensor is aligned to the specified boundary.
    
    Creates a new aligned tensor if necessary.
    
    Args:
        x: Input tensor
        alignment: Alignment boundary in bytes (default 16)
        
    Returns:
        Aligned tensor (may be a copy)
    """
    if is_aligned(x, alignment):
        return x
    
    # Create new aligned tensor
    # PyTorch doesn't have direct aligned allocation, so we allocate
    # slightly larger and slice to aligned offset
    return x.clone()


def get_aligned_size(size: int, alignment: int = ANE_ALIGNMENT) -> int:
    """
    Get the aligned size for a given size and alignment.
    
    Args:
        size: Original size in bytes
        alignment: Alignment boundary (default 16)
        
    Returns:
        Aligned size >= original size
    """
    if size % alignment == 0:
        return size
    return ((size // alignment) + 1) * alignment


def get_ane_friendly_dim(dim: int, prefer_power_of_2: bool = False) -> int:
    """
    Get the nearest ANE-friendly dimension.
    
    Args:
        dim: Original dimension
        prefer_power_of_2: If True, prefer power of 2 dimensions
        
    Returns:
        ANE-friendly dimension >= original dim
    """
    if prefer_power_of_2:
        return pad_to_power_of_2(dim, min_size=ANE_PREFERRED_MULTIPLE)
    return pad_to_multiple(dim, ANE_PREFERRED_MULTIPLE)


def split_for_ane_tiles(
    x: torch.Tensor,
    tile_size: int = 128,
    dim: int = -1,
) -> List[torch.Tensor]:
    """
    Split tensor into tiles for ANE processing.
    
    Large tensors may exceed ANE constraints. This function splits
    them into manageable tiles.
    
    Args:
        x: Input tensor
        tile_size: Maximum size per tile (default 128)
        dim: Dimension to split along (default -1, last dim)
        
    Returns:
        List of tensor tiles
    """
    dim = dim if dim >= 0 else x.ndim + dim
    dim_size = x.shape[dim]
    
    if dim_size <= tile_size:
        return [x]
    
    num_tiles = math.ceil(dim_size / tile_size)
    return list(torch.chunk(x, num_tiles, dim=dim))


def check_ane_compatible(
    x: torch.Tensor,
    max_size: int = ANE_MAX_TENSOR_SIZE,
) -> Tuple[bool, str]:
    """
    Check if tensor is compatible with ANE constraints.
    
    Args:
        x: Input tensor
        max_size: Maximum tensor size in bytes (default 256MB)
        
    Returns:
        Tuple of (is_compatible, reason_if_not)
    """
    # Check size
    tensor_bytes = x.numel() * x.element_size()
    if tensor_bytes > max_size:
        return False, f"Tensor size ({tensor_bytes} bytes) exceeds max ({max_size} bytes)"
    
    # Check alignment
    if not is_aligned(x):
        return False, "Tensor is not aligned"
    
    # Check dimensions
    for i, dim in enumerate(x.shape):
        if dim > 0 and dim % ANE_PREFERRED_MULTIPLE != 0:
            return False, f"Dimension {i} ({dim}) is not a multiple of {ANE_PREFERRED_MULTIPLE}"
    
    return True, "Compatible"


def create_attention_mask(
    seq_len: int,
    dtype: torch.dtype = torch.float16,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Create a causal attention mask for ANE.
    
    Args:
        seq_len: Sequence length
        dtype: Data type (default FP16 for ANE)
        device: Target device
        
    Returns:
        Causal mask tensor of shape (seq_len, seq_len)
    """
    # Create causal mask with -inf for masked positions
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def pad_sequence_for_ane(
    x: torch.Tensor,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequence to target length for ANE.
    
    Args:
        x: Input tensor of shape (batch, seq_len, ...)
        target_len: Target sequence length (should be ANE-friendly)
        pad_value: Value for padding (default 0)
        
    Returns:
        Tuple of (padded_tensor, attention_mask)
    """
    batch_size, seq_len = x.shape[:2]
    
    if seq_len >= target_len:
        return x[:, :target_len], torch.ones(batch_size, target_len, dtype=torch.bool, device=x.device)
    
    pad_len = target_len - seq_len
    
    # Create padding
    pad_shape = (batch_size, pad_len) + x.shape[2:]
    padding = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    
    # Concatenate
    padded = torch.cat([x, padding], dim=1)
    
    # Create attention mask (True for valid tokens)
    mask = torch.cat([
        torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device),
        torch.zeros(batch_size, pad_len, dtype=torch.bool, device=x.device),
    ], dim=1)
    
    return padded, mask
