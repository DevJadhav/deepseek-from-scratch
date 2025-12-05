"""
DeepSeek Custom Triton Kernels

This package provides optimized Triton kernels for DeepSeek:
- Fused MLA Attention (compress-attend-decompress)
- Fused SwiGLU Activation
- Fused RMSNorm with Residual Addition
- Fused Softmax with Online Normalization

These kernels provide significant speedups over PyTorch native operations
when Triton is available (requires CUDA GPU).

Usage:
    from deepseek.kernels import (
        fused_swiglu,
        fused_rmsnorm,
        fused_rmsnorm_residual,
        fused_softmax,
        fused_mla_attention,
        TRITON_AVAILABLE,
    )
    
    # Check availability
    if TRITON_AVAILABLE:
        output = fused_swiglu(gate, up)
    else:
        # Fallback to native ops
        output = F.silu(gate) * up
"""

from deepseek.kernels.triton_kernels import (
    TRITON_AVAILABLE,
    fused_swiglu,
    fused_swiglu_backward,
    fused_rmsnorm,
    fused_rmsnorm_residual,
    fused_softmax,
    fused_mla_attention,
    get_kernel_autotuner,
    KernelAutotuner,
)

__all__ = [
    "TRITON_AVAILABLE",
    "fused_swiglu",
    "fused_swiglu_backward",
    "fused_rmsnorm",
    "fused_rmsnorm_residual",
    "fused_softmax",
    "fused_mla_attention",
    "get_kernel_autotuner",
    "KernelAutotuner",
]
