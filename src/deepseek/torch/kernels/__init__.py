"""
DeepSeek Custom Triton Kernels

This package provides optimized Triton kernels for DeepSeek:
- Fused MLA Attention (compress-attend-decompress)
- Fused SwiGLU Activation
- Fused RMSNorm with Residual Addition
- Fused Softmax with Online Normalization
- CUDA Feature Detection (TMA, Warpgroup, FP8)
- Architecture-specific Fused Kernels

These kernels provide significant speedups over PyTorch native operations
when Triton is available (requires CUDA GPU).

Usage:
    from deepseek.torch.kernels import (
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

from deepseek.torch.kernels.triton_kernels import (
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

# CUDA features and advanced fused kernels
from deepseek.torch.kernels.cuda_features import (
    CUDAFeatures,
    ComputeCapability,
    KernelBackend,
    select_backend,
    dispatch_kernel,
    TMADescriptor,
    WarpgroupConfig,
    FP8Format,
    FP8Config,
    fp8_matmul,
    KernelStats,
)

from deepseek.torch.kernels.fused_kernels import (
    softmax_fused,
    rms_norm_fused,
    attention_fused,
    matmul_fp16_tiled,
    MatmulConfig,
    FusedKernelDispatcher,
    get_dispatcher,
    TRITON_AVAILABLE as TRITON_KERNELS_AVAILABLE,
)

__all__ = [
    # Original triton kernels
    "TRITON_AVAILABLE",
    "fused_swiglu",
    "fused_swiglu_backward",
    "fused_rmsnorm",
    "fused_rmsnorm_residual",
    "fused_softmax",
    "fused_mla_attention",
    "get_kernel_autotuner",
    "KernelAutotuner",
    # CUDA features
    "CUDAFeatures",
    "ComputeCapability",
    "KernelBackend",
    "select_backend",
    "dispatch_kernel",
    "TMADescriptor",
    "WarpgroupConfig",
    "FP8Format",
    "FP8Config",
    "fp8_matmul",
    "KernelStats",
    # Fused kernels
    "softmax_fused",
    "rms_norm_fused",
    "attention_fused",
    "matmul_fp16_tiled",
    "MatmulConfig",
    "FusedKernelDispatcher",
    "get_dispatcher",
    "TRITON_KERNELS_AVAILABLE",
]
