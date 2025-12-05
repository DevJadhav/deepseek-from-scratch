"""
ANE-Compatible Activation Functions

This module implements ANE-optimized activation functions:
- ANESiLU: Sigmoid Linear Unit (SiLU/Swish)
- ANEGELU: Gaussian Error Linear Unit
- ANESwiGLU: Swish-Gated Linear Unit (used in LLaMA/DeepSeek FFN)

These activations are optimized for:
- FP16 computation for ANE efficiency
- In-place operations where safe
- Minimal memory allocation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ANESiLU(nn.Module):
    """
    ANE-optimized Sigmoid Linear Unit (SiLU/Swish).

    SiLU(x) = x * sigmoid(x)

    This is the default activation in modern transformers and is
    well-supported on ANE.

    Args:
        use_fp16: Use FP16 for computation (default True)
        inplace: Use inplace operation where possible (default False)
    """

    def __init__(self, use_fp16: bool = True, inplace: bool = False):
        super().__init__()
        self.use_fp16 = use_fp16
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation."""
        original_dtype = x.dtype

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Apply SiLU (x * sigmoid(x))
        x = F.silu(x, inplace=self.inplace)

        # Convert back if needed
        if x.dtype != original_dtype:
            x = x.to(original_dtype)

        return x


class ANEGELU(nn.Module):
    """
    ANE-optimized Gaussian Error Linear Unit (GELU).

    GELU(x) = x * Φ(x)

    Where Φ(x) is the cumulative distribution function of the
    standard normal distribution.

    Uses the tanh approximation for better ANE compatibility:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Args:
        use_fp16: Use FP16 for computation (default True)
        approximate: Use tanh approximation (default True, recommended for ANE)
    """

    def __init__(self, use_fp16: bool = True, approximate: bool = True):
        super().__init__()
        self.use_fp16 = use_fp16
        self.approximate = "tanh" if approximate else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        original_dtype = x.dtype

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Apply GELU with approximation
        x = F.gelu(x, approximate=self.approximate)

        # Convert back if needed
        if x.dtype != original_dtype:
            x = x.to(original_dtype)

        return x


class ANESwiGLU(nn.Module):
    """
    ANE-optimized Swish-Gated Linear Unit (SwiGLU).

    SwiGLU is used in LLaMA/DeepSeek FFN layers:
    SwiGLU(x) = SiLU(W1 @ x) * (W2 @ x)

    This combines the gate projection (W1) with SiLU activation
    and element-wise multiplication with the up projection (W2).

    This is a fused implementation that computes both projections
    together for efficiency.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension (typically 4 * in_features * 2/3 for efficiency)
        out_features: Output feature dimension (typically same as in_features)
        bias: Include bias terms (default False for transformer FFN)
        use_fp16: Use FP16 for computation (default True)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = False,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.use_fp16 = use_fp16

        # Gate and up projections (can be fused into single matmul)
        # W_gate and W_up: (in_features) -> (hidden_features)
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)

        # Down projection: (hidden_features) -> (out_features)
        self.down_proj = nn.Linear(hidden_features, self.out_features, bias=bias)

        # Activation
        self.act = ANESiLU(use_fp16=use_fp16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        original_dtype = x.dtype

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Gate projection with SiLU activation
        gate = self.act(self.gate_proj(x))

        # Up projection
        up = self.up_proj(x)

        # Element-wise product
        hidden = gate * up

        # Down projection
        output = self.down_proj(hidden)

        # Convert back if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}"
        )


class ANEFusedSwiGLU(nn.Module):
    """
    Fused SwiGLU that combines gate and up projections into a single matmul.

    This is more efficient when both projections have the same input:
    [gate, up] = x @ [W_gate; W_up].T

    Then: output = down_proj(SiLU(gate) * up)

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension per gate (total hidden = 2 * hidden_features)
        out_features: Output feature dimension
        bias: Include bias terms (default False)
        use_fp16: Use FP16 for computation (default True)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = False,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.use_fp16 = use_fp16

        # Fused gate+up projection: (in_features) -> (2 * hidden_features)
        self.gate_up_proj = nn.Linear(
            in_features, 2 * hidden_features, bias=bias
        )

        # Down projection: (hidden_features) -> (out_features)
        self.down_proj = nn.Linear(hidden_features, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fused SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        original_dtype = x.dtype

        # Convert to FP16 for ANE efficiency
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Fused gate+up projection
        gate_up = self.gate_up_proj(x)

        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)

        # Apply SiLU to gate and multiply with up
        hidden = F.silu(gate) * up

        # Down projection
        output = self.down_proj(hidden)

        # Convert back if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}"
        )
