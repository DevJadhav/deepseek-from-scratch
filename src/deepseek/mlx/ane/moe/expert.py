"""
ANE-Optimized Expert Module

This module implements ANE-optimized FFN experts with:
- FP16 computation for ANE acceleration
- INT8 weight quantization support
- SwiGLU activation (DeepSeek style)
- Tiled computation for large hidden dimensions
- Memory-efficient weight sharing
"""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.base import ANELinear


class ActivationType(Enum):
    """Activation function types for experts."""

    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
    RELU = "relu"


@dataclass
class ANEExpertConfig:
    """Configuration for ANE-optimized expert."""

    d_model: int = 4096
    d_hidden: int = 2048  # Fine-grained intermediate (DeepSeek-V3 uses 2048)
    activation: ActivationType = ActivationType.SWIGLU
    use_fp16: bool = True
    use_quantized_weights: bool = False
    dropout: float = 0.0
    bias: bool = False

    @classmethod
    def for_routed_expert(
        cls,
        d_model: int = 4096,
        hidden_mult: float = 0.5,
    ) -> "ANEExpertConfig":
        """Create config for routed expert (fine-grained, smaller)."""
        return cls(
            d_model=d_model,
            d_hidden=int(d_model * hidden_mult),
            activation=ActivationType.SWIGLU,
        )

    @classmethod
    def for_shared_expert(
        cls,
        d_model: int = 4096,
        hidden_mult: float = 4.0,
    ) -> "ANEExpertConfig":
        """Create config for shared expert (larger capacity)."""
        return cls(
            d_model=d_model,
            d_hidden=int(d_model * hidden_mult),
            activation=ActivationType.SWIGLU,
        )


class ANEExpert(nn.Module):
    """
    ANE-Optimized Feed-Forward Expert.

    Architecture (SwiGLU):
    ┌─────────────────────────────────────────────────────────────────┐
    │ ANE Expert (SwiGLU)                                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Input x ──┬──► Gate Linear ──► SiLU ──┐                       │
    │            │                            ├──► Element-wise ──►   │
    │            └──► Up Linear ─────────────┘    Multiply           │
    │                                                                 │
    │                                        ──► Down Linear ──► Out │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Features:
    - FP16 computation for ANE
    - INT8 weight quantization (optional)
    - SwiGLU activation (default) or GELU/SiLU/ReLU
    - Dropout for regularization
    """

    def __init__(self, config: ANEExpertConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_hidden = config.d_hidden
        self.use_fp16 = config.use_fp16
        self.activation_type = config.activation

        LinearClass = ANELinear if config.use_quantized_weights else nn.Linear

        if config.activation == ActivationType.SWIGLU:
            # SwiGLU: gate and up projections
            self.gate_proj = LinearClass(
                config.d_model, config.d_hidden, bias=config.bias
            )
            self.up_proj = LinearClass(
                config.d_model, config.d_hidden, bias=config.bias
            )
        else:
            # Standard: single up projection
            self.up_proj = LinearClass(
                config.d_model, config.d_hidden, bias=config.bias
            )
            self.gate_proj = None

        self.down_proj = LinearClass(
            config.d_hidden, config.d_model, bias=config.bias
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Convert to FP16 if requested
        if config.use_fp16:
            self.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of expert.

        Args:
            x: Input tensor (batch, seq_len, d_model) or (num_tokens, d_model)

        Returns:
            Output tensor with same shape as input
        """
        # Convert to FP16 if needed
        original_dtype = x.dtype
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Apply activation
        if self.activation_type == ActivationType.SWIGLU:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
        elif self.activation_type == ActivationType.GELU:
            hidden = F.gelu(self.up_proj(x))
        elif self.activation_type == ActivationType.SILU:
            hidden = F.silu(self.up_proj(x))
        else:  # RELU
            hidden = F.relu(self.up_proj(x))

        # Down projection
        output = self.down_proj(hidden)

        # Dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # Convert back to original dtype if needed
        if self.use_fp16 and original_dtype != torch.float16:
            output = output.to(original_dtype)

        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"activation={self.activation_type.value}"
        )


class ANESharedExpert(nn.Module):
    """
    ANE-Optimized Shared Expert.

    The shared expert processes ALL tokens, not just routed ones.
    It typically has a larger hidden dimension for more capacity.

    Features:
    - Always active (no routing)
    - Larger capacity than routed experts
    - Can be multiple shared experts averaged together
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_hidden: int = 16384,
        num_shared: int = 1,
        use_fp16: bool = True,
        use_quantized_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_shared = num_shared
        self.use_fp16 = use_fp16

        config = ANEExpertConfig(
            d_model=d_model,
            d_hidden=d_hidden,
            use_fp16=use_fp16,
            use_quantized_weights=use_quantized_weights,
        )

        self.experts = nn.ModuleList(
            [ANEExpert(config) for _ in range(num_shared)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all shared experts.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Summed output from all shared experts
        """
        output = torch.zeros_like(x)
        for expert in self.experts:
            output = output + expert(x)
        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"num_shared={self.num_shared}"
        )


class ANEExpertGroup(nn.Module):
    """
    ANE-Optimized Expert Group for Hierarchical Routing.

    Groups multiple experts together for efficient batch processing.
    Used in DeepSeek-V3's hierarchical routing (8 groups of 32 experts each).

    Features:
    - Batched expert computation within group
    - Efficient memory access patterns
    - Support for expert consolidation
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_hidden: int = 2048,
        num_experts_in_group: int = 32,
        use_fp16: bool = True,
        use_quantized_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts_in_group
        self.use_fp16 = use_fp16

        config = ANEExpertConfig(
            d_model=d_model,
            d_hidden=d_hidden,
            use_fp16=use_fp16,
            use_quantized_weights=use_quantized_weights,
        )

        self.experts = nn.ModuleList(
            [ANEExpert(config) for _ in range(num_experts_in_group)]
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with batched expert computation.

        Args:
            x: Input tensor (num_tokens, d_model)
            expert_indices: Expert indices for each token (num_tokens,)
                          Values should be 0 to num_experts-1 (local to group)
            expert_weights: Gating weights for each token (num_tokens,)

        Returns:
            Weighted expert outputs (num_tokens, d_model)
        """
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)

        # Process each expert in the group
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = expert_indices == expert_idx

            if mask.any():
                # Get tokens for this expert
                expert_input = x[mask]

                # Compute expert output
                expert_output = self.experts[expert_idx](expert_input)

                # Weight and accumulate
                weights = expert_weights[mask].unsqueeze(-1)
                output[mask] = output[mask] + expert_output * weights

        return output

    def forward_fused(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        top_k: int = 1,
    ) -> torch.Tensor:
        """
        Fused forward pass for top-k experts per token.

        More efficient when tokens are routed to multiple experts
        within the same group.

        Args:
            x: Input tensor (num_tokens, d_model)
            expert_indices: Expert indices (num_tokens, top_k)
            expert_weights: Gating weights (num_tokens, top_k)
            top_k: Number of experts per token

        Returns:
            Summed weighted expert outputs (num_tokens, d_model)
        """
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)

        for k in range(top_k):
            indices_k = expert_indices[:, k]
            weights_k = expert_weights[:, k]

            for expert_idx in range(self.num_experts):
                mask = indices_k == expert_idx

                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    weights = weights_k[mask].unsqueeze(-1)
                    output[mask] = output[mask] + expert_output * weights

        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"num_experts={self.num_experts}"
        )


class ANEFusedExpert(nn.Module):
    """
    ANE-Optimized Fused Expert for Expert Fusion.

    Instead of running multiple experts and combining,
    this pre-combines expert weights for a single pass.
    Best for small expert counts (< 16).

    The fusion formula:
        output = Σᵢ(wᵢ × Expertᵢ(x))
               ≈ FusedExpert(x, combined_weights)

    This is an approximation that trades accuracy for efficiency
    when dynamic fusion weights are computed.
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_hidden: int = 2048,
        num_experts: int = 8,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.use_fp16 = use_fp16

        # Store expert weights as 3D tensors for efficient fusion
        # Shape: (num_experts, d_hidden, d_model) for gate/up
        # Shape: (num_experts, d_model, d_hidden) for down
        self.gate_weights = nn.Parameter(
            torch.randn(num_experts, d_hidden, d_model) * 0.02
        )
        self.up_weights = nn.Parameter(
            torch.randn(num_experts, d_hidden, d_model) * 0.02
        )
        self.down_weights = nn.Parameter(
            torch.randn(num_experts, d_model, d_hidden) * 0.02
        )

        if use_fp16:
            self.half()

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fused expert forward with dynamic weight combination.

        Args:
            x: Input tensor (batch, seq_len, d_model) or (num_tokens, d_model)
            expert_weights: Weights for each expert (num_experts,) or
                          (batch, seq_len, num_experts)

        Returns:
            Fused output tensor
        """
        original_shape = x.shape
        original_dtype = x.dtype

        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Handle different weight shapes
        if expert_weights.dim() == 1:
            # Same weights for all tokens: (num_experts,)
            # Combine expert weights: (num_experts, d_hidden, d_model) -> (d_hidden, d_model)
            combined_gate = torch.einsum(
                "e,ehd->hd", expert_weights, self.gate_weights
            )
            combined_up = torch.einsum(
                "e,ehd->hd", expert_weights, self.up_weights
            )
            combined_down = torch.einsum(
                "e,edh->dh", expert_weights, self.down_weights
            )

            # Apply fused computation
            # x: (..., d_model)
            gate = F.silu(F.linear(x, combined_gate))
            up = F.linear(x, combined_up)
            hidden = gate * up
            output = F.linear(hidden, combined_down)

        else:
            # Per-token weights: (batch, seq_len, num_experts) or (num_tokens, num_experts)
            # This requires per-token weight combination - less efficient but more flexible
            x_flat = x.view(-1, self.d_model)
            expert_weights_flat = expert_weights.view(-1, self.num_experts)

            # Process in batches to avoid OOM
            outputs = []
            for i in range(x_flat.shape[0]):
                w = expert_weights_flat[i]  # (num_experts,)
                xi = x_flat[i:i + 1]  # (1, d_model)

                combined_gate = torch.einsum("e,ehd->hd", w, self.gate_weights)
                combined_up = torch.einsum("e,ehd->hd", w, self.up_weights)
                combined_down = torch.einsum("e,edh->dh", w, self.down_weights)

                gate = F.silu(F.linear(xi, combined_gate))
                up = F.linear(xi, combined_up)
                hidden = gate * up
                out = F.linear(hidden, combined_down)
                outputs.append(out)

            output = torch.cat(outputs, dim=0)
            output = output.view(original_shape)

        if self.use_fp16 and original_dtype != torch.float16:
            output = output.to(original_dtype)

        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"num_experts={self.num_experts}"
        )
