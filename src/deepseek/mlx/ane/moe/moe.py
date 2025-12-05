"""
ANE-Optimized Mixture of Experts (MoE) Layer

This module implements ANE-optimized MoE layers with:
- Expert fusion for small expert counts (< 16)
- Batched expert processing for medium counts (16-64)
- Hierarchical routing for large counts (256)
- FP16 computation and INT8 weights
- Auxiliary-loss-free load balancing
"""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert import (
    ANEExpert,
    ANEExpertConfig,
    ANESharedExpert,
    ANEExpertGroup,
    ANEFusedExpert,
    ActivationType,
)
from .router import (
    ANERouter,
    ANERouterConfig,
    ANEHierarchicalRouter,
    ANEHierarchicalRouterConfig,
    RoutingStrategy,
)


class MoEStrategy(Enum):
    """MoE computation strategy."""

    FUSED = "fused"  # Expert fusion (< 16 experts)
    BATCHED = "batched"  # Batched processing (16-64 experts)
    HIERARCHICAL = "hierarchical"  # Hierarchical routing (256 experts)


@dataclass
class ANEMoEConfig:
    """Configuration for ANE-optimized MoE layer."""

    d_model: int = 4096
    num_routed_experts: int = 32
    num_shared_experts: int = 1
    top_k: int = 4
    num_groups: int = 8  # For hierarchical routing
    # Expert dimensions
    routed_hidden_mult: float = 0.5  # Fine-grained (2048 for d_model=4096)
    shared_hidden_mult: float = 4.0  # Larger (16384 for d_model=4096)
    # ANE optimization
    use_fp16: bool = True
    use_quantized_weights: bool = False
    # Strategy selection
    strategy: MoEStrategy | None = None  # Auto-select if None
    # Load balancing
    use_bias_adjustment: bool = True
    routing_strategy: RoutingStrategy = RoutingStrategy.SIGMOID
    # Dropout
    dropout: float = 0.0

    @property
    def routed_d_hidden(self) -> int:
        return int(self.d_model * self.routed_hidden_mult)

    @property
    def shared_d_hidden(self) -> int:
        return int(self.d_model * self.shared_hidden_mult)

    @property
    def experts_per_group(self) -> int:
        return self.num_routed_experts // self.num_groups

    def get_strategy(self) -> MoEStrategy:
        """Auto-select strategy based on expert count."""
        if self.strategy is not None:
            return self.strategy

        if self.num_routed_experts <= 16:
            return MoEStrategy.FUSED
        elif self.num_routed_experts <= 64:
            return MoEStrategy.BATCHED
        else:
            return MoEStrategy.HIERARCHICAL

    @classmethod
    def small_8_2(cls) -> "ANEMoEConfig":
        """Small config for testing: 8 experts, top-2."""
        return cls(
            d_model=256,
            num_routed_experts=8,
            num_shared_experts=1,
            top_k=2,
            num_groups=2,
            routed_hidden_mult=0.5,
            shared_hidden_mult=2.0,
        )

    @classmethod
    def medium_32_4(cls) -> "ANEMoEConfig":
        """Medium config: 32 experts, top-4."""
        return cls(
            d_model=2048,
            num_routed_experts=32,
            num_shared_experts=2,
            top_k=4,
            num_groups=4,
        )

    @classmethod
    def large_256_8(cls) -> "ANEMoEConfig":
        """Large config: 256 experts, top-8 (DeepSeek-V3 style)."""
        return cls(
            d_model=4096,
            num_routed_experts=256,
            num_shared_experts=1,
            top_k=8,
            num_groups=8,
            routing_strategy=RoutingStrategy.SIGMOID,
        )


class ANEMoE(nn.Module):
    """
    ANE-Optimized Mixture of Experts Layer.

    Automatically selects the best strategy based on expert count:
    - FUSED: Expert fusion for < 16 experts
    - BATCHED: Batched processing for 16-64 experts
    - HIERARCHICAL: Two-stage routing for 256 experts

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ ANE MoE Layer                                                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Input x ──┬──► Shared Expert(s) ────────────────┐             │
    │            │                                      │             │
    │            └──► Router ──► Top-K Selection ──►   │             │
    │                           │                       │             │
    │                           ▼                       │             │
    │                    Expert Computation              │             │
    │                    (Fused/Batched/Hier)           │             │
    │                           │                       │             │
    │                           ▼                       ▼             │
    │                    Routed Output    +     Shared Output         │
    │                           │                       │             │
    │                           └───────────┬───────────┘             │
    │                                       ▼                         │
    │                                   Output                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: ANEMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_routed_experts = config.num_routed_experts
        self.num_shared_experts = config.num_shared_experts
        self.top_k = config.top_k
        self.use_fp16 = config.use_fp16
        self.strategy = config.get_strategy()

        # Shared experts (always active)
        if config.num_shared_experts > 0:
            self.shared_experts = ANESharedExpert(
                d_model=config.d_model,
                d_hidden=config.shared_d_hidden,
                num_shared=config.num_shared_experts,
                use_fp16=config.use_fp16,
                use_quantized_weights=config.use_quantized_weights,
            )
        else:
            self.shared_experts = None

        # Initialize based on strategy
        if self.strategy == MoEStrategy.FUSED:
            self._init_fused(config)
        elif self.strategy == MoEStrategy.BATCHED:
            self._init_batched(config)
        else:  # HIERARCHICAL
            self._init_hierarchical(config)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def _init_fused(self, config: ANEMoEConfig):
        """Initialize fused expert computation."""
        # Router
        router_config = ANERouterConfig(
            d_model=config.d_model,
            num_experts=config.num_routed_experts,
            top_k=config.top_k,
            routing_strategy=config.routing_strategy,
            use_fp16=config.use_fp16,
            use_bias_adjustment=config.use_bias_adjustment,
        )
        self.router = ANERouter(router_config)

        # Fused expert
        self.fused_expert = ANEFusedExpert(
            d_model=config.d_model,
            d_hidden=config.routed_d_hidden,
            num_experts=config.num_routed_experts,
            use_fp16=config.use_fp16,
        )

        # Not used in fused mode
        self.experts = None
        self.expert_groups = None
        self.hierarchical_router = None

    def _init_batched(self, config: ANEMoEConfig):
        """Initialize batched expert computation."""
        # Router
        router_config = ANERouterConfig(
            d_model=config.d_model,
            num_experts=config.num_routed_experts,
            top_k=config.top_k,
            routing_strategy=config.routing_strategy,
            use_fp16=config.use_fp16,
            use_bias_adjustment=config.use_bias_adjustment,
        )
        self.router = ANERouter(router_config)

        # Individual experts
        expert_config = ANEExpertConfig(
            d_model=config.d_model,
            d_hidden=config.routed_d_hidden,
            use_fp16=config.use_fp16,
            use_quantized_weights=config.use_quantized_weights,
        )
        self.experts = nn.ModuleList([
            ANEExpert(expert_config)
            for _ in range(config.num_routed_experts)
        ])

        # Not used in batched mode
        self.fused_expert = None
        self.expert_groups = None
        self.hierarchical_router = None

    def _init_hierarchical(self, config: ANEMoEConfig):
        """Initialize hierarchical routing and expert groups."""
        # Hierarchical router
        hier_config = ANEHierarchicalRouterConfig(
            d_model=config.d_model,
            num_experts=config.num_routed_experts,
            num_groups=config.num_groups,
            top_k=config.top_k,
            top_k_groups=min(config.num_groups, config.top_k),
            routing_strategy=config.routing_strategy,
            use_fp16=config.use_fp16,
            use_bias_adjustment=config.use_bias_adjustment,
        )
        self.hierarchical_router = ANEHierarchicalRouter(hier_config)

        # Expert groups
        self.expert_groups = nn.ModuleList([
            ANEExpertGroup(
                d_model=config.d_model,
                d_hidden=config.routed_d_hidden,
                num_experts_in_group=config.experts_per_group,
                use_fp16=config.use_fp16,
                use_quantized_weights=config.use_quantized_weights,
            )
            for _ in range(config.num_groups)
        ])

        # Not used in hierarchical mode
        self.router = None
        self.fused_expert = None
        self.experts = None

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            Tuple of (output, aux_loss) where aux_loss is load imbalance
        """
        # Shared expert path
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x)
        else:
            shared_out = torch.zeros_like(x)

        # Routed expert path
        if self.strategy == MoEStrategy.FUSED:
            routed_out, aux_loss = self._forward_fused(x, return_aux_loss)
        elif self.strategy == MoEStrategy.BATCHED:
            routed_out, aux_loss = self._forward_batched(x, return_aux_loss)
        else:
            routed_out, aux_loss = self._forward_hierarchical(x, return_aux_loss)

        # Combine
        output = shared_out + routed_out

        # Dropout
        if self.dropout is not None:
            output = self.dropout(output)

        if return_aux_loss:
            return output, aux_loss
        return output, None

    def _forward_fused(
        self,
        x: torch.Tensor,
        return_aux_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward with expert fusion."""
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)

        # Get routing
        expert_indices, expert_weights, aux_loss = self.router(x, return_aux_loss)

        # Convert to full expert weights
        # expert_indices: (num_tokens, top_k)
        # expert_weights: (num_tokens, top_k)
        num_tokens = x_flat.shape[0]
        full_weights = torch.zeros(
            num_tokens, self.num_routed_experts,
            dtype=x_flat.dtype, device=x.device,
        )
        # Ensure expert_weights has same dtype as full_weights for scatter_add_
        expert_weights = expert_weights.to(dtype=full_weights.dtype)
        full_weights.scatter_add_(
            dim=1,
            index=expert_indices,
            src=expert_weights,
        )

        # Fused computation
        output = self.fused_expert(x_flat, full_weights)
        output = output.view(original_shape)

        return output, aux_loss

    def _forward_batched(
        self,
        x: torch.Tensor,
        return_aux_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward with batched expert computation."""
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        # Get routing
        expert_indices, expert_weights, aux_loss = self.router(x, return_aux_loss)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each top-k position
        for k in range(self.top_k):
            indices_k = expert_indices[:, k]  # (num_tokens,)
            weights_k = expert_weights[:, k]  # (num_tokens,)

            # Process each expert
            for expert_idx in range(self.num_routed_experts):
                mask = indices_k == expert_idx

                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    weights = weights_k[mask].unsqueeze(-1)
                    output[mask] = output[mask] + expert_output * weights

        output = output.view(original_shape)
        return output, aux_loss

    def _forward_hierarchical(
        self,
        x: torch.Tensor,
        return_aux_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward with hierarchical routing."""
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        # Get hierarchical routing
        expert_indices, expert_weights, group_indices = self.hierarchical_router.forward_batched(x)

        # expert_indices are global (0 to num_routed_experts-1)
        # Convert to group-local indices
        experts_per_group = self.config.experts_per_group

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert group
        for group_idx in range(self.config.num_groups):
            group_start = group_idx * experts_per_group
            group_end = group_start + experts_per_group

            # Find tokens routed to this group
            # A token is routed to this group if any of its expert_indices
            # falls within [group_start, group_end)
            in_group = (expert_indices >= group_start) & (expert_indices < group_end)
            tokens_in_group = in_group.any(dim=-1)  # (num_tokens,)

            if tokens_in_group.any():
                # Get tokens
                token_indices = torch.where(tokens_in_group)[0]
                group_x = x_flat[token_indices]

                # Get expert indices and weights for these tokens
                group_expert_indices = expert_indices[token_indices]
                group_expert_weights = expert_weights[token_indices]

                # Mask out indices not in this group
                local_mask = (group_expert_indices >= group_start) & (group_expert_indices < group_end)
                local_indices = torch.where(
                    local_mask,
                    group_expert_indices - group_start,
                    torch.zeros_like(group_expert_indices),
                )
                local_weights = torch.where(
                    local_mask,
                    group_expert_weights,
                    torch.zeros_like(group_expert_weights),
                )

                # Process through expert group
                group_output = self.expert_groups[group_idx].forward_fused(
                    group_x,
                    local_indices,
                    local_weights,
                    top_k=self.top_k,
                )

                # Accumulate to output
                output[token_indices] = output[token_indices] + group_output

        output = output.view(original_shape)

        # Compute aux loss (simple load imbalance)
        aux_loss = None
        if return_aux_loss:
            expert_counts = torch.bincount(
                expert_indices.view(-1),
                minlength=self.num_routed_experts,
            ).float()
            load_var = torch.var(expert_counts) / (expert_counts.mean() + 1e-8)
            aux_loss = load_var

        return output, aux_loss

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_routed_experts={self.num_routed_experts}, "
            f"num_shared_experts={self.num_shared_experts}, top_k={self.top_k}, "
            f"strategy={self.strategy.value}"
        )


class ANEMoEFused(ANEMoE):
    """
    ANE MoE with forced fused strategy.

    Best for small expert counts (< 16).
    """

    def __init__(self, config: ANEMoEConfig):
        config.strategy = MoEStrategy.FUSED
        super().__init__(config)


class ANEMoEBatched(ANEMoE):
    """
    ANE MoE with forced batched strategy.

    Best for medium expert counts (16-64).
    """

    def __init__(self, config: ANEMoEConfig):
        config.strategy = MoEStrategy.BATCHED
        super().__init__(config)


class ANEMoEHierarchical(ANEMoE):
    """
    ANE MoE with forced hierarchical strategy.

    Best for large expert counts (256+).
    """

    def __init__(self, config: ANEMoEConfig):
        config.strategy = MoEStrategy.HIERARCHICAL
        super().__init__(config)


class ExpertDistillation(nn.Module):
    """
    Expert Distillation Module.

    Compresses a large MoE (256 experts) into a smaller one (32 experts)
    through knowledge distillation.

    Training procedure:
    1. Run teacher (large MoE) to get expert outputs
    2. Run student (small MoE) on same inputs
    3. Minimize MSE between outputs + routing divergence

    This enables deploying large MoEs on ANE by distilling to
    a size that can use efficient batched/fused strategies.
    """

    def __init__(
        self,
        teacher_config: ANEMoEConfig,
        student_config: ANEMoEConfig,
        temperature: float = 2.0,
        alpha: float = 0.5,  # Balance between output and routing loss
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

        # Teacher (large, pretrained, frozen)
        self.teacher = ANEMoE(teacher_config)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Student (small, trainable)
        self.student = ANEMoE(student_config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward for training.

        Returns student output and distillation loss.
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out, _ = self.teacher(x)

        # Student forward
        student_out, _ = self.student(x)

        # Output distillation loss
        output_loss = F.mse_loss(student_out, teacher_out)

        return student_out, output_loss

    def get_student(self) -> ANEMoE:
        """Get the distilled student model for deployment."""
        return self.student
