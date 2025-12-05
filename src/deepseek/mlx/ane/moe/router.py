"""
ANE-Optimized Expert Router Module

This module implements ANE-optimized expert routing with:
- Standard top-k routing with softmax
- Sigmoid affinity routing (DeepSeek-V3 style)
- Hierarchical two-stage routing (group → expert)
- Auxiliary-loss-free load balancing via bias adjustment
- Pre-computed routing for ANE constraints
"""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingStrategy(Enum):
    """Expert routing strategy options."""

    SOFTMAX = "softmax"  # Standard softmax routing
    SIGMOID = "sigmoid"  # Sigmoid affinity (DeepSeek-V3 style)
    TOPK_SOFTMAX = "topk_softmax"  # Top-k selection then softmax


@dataclass
class ANERouterConfig:
    """Configuration for ANE-optimized router."""

    d_model: int = 4096
    num_experts: int = 256
    top_k: int = 8
    routing_strategy: RoutingStrategy = RoutingStrategy.SIGMOID
    use_fp16: bool = True
    # Load balancing
    use_bias_adjustment: bool = True
    bias_lr: float = 0.01
    bias_clamp: float = 2.0
    ema_decay: float = 0.99
    # Capacity
    capacity_factor: float = 1.25
    min_capacity: int = 4
    enable_token_dropping: bool = False

    @classmethod
    def for_small(cls, num_experts: int = 16, top_k: int = 2) -> "ANERouterConfig":
        """Create config for small MoE (testing)."""
        return cls(
            d_model=512,
            num_experts=num_experts,
            top_k=top_k,
            routing_strategy=RoutingStrategy.SOFTMAX,
        )

    @classmethod
    def for_deepseek_v3(cls) -> "ANERouterConfig":
        """Create config matching DeepSeek-V3."""
        return cls(
            d_model=4096,
            num_experts=256,
            top_k=8,
            routing_strategy=RoutingStrategy.SIGMOID,
            use_bias_adjustment=True,
        )


class ANERouter(nn.Module):
    """
    ANE-Optimized Expert Router.

    Supports multiple routing strategies:
    - SOFTMAX: Standard softmax over all experts, select top-k
    - SIGMOID: Sigmoid affinity scores (DeepSeek-V3 default)
    - TOPK_SOFTMAX: Select top-k first, then softmax normalize

    Features:
    - FP16 computation for ANE
    - Optional bias adjustment for load balancing
    - EMA tracking of expert usage
    """

    def __init__(self, config: ANERouterConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.routing_strategy = config.routing_strategy
        self.use_fp16 = config.use_fp16

        # Router projection
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)

        # Load balancing bias (auxiliary-loss-free)
        if config.use_bias_adjustment:
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.num_experts),
            )
            self.register_buffer(
                "expert_usage_ema",
                torch.ones(config.num_experts) / config.num_experts,
            )
        else:
            self.expert_bias = None
            self.expert_usage_ema = None

        # Convert to FP16 if requested
        if config.use_fp16:
            self.router = self.router.half()

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Compute expert routing.

        Args:
            x: Input tensor (batch, seq_len, d_model) or (num_tokens, d_model)
            return_aux_loss: Whether to return auxiliary load balancing loss

        Returns:
            Tuple of:
            - expert_indices: Selected expert indices (num_tokens, top_k)
            - expert_weights: Gating weights (num_tokens, top_k)
            - aux_loss: Load balancing loss (optional)
        """
        # Flatten to (num_tokens, d_model)
        original_shape = x.shape[:-1]
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        # Convert to FP16 if needed
        if self.use_fp16 and x_flat.dtype != torch.float16:
            x_flat = x_flat.half()

        # Compute router logits
        router_logits = self.router(x_flat)  # (num_tokens, num_experts)

        # Apply bias adjustment for load balancing
        if self.expert_bias is not None:
            router_logits = router_logits + self.expert_bias

        # Compute routing based on strategy
        if self.routing_strategy == RoutingStrategy.SOFTMAX:
            expert_indices, expert_weights = self._softmax_routing(router_logits)
        elif self.routing_strategy == RoutingStrategy.SIGMOID:
            expert_indices, expert_weights = self._sigmoid_routing(router_logits)
        else:  # TOPK_SOFTMAX
            expert_indices, expert_weights = self._topk_softmax_routing(router_logits)

        # Update load balancing statistics
        aux_loss = None
        if self.training and self.expert_bias is not None:
            aux_loss = self._update_load_balance(expert_indices, num_tokens)

        return expert_indices, expert_weights, aux_loss

    def _softmax_routing(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard softmax routing with top-k selection."""
        # Softmax over all experts
        probs = F.softmax(logits, dim=-1)

        # Select top-k
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return top_k_indices, top_k_probs

    def _sigmoid_routing(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sigmoid affinity routing (DeepSeek-V3 style)."""
        # Sigmoid for affinity scores (not mutually exclusive)
        affinities = torch.sigmoid(logits)

        # Select top-k by affinity
        top_k_affinities, top_k_indices = torch.topk(affinities, self.top_k, dim=-1)

        # Normalize to get weights
        top_k_weights = top_k_affinities / top_k_affinities.sum(dim=-1, keepdim=True)

        return top_k_indices, top_k_weights

    def _topk_softmax_routing(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k first, then apply softmax."""
        # Select top-k by raw logits
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over selected experts only
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        return top_k_indices, top_k_probs

    def _update_load_balance(
        self,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Update load balancing via bias adjustment.

        This is auxiliary-loss-free: instead of adding a loss term,
        we directly adjust biases based on expert usage.
        """
        config = self.config

        # Count expert usage
        expert_counts = torch.bincount(
            expert_indices.view(-1),
            minlength=self.num_experts,
        ).float()

        # Normalize to get usage rate
        total_assignments = num_tokens * self.top_k
        usage_rate = expert_counts / total_assignments

        # Update EMA
        self.expert_usage_ema = (
            config.ema_decay * self.expert_usage_ema
            + (1 - config.ema_decay) * usage_rate
        )

        # Compute target (uniform distribution)
        target_usage = 1.0 / self.num_experts

        # Adjust bias to encourage uniform usage
        # Increase bias for underused experts, decrease for overused
        bias_delta = config.bias_lr * (target_usage - self.expert_usage_ema)
        self.expert_bias = torch.clamp(
            self.expert_bias + bias_delta,
            -config.bias_clamp,
            config.bias_clamp,
        )

        # Compute load imbalance as auxiliary "loss" for monitoring
        load_imbalance = torch.var(self.expert_usage_ema) * self.num_experts
        return load_imbalance

    def get_capacity(self, num_tokens: int) -> int:
        """Compute expert capacity for token dropping."""
        config = self.config
        uniform_capacity = num_tokens * self.top_k / self.num_experts
        capacity = int(uniform_capacity * config.capacity_factor)
        return max(capacity, config.min_capacity)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, strategy={self.routing_strategy.value}"
        )


@dataclass
class ANEHierarchicalRouterConfig:
    """Configuration for hierarchical router."""

    d_model: int = 4096
    num_experts: int = 256
    num_groups: int = 8
    top_k: int = 8
    top_k_groups: int = 4  # Number of groups to select
    routing_strategy: RoutingStrategy = RoutingStrategy.SIGMOID
    use_fp16: bool = True
    use_bias_adjustment: bool = True

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.num_groups


class ANEHierarchicalRouter(nn.Module):
    """
    ANE-Optimized Hierarchical Router (DeepSeek-V3 Style).

    Two-stage routing:
    1. Select top groups (coarse routing)
    2. Select experts within groups (fine routing)

    Benefits:
    - Reduces routing computation from O(num_experts) to O(num_groups + experts_per_group)
    - Better locality for expert grouping
    - More efficient load balancing per group
    """

    def __init__(self, config: ANEHierarchicalRouterConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.num_groups = config.num_groups
        self.experts_per_group = config.experts_per_group
        self.top_k = config.top_k
        self.top_k_groups = config.top_k_groups
        self.use_fp16 = config.use_fp16

        # Group router
        self.group_router = nn.Linear(
            config.d_model, config.num_groups, bias=False
        )

        # Expert routers (one per group)
        self.expert_routers = nn.ModuleList([
            nn.Linear(config.d_model, config.experts_per_group, bias=False)
            for _ in range(config.num_groups)
        ])

        # Load balancing biases
        if config.use_bias_adjustment:
            self.register_buffer(
                "group_bias",
                torch.zeros(config.num_groups),
            )
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.num_groups, config.experts_per_group),
            )
        else:
            self.group_bias = None
            self.expert_bias = None

        # Convert to FP16 if requested
        if config.use_fp16:
            self.group_router = self.group_router.half()
            for router in self.expert_routers:
                router.half()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchical expert routing.

        Args:
            x: Input tensor (batch, seq_len, d_model) or (num_tokens, d_model)

        Returns:
            Tuple of:
            - expert_indices: Global expert indices (num_tokens, top_k)
            - expert_weights: Gating weights (num_tokens, top_k)
            - group_indices: Selected group indices (num_tokens, top_k_groups)
        """
        # Flatten to (num_tokens, d_model)
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        if self.use_fp16 and x_flat.dtype != torch.float16:
            x_flat = x_flat.half()

        # Stage 1: Group selection
        group_logits = self.group_router(x_flat)  # (num_tokens, num_groups)

        if self.group_bias is not None:
            group_logits = group_logits + self.group_bias

        group_probs = F.softmax(group_logits, dim=-1)
        top_groups_probs, top_groups = torch.topk(
            group_probs, self.top_k_groups, dim=-1
        )  # (num_tokens, top_k_groups)

        # Stage 2: Expert selection within groups
        # For each token, route to experts within selected groups
        all_expert_indices = []
        all_expert_weights = []

        # Compute experts per group selected
        experts_per_selected_group = self.top_k // self.top_k_groups

        for token_idx in range(num_tokens):
            token_x = x_flat[token_idx:token_idx + 1]  # (1, d_model)
            token_groups = top_groups[token_idx]  # (top_k_groups,)
            token_group_probs = top_groups_probs[token_idx]  # (top_k_groups,)

            token_experts = []
            token_weights = []

            for g_idx, (group_id, group_prob) in enumerate(
                zip(token_groups, token_group_probs)
            ):
                group_id_int = group_id.item()

                # Get expert logits for this group
                expert_logits = self.expert_routers[group_id_int](token_x)  # (1, experts_per_group)

                if self.expert_bias is not None:
                    expert_logits = expert_logits + self.expert_bias[group_id_int]

                # Select top experts within group
                expert_probs = F.softmax(expert_logits, dim=-1)
                top_expert_probs, top_experts = torch.topk(
                    expert_probs, experts_per_selected_group, dim=-1
                )  # (1, experts_per_selected_group)

                # Convert local expert index to global
                global_expert_ids = (
                    group_id_int * self.experts_per_group + top_experts.squeeze(0)
                )

                # Combine group and expert probabilities
                combined_weights = group_prob * top_expert_probs.squeeze(0)

                token_experts.append(global_expert_ids)
                token_weights.append(combined_weights)

            # Concatenate experts from all groups
            all_expert_indices.append(torch.cat(token_experts))
            all_expert_weights.append(torch.cat(token_weights))

        expert_indices = torch.stack(all_expert_indices)  # (num_tokens, top_k)
        expert_weights = torch.stack(all_expert_weights)  # (num_tokens, top_k)

        # Renormalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_indices, expert_weights, top_groups

    def forward_batched(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched hierarchical routing (more ANE-friendly).

        Instead of per-token loops, processes in batches per group.
        """
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        if self.use_fp16 and x_flat.dtype != torch.float16:
            x_flat = x_flat.half()

        # Stage 1: Group selection (batched)
        group_logits = self.group_router(x_flat)

        if self.group_bias is not None:
            group_logits = group_logits + self.group_bias

        group_probs = F.softmax(group_logits, dim=-1)
        top_groups_probs, top_groups = torch.topk(
            group_probs, self.top_k_groups, dim=-1
        )

        # Stage 2: Expert selection (batched per group)
        experts_per_selected = self.top_k // self.top_k_groups

        # Initialize output tensors
        expert_indices = torch.zeros(
            num_tokens, self.top_k, dtype=torch.long, device=x.device
        )
        expert_weights = torch.zeros(
            num_tokens, self.top_k, dtype=x_flat.dtype, device=x.device
        )

        for g_pos in range(self.top_k_groups):
            group_ids = top_groups[:, g_pos]  # (num_tokens,)
            group_weights = top_groups_probs[:, g_pos]  # (num_tokens,)

            # Process each group
            for group_id in range(self.num_groups):
                mask = group_ids == group_id

                if mask.any():
                    tokens_in_group = x_flat[mask]

                    # Expert routing within group
                    expert_logits = self.expert_routers[group_id](tokens_in_group)

                    if self.expert_bias is not None:
                        expert_logits = expert_logits + self.expert_bias[group_id]

                    expert_probs = F.softmax(expert_logits, dim=-1)
                    top_probs, top_local = torch.topk(
                        expert_probs, experts_per_selected, dim=-1
                    )

                    # Convert to global indices
                    global_indices = group_id * self.experts_per_group + top_local

                    # Combined weights
                    combined = group_weights[mask].unsqueeze(-1) * top_probs

                    # Store in output
                    start_idx = g_pos * experts_per_selected
                    end_idx = start_idx + experts_per_selected
                    expert_indices[mask, start_idx:end_idx] = global_indices
                    expert_weights[mask, start_idx:end_idx] = combined

        # Renormalize
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_indices, expert_weights, top_groups

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_experts={self.num_experts}, "
            f"num_groups={self.num_groups}, top_k={self.top_k}"
        )
