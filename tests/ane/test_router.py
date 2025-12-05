"""
Tests for ANE-Optimized Router Module

Tests for ANERouter and ANEHierarchicalRouter.
"""

import pytest
import torch

from deepseek.mlx.ane.moe.router import (
    ANERouter,
    ANERouterConfig,
    ANEHierarchicalRouter,
    ANEHierarchicalRouterConfig,
    RoutingStrategy,
)


class TestANERouterConfig:
    """Tests for router configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANERouterConfig()

        assert config.d_model == 4096
        assert config.num_experts == 256
        assert config.top_k == 8
        assert config.routing_strategy == RoutingStrategy.SIGMOID

    def test_for_small(self):
        """Test small config factory."""
        config = ANERouterConfig.for_small(num_experts=16, top_k=2)

        assert config.num_experts == 16
        assert config.top_k == 2
        assert config.d_model == 512

    def test_for_deepseek_v3(self):
        """Test DeepSeek-V3 config factory."""
        config = ANERouterConfig.for_deepseek_v3()

        assert config.d_model == 4096
        assert config.num_experts == 256
        assert config.top_k == 8
        assert config.routing_strategy == RoutingStrategy.SIGMOID


class TestANERouter:
    """Tests for ANE Router."""

    def test_initialization(self):
        """Test router initialization."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            use_fp16=False,
        )
        router = ANERouter(config)

        assert router.d_model == 256
        assert router.num_experts == 8
        assert router.top_k == 2

    def test_forward_output_shape(self):
        """Test forward pass output shapes."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 32, 256)  # batch, seq, d_model
        expert_indices, expert_weights, aux_loss = router(x)

        # Flattened: 2 * 32 = 64 tokens
        assert expert_indices.shape == (64, 2)  # (num_tokens, top_k)
        assert expert_weights.shape == (64, 2)  # (num_tokens, top_k)

    def test_weights_sum_to_one(self):
        """Test that expert weights sum to 1 per token."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=4,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 16, 256)
        _, expert_weights, _ = router(x)

        # Weights should sum to approximately 1
        weight_sums = expert_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_softmax_routing(self):
        """Test softmax routing strategy."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            routing_strategy=RoutingStrategy.SOFTMAX,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 16, 256)
        expert_indices, expert_weights, _ = router(x)

        assert expert_indices.shape[1] == 2
        assert expert_weights.min() >= 0

    def test_sigmoid_routing(self):
        """Test sigmoid routing strategy."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            routing_strategy=RoutingStrategy.SIGMOID,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 16, 256)
        expert_indices, expert_weights, _ = router(x)

        assert expert_indices.shape[1] == 2
        assert expert_weights.min() >= 0

    def test_topk_softmax_routing(self):
        """Test top-k softmax routing strategy."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            routing_strategy=RoutingStrategy.TOPK_SOFTMAX,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 16, 256)
        expert_indices, expert_weights, _ = router(x)

        assert expert_indices.shape[1] == 2
        assert expert_weights.min() >= 0

    def test_expert_indices_in_range(self):
        """Test that expert indices are within valid range."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 16, 256)
        expert_indices, _, _ = router(x)

        assert expert_indices.min() >= 0
        assert expert_indices.max() < 8

    def test_bias_adjustment_initialization(self):
        """Test bias adjustment buffer initialization."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            use_bias_adjustment=True,
            use_fp16=False,
        )
        router = ANERouter(config)

        assert router.expert_bias is not None
        assert router.expert_usage_ema is not None
        assert router.expert_bias.shape == (8,)

    def test_no_bias_adjustment(self):
        """Test router without bias adjustment."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            use_bias_adjustment=False,
            use_fp16=False,
        )
        router = ANERouter(config)

        assert router.expert_bias is None
        assert router.expert_usage_ema is None

    def test_capacity_calculation(self):
        """Test expert capacity calculation."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            capacity_factor=1.25,
            min_capacity=4,
        )
        router = ANERouter(config)

        # 64 tokens, top_k=2, 8 experts
        # Uniform capacity = 64 * 2 / 8 = 16
        # With factor 1.25 = 20
        capacity = router.get_capacity(num_tokens=64)
        assert capacity == 20

    def test_min_capacity(self):
        """Test minimum capacity enforcement."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=100,
            top_k=1,
            capacity_factor=1.0,
            min_capacity=4,
        )
        router = ANERouter(config)

        # 4 tokens, top_k=1, 100 experts
        # Uniform capacity = 4 * 1 / 100 = 0.04 -> 0
        # But min_capacity = 4
        capacity = router.get_capacity(num_tokens=4)
        assert capacity >= 4


class TestANEHierarchicalRouter:
    """Tests for ANE Hierarchical Router."""

    def test_initialization(self):
        """Test hierarchical router initialization."""
        config = ANEHierarchicalRouterConfig(
            d_model=256,
            num_experts=64,
            num_groups=8,
            top_k=4,
            top_k_groups=2,
            use_fp16=False,
        )
        router = ANEHierarchicalRouter(config)

        assert router.d_model == 256
        assert router.num_experts == 64
        assert router.num_groups == 8
        assert router.experts_per_group == 8
        assert router.top_k == 4
        assert router.top_k_groups == 2

    def test_forward_output_shape(self):
        """Test forward pass output shapes."""
        config = ANEHierarchicalRouterConfig(
            d_model=256,
            num_experts=32,
            num_groups=4,
            top_k=4,
            top_k_groups=2,
            use_fp16=False,
        )
        router = ANEHierarchicalRouter(config)

        x = torch.randn(2, 16, 256)  # batch, seq, d_model
        expert_indices, expert_weights, group_indices = router(x)

        num_tokens = 2 * 16  # 32
        assert expert_indices.shape == (num_tokens, 4)  # top_k
        assert expert_weights.shape == (num_tokens, 4)
        assert group_indices.shape == (num_tokens, 2)  # top_k_groups

    def test_forward_batched(self):
        """Test batched forward pass."""
        config = ANEHierarchicalRouterConfig(
            d_model=256,
            num_experts=32,
            num_groups=4,
            top_k=4,
            top_k_groups=2,
            use_fp16=False,
        )
        router = ANEHierarchicalRouter(config)

        x = torch.randn(2, 16, 256)
        expert_indices, expert_weights, group_indices = router.forward_batched(x)

        num_tokens = 32
        assert expert_indices.shape == (num_tokens, 4)
        assert expert_weights.shape == (num_tokens, 4)
        assert group_indices.shape == (num_tokens, 2)

    def test_expert_indices_in_range(self):
        """Test that expert indices are within valid range."""
        config = ANEHierarchicalRouterConfig(
            d_model=256,
            num_experts=64,
            num_groups=8,
            top_k=4,
            top_k_groups=2,
            use_fp16=False,
        )
        router = ANEHierarchicalRouter(config)

        x = torch.randn(2, 16, 256)
        expert_indices, _, _ = router.forward_batched(x)

        assert expert_indices.min() >= 0
        assert expert_indices.max() < 64

    def test_weights_sum_to_one(self):
        """Test that expert weights sum to approximately 1."""
        config = ANEHierarchicalRouterConfig(
            d_model=256,
            num_experts=32,
            num_groups=4,
            top_k=4,
            top_k_groups=2,
            use_fp16=False,
        )
        router = ANEHierarchicalRouter(config)

        x = torch.randn(2, 16, 256)
        _, expert_weights, _ = router.forward_batched(x)

        weight_sums = expert_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


class TestRouterNumerical:
    """Numerical stability tests for routers."""

    def test_no_nan_weights(self):
        """Test that router weights contain no NaN."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 32, 256)
        _, expert_weights, _ = router(x)

        assert not torch.isnan(expert_weights).any()

    def test_no_negative_weights(self):
        """Test that router weights are non-negative."""
        config = ANERouterConfig(
            d_model=256,
            num_experts=8,
            top_k=2,
            use_fp16=False,
        )
        router = ANERouter(config)

        x = torch.randn(2, 32, 256)
        _, expert_weights, _ = router(x)

        assert (expert_weights >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
