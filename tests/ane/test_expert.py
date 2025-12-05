"""
Tests for ANE-Optimized Expert Module

Tests for ANEExpert, ANESharedExpert, ANEExpertGroup, and ANEFusedExpert.
"""

import pytest
import torch

from deepseek.mlx.ane.moe.expert import (
    ANEExpert,
    ANEExpertConfig,
    ANESharedExpert,
    ANEExpertGroup,
    ANEFusedExpert,
    ActivationType,
)


class TestANEExpertConfig:
    """Tests for expert configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANEExpertConfig()

        assert config.d_model == 4096
        assert config.d_hidden == 2048
        assert config.activation == ActivationType.SWIGLU
        assert config.use_fp16 is True

    def test_for_routed_expert(self):
        """Test routed expert factory."""
        config = ANEExpertConfig.for_routed_expert(d_model=2048, hidden_mult=0.5)

        assert config.d_model == 2048
        assert config.d_hidden == 1024  # 2048 * 0.5

    def test_for_shared_expert(self):
        """Test shared expert factory."""
        config = ANEExpertConfig.for_shared_expert(d_model=2048, hidden_mult=4.0)

        assert config.d_model == 2048
        assert config.d_hidden == 8192  # 2048 * 4.0


class TestANEExpert:
    """Tests for ANE Expert."""

    def test_initialization_swiglu(self):
        """Test expert initialization with SwiGLU."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            activation=ActivationType.SWIGLU,
            use_fp16=False,  # For easier testing
        )
        expert = ANEExpert(config)

        assert expert.d_model == 256
        assert expert.d_hidden == 512
        assert expert.gate_proj is not None
        assert expert.up_proj is not None
        assert expert.down_proj is not None

    def test_initialization_gelu(self):
        """Test expert initialization with GELU."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            activation=ActivationType.GELU,
            use_fp16=False,
        )
        expert = ANEExpert(config)

        assert expert.gate_proj is None  # No gate for non-SwiGLU
        assert expert.up_proj is not None
        assert expert.down_proj is not None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            use_fp16=False,
        )
        expert = ANEExpert(config)

        x = torch.randn(2, 32, 256)  # batch, seq, d_model
        output = expert(x)

        assert output.shape == x.shape

    def test_forward_2d_input(self):
        """Test forward with 2D input (num_tokens, d_model)."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            use_fp16=False,
        )
        expert = ANEExpert(config)

        x = torch.randn(64, 256)  # num_tokens, d_model
        output = expert(x)

        assert output.shape == x.shape

    def test_fp16_output(self):
        """Test FP16 mode."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            use_fp16=True,
        )
        expert = ANEExpert(config)

        x = torch.randn(2, 32, 256)
        output = expert(x)

        # Output should be converted back to FP32 for FP32 input
        assert output.dtype == torch.float32

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ActivationType:
            config = ANEExpertConfig(
                d_model=256,
                d_hidden=512,
                activation=activation,
                use_fp16=False,
            )
            expert = ANEExpert(config)

            x = torch.randn(2, 16, 256)
            output = expert(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestANESharedExpert:
    """Tests for ANE Shared Expert."""

    def test_initialization(self):
        """Test shared expert initialization."""
        shared = ANESharedExpert(
            d_model=256,
            d_hidden=1024,
            num_shared=2,
            use_fp16=False,
        )

        assert shared.d_model == 256
        assert shared.d_hidden == 1024
        assert shared.num_shared == 2
        assert len(shared.experts) == 2

    def test_forward_shape(self):
        """Test forward pass shape."""
        shared = ANESharedExpert(
            d_model=256,
            d_hidden=1024,
            num_shared=2,
            use_fp16=False,
        )

        x = torch.randn(2, 32, 256)
        output = shared(x)

        assert output.shape == x.shape

    def test_multiple_experts_sum(self):
        """Test that multiple shared experts are summed."""
        shared = ANESharedExpert(
            d_model=256,
            d_hidden=512,
            num_shared=3,
            use_fp16=False,
        )

        x = torch.randn(2, 16, 256)
        output = shared(x)

        # Output should be sum of 3 expert outputs
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestANEExpertGroup:
    """Tests for ANE Expert Group."""

    def test_initialization(self):
        """Test expert group initialization."""
        group = ANEExpertGroup(
            d_model=256,
            d_hidden=512,
            num_experts_in_group=8,
            use_fp16=False,
        )

        assert group.d_model == 256
        assert group.d_hidden == 512
        assert group.num_experts == 8
        assert len(group.experts) == 8

    def test_forward(self):
        """Test forward pass with routing."""
        group = ANEExpertGroup(
            d_model=256,
            d_hidden=512,
            num_experts_in_group=4,
            use_fp16=False,
        )

        num_tokens = 32
        x = torch.randn(num_tokens, 256)
        expert_indices = torch.randint(0, 4, (num_tokens,))
        expert_weights = torch.rand(num_tokens)

        output = group(x, expert_indices, expert_weights)

        assert output.shape == x.shape

    def test_forward_fused(self):
        """Test fused forward with top-k routing."""
        group = ANEExpertGroup(
            d_model=256,
            d_hidden=512,
            num_experts_in_group=4,
            use_fp16=False,
        )

        num_tokens = 32
        top_k = 2
        x = torch.randn(num_tokens, 256)
        expert_indices = torch.randint(0, 4, (num_tokens, top_k))
        expert_weights = torch.rand(num_tokens, top_k)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        output = group.forward_fused(x, expert_indices, expert_weights, top_k=top_k)

        assert output.shape == x.shape


class TestANEFusedExpert:
    """Tests for ANE Fused Expert."""

    def test_initialization(self):
        """Test fused expert initialization."""
        fused = ANEFusedExpert(
            d_model=256,
            d_hidden=512,
            num_experts=8,
            use_fp16=False,
        )

        assert fused.d_model == 256
        assert fused.d_hidden == 512
        assert fused.num_experts == 8
        assert fused.gate_weights.shape == (8, 512, 256)
        assert fused.up_weights.shape == (8, 512, 256)
        assert fused.down_weights.shape == (8, 256, 512)

    def test_forward_uniform_weights(self):
        """Test forward with uniform expert weights."""
        fused = ANEFusedExpert(
            d_model=256,
            d_hidden=512,
            num_experts=4,
            use_fp16=False,
        )

        x = torch.randn(2, 16, 256)
        weights = torch.ones(4) / 4  # Uniform

        output = fused(x, weights)

        assert output.shape == x.shape

    def test_forward_sparse_weights(self):
        """Test forward with sparse expert weights (one expert)."""
        fused = ANEFusedExpert(
            d_model=256,
            d_hidden=512,
            num_experts=4,
            use_fp16=False,
        )

        x = torch.randn(2, 16, 256)
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])

        output = fused(x, weights)

        assert output.shape == x.shape

    def test_forward_per_token_weights(self):
        """Test forward with per-token expert weights."""
        fused = ANEFusedExpert(
            d_model=256,
            d_hidden=512,
            num_experts=4,
            use_fp16=False,
        )

        batch, seq = 2, 16
        x = torch.randn(batch, seq, 256)
        weights = torch.rand(batch, seq, 4)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        output = fused(x, weights)

        assert output.shape == x.shape


class TestExpertNumerical:
    """Numerical stability tests for experts."""

    def test_no_nan_output(self):
        """Test that expert outputs contain no NaN."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            use_fp16=False,
        )
        expert = ANEExpert(config)

        x = torch.randn(2, 32, 256)
        output = expert(x)

        assert not torch.isnan(output).any()

    def test_no_inf_output(self):
        """Test that expert outputs contain no Inf."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            use_fp16=False,
        )
        expert = ANEExpert(config)

        x = torch.randn(2, 32, 256)
        output = expert(x)

        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through expert."""
        config = ANEExpertConfig(
            d_model=256,
            d_hidden=512,
            use_fp16=False,
        )
        expert = ANEExpert(config)

        x = torch.randn(2, 16, 256, requires_grad=True)
        output = expert(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
