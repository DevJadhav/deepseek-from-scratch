"""
Tests for ANE-Optimized MoE Layer

Tests for ANEMoE, ANEMoEFused, ANEMoEBatched, and ANEMoEHierarchical.
"""

import pytest
import torch

from deepseek.mlx.ane.moe.moe import (
    ANEMoE,
    ANEMoEConfig,
    ANEMoEFused,
    ANEMoEBatched,
    ANEMoEHierarchical,
    MoEStrategy,
    ExpertDistillation,
)


class TestANEMoEConfig:
    """Tests for MoE configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANEMoEConfig()

        assert config.d_model == 4096
        assert config.num_routed_experts == 32
        assert config.num_shared_experts == 1
        assert config.top_k == 4

    def test_routed_d_hidden(self):
        """Test routed hidden dimension calculation."""
        config = ANEMoEConfig(d_model=4096, routed_hidden_mult=0.5)

        assert config.routed_d_hidden == 2048

    def test_shared_d_hidden(self):
        """Test shared hidden dimension calculation."""
        config = ANEMoEConfig(d_model=4096, shared_hidden_mult=4.0)

        assert config.shared_d_hidden == 16384

    def test_get_strategy_auto_fused(self):
        """Test auto strategy selection for small MoE."""
        config = ANEMoEConfig(num_routed_experts=8)

        assert config.get_strategy() == MoEStrategy.FUSED

    def test_get_strategy_auto_batched(self):
        """Test auto strategy selection for medium MoE."""
        config = ANEMoEConfig(num_routed_experts=32)

        assert config.get_strategy() == MoEStrategy.BATCHED

    def test_get_strategy_auto_hierarchical(self):
        """Test auto strategy selection for large MoE."""
        config = ANEMoEConfig(num_routed_experts=256)

        assert config.get_strategy() == MoEStrategy.HIERARCHICAL

    def test_small_8_2(self):
        """Test small config factory."""
        config = ANEMoEConfig.small_8_2()

        assert config.d_model == 256
        assert config.num_routed_experts == 8
        assert config.top_k == 2

    def test_medium_32_4(self):
        """Test medium config factory."""
        config = ANEMoEConfig.medium_32_4()

        assert config.d_model == 2048
        assert config.num_routed_experts == 32
        assert config.top_k == 4

    def test_large_256_8(self):
        """Test large config factory."""
        config = ANEMoEConfig.large_256_8()

        assert config.d_model == 4096
        assert config.num_routed_experts == 256
        assert config.top_k == 8


class TestANEMoEFused:
    """Tests for fused MoE strategy."""

    def test_initialization(self):
        """Test fused MoE initialization."""
        config = ANEMoEConfig.small_8_2()
        config.use_fp16 = False
        moe = ANEMoEFused(config)

        assert moe.strategy == MoEStrategy.FUSED
        assert moe.fused_expert is not None
        assert moe.router is not None

    def test_forward_shape(self):
        """Test forward pass shape."""
        config = ANEMoEConfig.small_8_2()
        config.use_fp16 = False
        moe = ANEMoEFused(config)

        x = torch.randn(2, 32, 256)
        output, aux_loss = moe(x)

        assert output.shape == x.shape

    def test_forward_with_shared(self):
        """Test forward with shared experts."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=8,
            num_shared_experts=2,
            top_k=2,
            use_fp16=False,
        )
        moe = ANEMoEFused(config)

        x = torch.randn(2, 16, 256)
        output, _ = moe(x)

        assert output.shape == x.shape

    def test_forward_no_shared(self):
        """Test forward without shared experts."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=8,
            num_shared_experts=0,
            top_k=2,
            use_fp16=False,
        )
        moe = ANEMoEFused(config)

        x = torch.randn(2, 16, 256)
        output, _ = moe(x)

        assert output.shape == x.shape


class TestANEMoEBatched:
    """Tests for batched MoE strategy."""

    def test_initialization(self):
        """Test batched MoE initialization."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=16,
            top_k=2,
            use_fp16=False,
        )
        moe = ANEMoEBatched(config)

        assert moe.strategy == MoEStrategy.BATCHED
        assert moe.experts is not None
        assert len(moe.experts) == 16

    def test_forward_shape(self):
        """Test forward pass shape."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=16,
            top_k=2,
            use_fp16=False,
        )
        moe = ANEMoEBatched(config)

        x = torch.randn(2, 32, 256)
        output, _ = moe(x)

        assert output.shape == x.shape

    def test_aux_loss_returned(self):
        """Test auxiliary loss is returned when requested."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=16,
            top_k=2,
            use_fp16=False,
        )
        moe = ANEMoEBatched(config)
        moe.train()

        x = torch.randn(2, 16, 256)
        output, aux_loss = moe(x, return_aux_loss=True)

        assert output.shape == x.shape
        assert aux_loss is not None


class TestANEMoEHierarchical:
    """Tests for hierarchical MoE strategy."""

    def test_initialization(self):
        """Test hierarchical MoE initialization."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=64,
            num_groups=8,
            top_k=4,
            use_fp16=False,
        )
        moe = ANEMoEHierarchical(config)

        assert moe.strategy == MoEStrategy.HIERARCHICAL
        assert moe.hierarchical_router is not None
        assert moe.expert_groups is not None
        assert len(moe.expert_groups) == 8

    def test_forward_shape(self):
        """Test forward pass shape."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=64,
            num_groups=8,
            top_k=4,
            use_fp16=False,
        )
        moe = ANEMoEHierarchical(config)

        x = torch.randn(2, 16, 256)
        output, _ = moe(x)

        assert output.shape == x.shape


class TestANEMoEGeneral:
    """General tests for ANE MoE."""

    def test_auto_strategy_selection(self):
        """Test automatic strategy selection."""
        # Small - should use FUSED
        config_small = ANEMoEConfig(num_routed_experts=8, d_model=256, use_fp16=False)
        moe_small = ANEMoE(config_small)
        assert moe_small.strategy == MoEStrategy.FUSED

        # Medium - should use BATCHED
        config_medium = ANEMoEConfig(num_routed_experts=32, d_model=256, use_fp16=False)
        moe_medium = ANEMoE(config_medium)
        assert moe_medium.strategy == MoEStrategy.BATCHED

        # Large - should use HIERARCHICAL
        config_large = ANEMoEConfig(num_routed_experts=256, num_groups=8, d_model=256, use_fp16=False)
        moe_large = ANEMoE(config_large)
        assert moe_large.strategy == MoEStrategy.HIERARCHICAL

    def test_explicit_strategy_override(self):
        """Test explicit strategy override."""
        config = ANEMoEConfig(
            num_routed_experts=64,
            strategy=MoEStrategy.FUSED,
            d_model=256,
            use_fp16=False,
        )
        moe = ANEMoE(config)

        assert moe.strategy == MoEStrategy.FUSED

    def test_dropout_applied(self):
        """Test dropout is applied when specified."""
        config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=8,
            top_k=2,
            dropout=0.1,
            use_fp16=False,
        )
        moe = ANEMoE(config)

        assert moe.dropout is not None


class TestMoENumerical:
    """Numerical stability tests for MoE."""

    def test_no_nan_output(self):
        """Test that MoE output contains no NaN."""
        config = ANEMoEConfig.small_8_2()
        config.use_fp16 = False
        moe = ANEMoE(config)

        x = torch.randn(2, 32, 256)
        output, _ = moe(x)

        assert not torch.isnan(output).any()

    def test_no_inf_output(self):
        """Test that MoE output contains no Inf."""
        config = ANEMoEConfig.small_8_2()
        config.use_fp16 = False
        moe = ANEMoE(config)

        x = torch.randn(2, 32, 256)
        output, _ = moe(x)

        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through MoE."""
        config = ANEMoEConfig.small_8_2()
        config.use_fp16 = False
        moe = ANEMoE(config)

        x = torch.randn(2, 16, 256, requires_grad=True)
        output, _ = moe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestExpertDistillation:
    """Tests for expert distillation."""

    def test_initialization(self):
        """Test distillation module initialization."""
        teacher_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=64,
            top_k=4,
            use_fp16=False,
        )
        student_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=16,
            top_k=2,
            use_fp16=False,
        )

        distill = ExpertDistillation(teacher_config, student_config)

        assert distill.teacher is not None
        assert distill.student is not None

    def test_teacher_frozen(self):
        """Test that teacher parameters are frozen."""
        teacher_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=32,
            top_k=2,
            use_fp16=False,
        )
        student_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=8,
            top_k=2,
            use_fp16=False,
        )

        distill = ExpertDistillation(teacher_config, student_config)

        for param in distill.teacher.parameters():
            assert not param.requires_grad

    def test_forward(self):
        """Test distillation forward pass."""
        teacher_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=32,
            top_k=2,
            use_fp16=False,
        )
        student_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=8,
            top_k=2,
            use_fp16=False,
        )

        distill = ExpertDistillation(teacher_config, student_config)

        x = torch.randn(2, 16, 256)
        output, loss = distill(x)

        assert output.shape == x.shape
        assert loss is not None
        assert loss.dim() == 0  # Scalar loss

    def test_get_student(self):
        """Test getting distilled student model."""
        teacher_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=32,
            top_k=2,
            use_fp16=False,
        )
        student_config = ANEMoEConfig(
            d_model=256,
            num_routed_experts=8,
            top_k=2,
            use_fp16=False,
        )

        distill = ExpertDistillation(teacher_config, student_config)
        student = distill.get_student()

        assert isinstance(student, ANEMoE)
        assert student.num_routed_experts == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
