"""
Tests for Production GRPO Implementation (MLX)

Tests the following Phase 3 features:
1. PPO-style clipping for policy ratio
2. Dynamic beta adjustment based on KL divergence
3. Reference model update strategies
4. Entropy bonus for exploration
"""

from __future__ import annotations

import pytest

# Check for MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

from src.deepseek.mlx.grpo_production import (
    MLX_AVAILABLE as MODULE_MLX_AVAILABLE,
)
from src.deepseek.mlx.grpo_production import (
    BetaSchedule,
    GRPOConfig,
    GRPOState,
    ProductionGRPOTrainerMLX,
    ReferenceUpdateStrategy,
)

# Skip all tests if MLX not available
pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE or not MODULE_MLX_AVAILABLE,
    reason="MLX not available",
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestGRPOConfig:
    """Test GRPO configuration dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GRPOConfig()
        assert config.beta == 0.01
        assert config.clip_ratio == 0.2
        assert config.beta_schedule == BetaSchedule.ADAPTIVE
        assert config.ref_update_strategy == ReferenceUpdateStrategy.SOFT_UPDATE
        assert config.target_kl == 0.01
        assert config.entropy_coef == 0.01

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = GRPOConfig(
            beta=0.05,
            clip_ratio=0.1,
            beta_schedule=BetaSchedule.CONSTANT,
            ref_update_strategy=ReferenceUpdateStrategy.HARD_COPY,
        )
        assert config.beta == 0.05
        assert config.clip_ratio == 0.1
        assert config.beta_schedule == BetaSchedule.CONSTANT
        assert config.ref_update_strategy == ReferenceUpdateStrategy.HARD_COPY


class TestGRPOState:
    """Test GRPO state tracking."""

    def test_initial_state(self) -> None:
        """Test initial state values."""
        state = GRPOState(current_beta=0.01)
        assert state.step == 0
        assert state.current_beta == 0.01
        assert state.mean_kl == 0.0
        assert state.clip_fraction == 0.0
        assert len(state.kl_history) == 0


# =============================================================================
# Trainer Creation Tests
# =============================================================================


class TestTrainerCreation:
    """Test trainer initialization."""

    def test_basic_creation(self) -> None:
        """Test basic trainer creation."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainerMLX(config=config)
        assert trainer.config.beta == 0.01
        assert trainer.state.step == 0

    def test_creation_without_models(self) -> None:
        """Test trainer creation without models."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainerMLX(config=config)
        assert trainer.policy_model is None
        assert trainer.ref_model is None


# =============================================================================
# Advantage Computation Tests
# =============================================================================


class TestAdvantageComputation:
    """Test advantage computation."""

    def test_normalized_advantages(self) -> None:
        """Test that advantages are normalized."""
        config = GRPOConfig(advantage_normalization=True)
        trainer = ProductionGRPOTrainerMLX(config=config)

        rewards = mx.array([1.0, 2.0, 3.0, 4.0])
        advantages = trainer.compute_advantages(rewards)

        # Check normalized to ~0 mean, ~1 std
        mean_val = float(mx.mean(advantages))
        std_val = float(mx.std(advantages))
        assert abs(mean_val) < 0.1
        assert abs(std_val - 1.0) < 0.2

    def test_unnormalized_advantages(self) -> None:
        """Test advantages without normalization."""
        config = GRPOConfig(advantage_normalization=False)
        trainer = ProductionGRPOTrainerMLX(config=config)

        rewards = mx.array([1.0, 2.0, 3.0, 4.0])
        advantages = trainer.compute_advantages(rewards)

        # Should be centered around 0
        mean_val = float(mx.mean(advantages))
        assert abs(mean_val) < 0.1


# =============================================================================
# Beta Adjustment Tests
# =============================================================================


class TestBetaAdjustment:
    """Test dynamic beta adjustment."""

    def test_beta_increases_when_kl_high(self) -> None:
        """Test beta increases when KL exceeds target."""
        config = GRPOConfig(
            beta=0.01,
            beta_schedule=BetaSchedule.ADAPTIVE,
            target_kl=0.01,
            beta_adjustment_factor=2.0,
        )
        trainer = ProductionGRPOTrainerMLX(config=config)
        initial_beta = trainer.state.current_beta

        # High KL (> 1.5 * target)
        trainer.update_beta(0.02)
        assert trainer.state.current_beta > initial_beta

    def test_beta_decreases_when_kl_low(self) -> None:
        """Test beta decreases when KL is below target."""
        config = GRPOConfig(
            beta=0.01,
            beta_schedule=BetaSchedule.ADAPTIVE,
            target_kl=0.01,
            beta_adjustment_factor=2.0,
        )
        trainer = ProductionGRPOTrainerMLX(config=config)
        initial_beta = trainer.state.current_beta

        # Low KL (< 0.5 * target)
        trainer.update_beta(0.001)
        assert trainer.state.current_beta < initial_beta

    def test_beta_constant_schedule(self) -> None:
        """Test beta stays constant with CONSTANT schedule."""
        config = GRPOConfig(
            beta=0.01,
            beta_schedule=BetaSchedule.CONSTANT,
        )
        trainer = ProductionGRPOTrainerMLX(config=config)
        initial_beta = trainer.state.current_beta

        trainer.update_beta(0.1)  # Very high KL
        assert trainer.state.current_beta == initial_beta

    def test_beta_respects_bounds(self) -> None:
        """Test beta respects min/max bounds."""
        config = GRPOConfig(
            beta=0.01,
            beta_schedule=BetaSchedule.ADAPTIVE,
            target_kl=0.01,
            beta_min=0.005,
            beta_max=0.02,
            beta_adjustment_factor=10.0,
        )
        trainer = ProductionGRPOTrainerMLX(config=config)

        # Try to push beyond max
        for _ in range(10):
            trainer.update_beta(1.0)
        assert trainer.state.current_beta <= config.beta_max

        # Try to push below min
        for _ in range(10):
            trainer.update_beta(0.0001)
        assert trainer.state.current_beta >= config.beta_min


# =============================================================================
# KL Divergence Tests
# =============================================================================


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_same_distributions(self) -> None:
        """Test KL is ~0 for same distributions."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainerMLX(config=config)

        logits = mx.random.normal((4, 10, 100))
        kl = trainer.compute_kl_divergence(logits, logits)

        mean_kl = float(mx.mean(kl))
        assert mean_kl < 0.1  # Should be ~0

    def test_kl_different_distributions(self) -> None:
        """Test KL is positive for different distributions."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainerMLX(config=config)

        policy_logits = mx.random.normal((4, 10, 100))
        ref_logits = mx.random.normal((4, 10, 100))
        kl = trainer.compute_kl_divergence(policy_logits, ref_logits)

        mean_kl = float(mx.mean(kl))
        assert mean_kl > 0  # Should be positive


# =============================================================================
# Entropy Tests
# =============================================================================


class TestEntropy:
    """Test entropy computation."""

    def test_uniform_distribution_high_entropy(self) -> None:
        """Test uniform distribution has high entropy."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainerMLX(config=config)

        # Uniform logits
        logits = mx.zeros((4, 10, 100))
        entropy = trainer.compute_entropy(logits)

        mean_entropy = float(mx.mean(entropy))
        # Maximum entropy for 100 classes: log(100) ≈ 4.6
        assert mean_entropy > 4.0

    def test_peaked_distribution_low_entropy(self) -> None:
        """Test peaked distribution has low entropy."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainerMLX(config=config)

        # Very peaked logits
        logits = mx.full((4, 10, 100), -100.0)
        # Set first class to high value
        logits = logits.at[:, :, 0].add(200.0)
        entropy = trainer.compute_entropy(logits)

        mean_entropy = float(mx.mean(entropy))
        # Should be near 0
        assert mean_entropy < 0.1


# =============================================================================
# PPO Clipping Tests
# =============================================================================


class TestPPOClipping:
    """Test PPO-style clipping."""

    def test_clip_fraction_zero_for_equal_probs(self) -> None:
        """Test clip fraction is zero when log probs are equal."""
        config = GRPOConfig(clip_ratio=0.2)
        trainer = ProductionGRPOTrainerMLX(config=config)

        # Equal log probs -> ratio = 1 -> no clipping
        policy_log_probs = mx.zeros((4, 10))
        behavior_log_probs = mx.zeros((4, 10))
        advantages = mx.array([1.0, -1.0, 0.5, -0.5])

        _, ratio, clip_frac = trainer.compute_ppo_loss(
            policy_log_probs, behavior_log_probs, advantages
        )

        assert float(clip_frac) == 0.0
        assert abs(float(ratio) - 1.0) < 0.1


# =============================================================================
# State Dict Tests
# =============================================================================


class TestStateDict:
    """Test state serialization."""

    def test_get_state_dict(self) -> None:
        """Test state dict contains expected fields."""
        config = GRPOConfig(beta=0.02)
        trainer = ProductionGRPOTrainerMLX(config=config)
        trainer.state.step = 100
        trainer.state.current_beta = 0.015

        state_dict = trainer.get_state_dict()

        assert "config" in state_dict
        assert "state" in state_dict
        assert state_dict["state"]["step"] == 100
        assert state_dict["state"]["current_beta"] == 0.015


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
