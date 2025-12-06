"""
Tests for Production GRPO Implementation (PyTorch)

Tests the following Phase 3 features:
1. PPO-style clipping for policy ratio
2. Dynamic beta adjustment based on KL divergence
3. Reference model update strategies
4. Entropy bonus for exploration
"""

from __future__ import annotations

import pytest
import torch

from src.deepseek.torch.training.grpo_production import (
    BetaSchedule,
    GRPOConfig,
    GRPOState,
    ProductionGRPOTrainer,
    ReferenceUpdateStrategy,
    RolloutBatch,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> GRPOConfig:
    """Default GRPO configuration."""
    return GRPOConfig(
        beta=0.01,
        clip_ratio=0.2,
        beta_schedule=BetaSchedule.ADAPTIVE,
        target_kl=0.01,
        ref_update_strategy=ReferenceUpdateStrategy.SOFT_UPDATE,
    )


@pytest.fixture
def device() -> torch.device:
    """Get available device with fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def simple_model(device: torch.device) -> torch.nn.Module:
    """Simple model for testing."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 32)
            self.proj = torch.nn.Linear(32, 100)

        def forward(
            self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            # Ignore attention_mask for simplicity
            return self.proj(self.embed(x))

    return SimpleModel().to(device)


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

    def test_basic_creation(self, simple_model: torch.nn.Module, device: torch.device) -> None:
        """Test basic trainer creation."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig()

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            config=config,
        )

        assert trainer.config.beta == 0.01
        assert trainer.state.step == 0

    def test_creation_without_models(self) -> None:
        """Test trainer creation without models."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainer(config=config)
        assert trainer.policy_model is None
        assert trainer.ref_model is None


# =============================================================================
# Advantage Computation Tests
# =============================================================================


class TestAdvantageComputation:
    """Test advantage computation."""

    def test_normalized_advantages(self, device: torch.device) -> None:
        """Test that advantages are normalized."""
        config = GRPOConfig(advantage_normalization=True)
        trainer = ProductionGRPOTrainer(config=config)

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        advantages = trainer.compute_advantages(rewards)

        # Check normalized to ~0 mean, ~1 std
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.2

    def test_unnormalized_advantages(self, device: torch.device) -> None:
        """Test advantages without normalization."""
        config = GRPOConfig(advantage_normalization=False)
        trainer = ProductionGRPOTrainer(config=config)

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        advantages = trainer.compute_advantages(rewards)

        # Should be centered around 0
        assert abs(advantages.mean().item()) < 0.1


# =============================================================================
# PPO Clipping Tests
# =============================================================================


class TestPPOClipping:
    """Test PPO-style clipping."""

    def test_clip_fraction_calculation(self, device: torch.device) -> None:
        """Test clip fraction is calculated correctly."""
        config = GRPOConfig(clip_ratio=0.2)
        trainer = ProductionGRPOTrainer(config=config)

        # Create mock log probs that will produce various ratios
        policy_log_probs = torch.zeros(4, 10, device=device)
        behavior_log_probs = torch.zeros(4, 10, device=device)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], device=device)

        loss, ratio, clip_frac = trainer.compute_ppo_loss(
            policy_log_probs, behavior_log_probs, advantages
        )

        # With equal log probs, ratio = 1, should not be clipped
        assert clip_frac == 0.0
        assert abs(ratio - 1.0) < 0.1

    def test_clipping_prevents_large_ratios(self, device: torch.device) -> None:
        """Test that large ratios are clipped."""
        config = GRPOConfig(clip_ratio=0.2)
        trainer = ProductionGRPOTrainer(config=config)

        # Large difference in log probs
        policy_log_probs = torch.ones(4, 10, device=device) * 2.0
        behavior_log_probs = torch.zeros(4, 10, device=device)
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

        _, ratio, clip_frac = trainer.compute_ppo_loss(
            policy_log_probs, behavior_log_probs, advantages
        )

        # With large log prob differences, should see clipping
        assert clip_frac > 0.0


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
        trainer = ProductionGRPOTrainer(config=config)
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
        trainer = ProductionGRPOTrainer(config=config)
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
        trainer = ProductionGRPOTrainer(config=config)
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
            beta_adjustment_factor=10.0,  # Large factor
        )
        trainer = ProductionGRPOTrainer(config=config)

        # Try to push beyond max
        for _ in range(10):
            trainer.update_beta(1.0)
        assert trainer.state.current_beta <= config.beta_max

        # Try to push below min
        for _ in range(10):
            trainer.update_beta(0.0001)
        assert trainer.state.current_beta >= config.beta_min


# =============================================================================
# Reference Model Update Tests
# =============================================================================


class TestReferenceModelUpdate:
    """Test reference model update strategies."""

    def test_no_update_strategy(self, simple_model: torch.nn.Module, device: torch.device) -> None:
        """Test NO_UPDATE strategy doesn't update reference."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig(
            ref_update_strategy=ReferenceUpdateStrategy.NO_UPDATE,
        )

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            config=config,
        )

        # Modify policy model
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(1.0)

        # Update reference - should return False
        updated = trainer.update_reference_model()
        assert not updated

    def test_hard_copy_strategy(self, simple_model: torch.nn.Module, device: torch.device) -> None:
        """Test HARD_COPY strategy."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig(
            ref_update_strategy=ReferenceUpdateStrategy.HARD_COPY,
            ref_update_interval=1,
        )

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            config=config,
        )

        # Modify policy
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(1.0)

        # Force step to trigger update
        trainer.state.step = 1
        updated = trainer.update_reference_model()
        assert updated

    def test_soft_update_strategy(
        self, simple_model: torch.nn.Module, device: torch.device
    ) -> None:
        """Test SOFT_UPDATE (Polyak averaging) strategy."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig(
            ref_update_strategy=ReferenceUpdateStrategy.SOFT_UPDATE,
            soft_update_tau=0.5,  # 50% update for clear testing
        )

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            config=config,
        )

        # Get initial reference params
        ref_param_before = next(ref_model.parameters()).clone()

        # Modify policy significantly
        with torch.no_grad():
            for p in simple_model.parameters():
                p.fill_(10.0)

        # Update reference
        updated = trainer.update_reference_model()
        assert updated

        # Reference should be between original and policy
        ref_param_after = next(ref_model.parameters())
        # With tau=0.5: new = 0.5 * old + 0.5 * 10.0
        expected = 0.5 * ref_param_before + 0.5 * 10.0
        assert torch.allclose(ref_param_after, expected, atol=1e-4)


# =============================================================================
# KL Divergence Tests
# =============================================================================


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_same_distributions(self, device: torch.device) -> None:
        """Test KL is ~0 for same distributions."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainer(config=config)

        logits = torch.randn(4, 10, 100, device=device)
        kl = trainer.compute_kl_divergence(logits, logits)

        assert kl.mean().item() < 0.1  # Should be ~0

    def test_kl_different_distributions(self, device: torch.device) -> None:
        """Test KL is positive for different distributions."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainer(config=config)

        policy_logits = torch.randn(4, 10, 100, device=device)
        ref_logits = torch.randn(4, 10, 100, device=device)
        kl = trainer.compute_kl_divergence(policy_logits, ref_logits)

        assert kl.mean().item() > 0  # Should be positive


# =============================================================================
# Entropy Tests
# =============================================================================


class TestEntropy:
    """Test entropy computation."""

    def test_uniform_distribution_high_entropy(self, device: torch.device) -> None:
        """Test uniform distribution has high entropy."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainer(config=config)

        # Uniform logits
        logits = torch.zeros(4, 10, 100, device=device)
        entropy = trainer.compute_entropy(logits)

        # Maximum entropy for 100 classes: log(100) ≈ 4.6
        assert entropy.mean().item() > 4.0

    def test_peaked_distribution_low_entropy(self, device: torch.device) -> None:
        """Test peaked distribution has low entropy."""
        config = GRPOConfig()
        trainer = ProductionGRPOTrainer(config=config)

        # Very peaked logits (one hot)
        logits = torch.full((4, 10, 100), -100.0, device=device)
        logits[:, :, 0] = 100.0  # Peak at first class
        entropy = trainer.compute_entropy(logits)

        # Should be near 0
        assert entropy.mean().item() < 0.1


# =============================================================================
# Full Loss Computation Tests
# =============================================================================


class TestGRPOLoss:
    """Test full GRPO loss computation."""

    def test_loss_computation(self, simple_model: torch.nn.Module, device: torch.device) -> None:
        """Test full loss computation."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig()

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            config=config,
        )

        # Create rollout batch
        input_ids = torch.randint(0, 100, (4, 10), device=device)
        rewards = torch.tensor([1.0, 0.5, -0.5, -1.0], device=device)

        rollouts = RolloutBatch(
            input_ids=input_ids,
            rewards=rewards,
        )

        policy_logits = simple_model(input_ids)
        ref_logits = ref_model(input_ids)

        loss_dict = trainer.compute_grpo_loss(rollouts, policy_logits, ref_logits)

        assert "total_loss" in loss_dict
        assert "policy_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert "entropy_loss" in loss_dict
        assert torch.isfinite(loss_dict["total_loss"])


# =============================================================================
# State Dict Tests
# =============================================================================


class TestStateDict:
    """Test state serialization."""

    def test_get_state_dict(self) -> None:
        """Test state dict contains expected fields."""
        config = GRPOConfig(beta=0.02)
        trainer = ProductionGRPOTrainer(config=config)
        trainer.state.step = 100
        trainer.state.current_beta = 0.015

        state_dict = trainer.get_state_dict()

        assert "config" in state_dict
        assert "state" in state_dict
        assert state_dict["state"]["step"] == 100
        assert state_dict["state"]["current_beta"] == 0.015


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full training flow."""

    def test_training_step_runs(self, simple_model: torch.nn.Module, device: torch.device) -> None:
        """Test a full training step executes without error."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig()

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config,
        )

        # Create rollout batch
        input_ids = torch.randint(0, 100, (4, 10), device=device)
        rewards = torch.tensor([1.0, 0.5, -0.5, -1.0], device=device)

        rollouts = RolloutBatch(
            input_ids=input_ids,
            rewards=rewards,
        )

        # Run training step
        metrics = trainer.train_step(rollouts)

        assert "loss" in metrics
        assert "mean_kl" in metrics
        assert "clip_fraction" in metrics
        assert trainer.state.step == 1

    def test_multiple_steps_update_state(
        self, simple_model: torch.nn.Module, device: torch.device
    ) -> None:
        """Test multiple training steps update state correctly."""
        import copy

        ref_model = copy.deepcopy(simple_model)
        config = GRPOConfig()

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        trainer = ProductionGRPOTrainer(
            policy_model=simple_model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config,
        )

        input_ids = torch.randint(0, 100, (4, 10), device=device)
        rewards = torch.tensor([1.0, 0.5, -0.5, -1.0], device=device)

        rollouts = RolloutBatch(
            input_ids=input_ids,
            rewards=rewards,
        )

        # Run multiple steps
        for _ in range(5):
            trainer.train_step(rollouts)

        assert trainer.state.step == 5
        assert len(trainer.state.kl_history) == 5
        assert len(trainer.state.loss_history) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
