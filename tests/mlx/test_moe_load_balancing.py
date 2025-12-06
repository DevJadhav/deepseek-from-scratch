"""
Tests for MLX RouterBiasController - DeepSeek-V3 Auxiliary-Loss-Free Load Balancing.

Section 1.2 of production_hardening.md remediation.
"""

import pytest
import mlx.core as mx
import dataclasses
from deepseek.mlx.moe import (
    DeepSeekMoEV3Config, RouterBiasController, LoadBalancingState,
    BIAS_UPDATE_ALPHA_RECOMMENDED
)


# ============================================================================
# RouterBiasController Tests
# ============================================================================

class TestRouterBiasController:
    """Tests for RouterBiasController - DeepSeek-V3 auxiliary-loss-free load balancing."""
    
    def test_creation(self):
        """Test RouterBiasController creation and basic properties."""
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config)
        
        # Verify initial bias is zeros
        bias = controller.get_bias()
        assert bias.shape == (config.n_routed_experts,)
        assert mx.allclose(bias, mx.zeros_like(bias), atol=1e-6), \
            "Initial bias should be zero"
        
        # Verify auxiliary loss is disabled
        assert not controller.use_auxiliary_loss(), \
            "RouterBiasController should disable auxiliary loss"
        
        # Verify initial step is 0
        assert controller.step == 0
    
    def test_uniform_update(self):
        """Test bias updates with uniform expert counts."""
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config)
        
        # Uniform distribution: all experts receive same count
        uniform_counts = mx.full((config.n_routed_experts,), 10.0)
        
        # Update with uniform counts
        controller.update_after_batch(uniform_counts)
        
        # Bias should remain close to zero for uniform distribution
        bias = controller.get_bias()
        max_bias = mx.max(mx.abs(bias)).item()
        assert max_bias < 0.1, \
            f"Bias should be near zero for uniform distribution, got max={max_bias}"
        
        assert controller.step == 1
    
    def test_imbalanced_update(self):
        """Test bias updates with imbalanced expert counts."""
        config = DeepSeekMoEV3Config.small_16_2()
        # Use higher bias_lr for visible effect in test
        controller = RouterBiasController(config, bias_update_alpha=0.1)
        
        # Imbalanced: first expert gets all tokens
        imbalanced_counts = mx.zeros((config.n_routed_experts,))
        # MLX doesn't support item assignment, create new array
        imbalanced_counts = mx.array([100.0] + [0.0] * (config.n_routed_experts - 1))
        
        # Update multiple times to build up bias
        for _ in range(10):
            controller.update_after_batch(imbalanced_counts)
        
        bias = controller.get_bias()
        
        # Overloaded expert (index 0) should have negative bias (discourage selection)
        assert bias[0].item() < 0.0, \
            f"Overloaded expert should have negative bias, got {bias[0].item()}"
        
        # At least some underloaded experts should have positive bias
        positive_count = mx.sum(bias[1:] > 0.0).item()
        assert positive_count > 0, \
            "Some underloaded experts should have positive bias"
    
    def test_bias_clamping(self):
        """Test that biases are properly clamped."""
        config = DeepSeekMoEV3Config.small_16_2()
        config = dataclasses.replace(config, bias_clamp=2.0)
        
        # Very high LR to trigger clamping quickly
        controller = RouterBiasController(config, bias_update_alpha=1.0)
        
        # Extreme imbalance
        extreme_counts = mx.array([10000.0] + [0.0] * (config.n_routed_experts - 1))
        
        # Many updates
        for _ in range(100):
            controller.update_after_batch(extreme_counts)
        
        # Verify all biases are within clamp range
        bias = controller.get_bias()
        assert mx.all(bias >= -config.bias_clamp).item(), \
            f"Bias should be >= -{config.bias_clamp}"
        assert mx.all(bias <= config.bias_clamp).item(), \
            f"Bias should be <= {config.bias_clamp}"
    
    def test_bias_update_alpha_alias(self):
        """Test that bias_update_alpha parameter works as alias."""
        config = DeepSeekMoEV3Config.small_16_2()
        
        alpha = 0.005
        controller = RouterBiasController(config, bias_update_alpha=alpha)
        
        # Should have overridden the config's bias_lr
        assert controller.config.bias_lr == alpha
    
    def test_statistics(self):
        """Test load balancing statistics."""
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config)
        
        # Add some data - varying counts
        counts = mx.arange(1, config.n_routed_experts + 1, dtype=mx.float32)
        controller.update_after_batch(counts)
        
        mean, imbalance, steps = controller.get_stats()
        
        assert mean > 0.0, "Mean should be positive"
        # imbalance is max/min ratio for MLX, should be > 1.0 for non-uniform
        assert imbalance >= 1.0, f"Imbalance should be >= 1.0, got {imbalance}"
        assert steps == 1.0, "Steps should be 1"


class TestLoadBalancingState:
    """Tests for LoadBalancingState (underlying implementation)."""
    
    def test_creation(self):
        """Test LoadBalancingState creation."""
        config = DeepSeekMoEV3Config.small_16_2()
        
        state = LoadBalancingState(config)
        
        assert state.bias.shape == (config.n_routed_experts,)
        assert state.step == 0
    
    def test_ema_decay(self):
        """Test that EMA decay is applied correctly."""
        config = DeepSeekMoEV3Config.small_16_2()
        config = dataclasses.replace(config, ema_decay=0.9)
        
        state = LoadBalancingState(config)
        
        # Initial counts are uniform 1/n_experts
        initial_ema = state.ema_counts
        
        # Update with new counts
        new_counts = mx.ones((config.n_routed_experts,)) * 10.0
        state.update(new_counts)
        
        # EMA should be: decay * old + (1 - decay) * new
        expected = 0.9 * initial_ema + 0.1 * new_counts
        assert mx.allclose(state.ema_counts, expected, atol=1e-5)
    
    def test_bias_clamp_from_config(self):
        """Test bias clamp uses config value."""
        config = DeepSeekMoEV3Config.small_16_2()
        config = dataclasses.replace(config, bias_clamp=1.0, bias_lr=10.0)
        
        state = LoadBalancingState(config)
        
        # Extreme imbalance
        extreme_counts = mx.array([1000.0] + [0.0] * (config.n_routed_experts - 1))
        
        for _ in range(50):
            state.update(extreme_counts)
        
        # Should be clamped to [-1, 1]
        assert mx.all(state.bias >= -1.0).item()
        assert mx.all(state.bias <= 1.0).item()


class TestBiasUpdateAlphaConstant:
    """Test the recommended constant value."""
    
    def test_constant_value(self):
        """Test BIAS_UPDATE_ALPHA_RECOMMENDED is 0.001."""
        assert abs(BIAS_UPDATE_ALPHA_RECOMMENDED - 0.001) < 1e-9


class TestAuxLossFreeConfig:
    """Test aux_loss_free configuration parameter."""
    
    def test_default_is_true(self):
        """Test that aux_loss_free defaults to True."""
        config = DeepSeekMoEV3Config()
        assert config.aux_loss_free is True
    
    def test_can_be_disabled(self):
        """Test that aux_loss_free can be set to False."""
        config = dataclasses.replace(DeepSeekMoEV3Config(), aux_loss_free=False)
        assert config.aux_loss_free is False


class TestDeepSeekMoEV3ConfigDefaults:
    """Test configuration defaults are correct."""
    
    def test_v3_256_8_config(self):
        """Test V3 256-expert config."""
        config = DeepSeekMoEV3Config.v3_256_8()
        assert config.n_routed_experts == 256
        assert config.top_k == 8
        assert config.n_expert_groups == 8
    
    def test_small_16_2_config(self):
        """Test small test config."""
        config = DeepSeekMoEV3Config.small_16_2()
        assert config.n_routed_experts == 16
        assert config.top_k == 2
        assert config.n_expert_groups == 4
    
    def test_bias_lr_default(self):
        """Test bias_lr default is 0.01."""
        config = DeepSeekMoEV3Config()
        assert config.bias_lr == 0.01
    
    def test_ema_decay_default(self):
        """Test ema_decay default is 0.99."""
        config = DeepSeekMoEV3Config()
        assert config.ema_decay == 0.99
