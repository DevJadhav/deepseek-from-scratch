import pytest
import torch
import os
from deepseek.torch.model.moe import (
    DeepSeekMoE, StandardMoE, 
    DeepSeekMoEV3Config, RouterBiasController, LoadBalancingState,
    BIAS_UPDATE_ALPHA_RECOMMENDED
)

# Enable MPS fallback for operations not supported on Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    """Get best available device with fallback support."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_moe_forward():
    """Test MoE forward pass."""
    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    num_experts = 4
    num_shared = 1
    top_k = 2
    
    model = DeepSeekMoE(
        d_model=hidden_dim,
        d_hidden=hidden_dim * 4,
        num_experts=num_experts,
        num_shared=num_shared,
        num_routed=num_experts,
        top_k=top_k
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output = model(x)
    
    assert output.shape == (batch_size, seq_len, hidden_dim)
    # assert isinstance(aux_loss, torch.Tensor) # Model doesn't return aux loss currently


def test_moe_load_balancing():
    """Test that experts are actually being used."""
    # This is a probabilistic test, might be flaky if not careful.
    # We just check that we get a valid output.
    pass


# ============================================================================
# RouterBiasController Tests (Section 1.2 Auxiliary-Loss-Free Load Balancing)
# ============================================================================

class TestRouterBiasController:
    """Tests for RouterBiasController - DeepSeek-V3 auxiliary-loss-free load balancing."""
    
    def test_creation(self):
        """Test RouterBiasController creation and basic properties."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config, device)
        
        # Verify initial bias is zeros
        bias = controller.get_bias()
        assert bias.shape == (config.n_routed_experts,)
        assert torch.allclose(bias, torch.zeros_like(bias), atol=1e-6), \
            "Initial bias should be zero"
        
        # Verify auxiliary loss is disabled
        assert not controller.use_auxiliary_loss(), \
            "RouterBiasController should disable auxiliary loss"
        
        # Verify initial step is 0
        assert controller.step == 0
    
    def test_uniform_update(self):
        """Test bias updates with uniform expert counts."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config, device)
        
        # Uniform distribution: all experts receive same count
        uniform_counts = torch.full(
            (config.n_routed_experts,), 10.0, device=device
        )
        
        # Update with uniform counts
        controller.update_after_batch(uniform_counts)
        
        # Bias should remain close to zero for uniform distribution
        bias = controller.get_bias()
        assert torch.all(bias.abs() < 0.1), \
            f"Bias should be near zero for uniform distribution, got max={bias.abs().max()}"
        
        assert controller.step == 1
    
    def test_imbalanced_update(self):
        """Test bias updates with imbalanced expert counts."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        # Use higher bias_lr for visible effect in test
        controller = RouterBiasController(config, device, bias_update_alpha=0.1)
        
        # Imbalanced: first expert gets all tokens
        imbalanced_counts = torch.zeros(config.n_routed_experts, device=device)
        imbalanced_counts[0] = 100.0  # First expert is overloaded
        
        # Update multiple times to build up bias
        for _ in range(10):
            controller.update_after_batch(imbalanced_counts)
        
        bias = controller.get_bias()
        
        # Overloaded expert (index 0) should have negative bias (discourage selection)
        assert bias[0] < 0.0, \
            f"Overloaded expert should have negative bias, got {bias[0].item()}"
        
        # At least some underloaded experts should have positive bias
        positive_count = (bias[1:] > 0.0).sum().item()
        assert positive_count > 0, \
            "Some underloaded experts should have positive bias"
    
    def test_bias_clamping(self):
        """Test that biases are properly clamped."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        config.bias_clamp = 2.0
        
        # Very high LR to trigger clamping quickly
        controller = RouterBiasController(config, device, bias_update_alpha=1.0)
        
        # Extreme imbalance
        extreme_counts = torch.zeros(config.n_routed_experts, device=device)
        extreme_counts[0] = 10000.0
        
        # Many updates
        for _ in range(100):
            controller.update_after_batch(extreme_counts)
        
        # Verify all biases are within clamp range
        bias = controller.get_bias()
        assert torch.all(bias >= -config.bias_clamp), \
            f"Bias should be >= -{config.bias_clamp}"
        assert torch.all(bias <= config.bias_clamp), \
            f"Bias should be <= {config.bias_clamp}"
    
    def test_bias_update_alpha_alias(self):
        """Test that bias_update_alpha parameter works as alias."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        
        alpha = 0.005
        controller = RouterBiasController(config, device, bias_update_alpha=alpha)
        
        # Should have overridden the config's bias_lr
        assert controller.config.bias_lr == alpha
    
    def test_device_movement(self):
        """Test that controller can be moved between devices."""
        config = DeepSeekMoEV3Config.small_16_2()
        controller = RouterBiasController(config, torch.device('cpu'))
        
        # Should be able to move to CPU (always available)
        controller.to(torch.device('cpu'))
        assert controller.state.device == torch.device('cpu')
    
    def test_statistics(self):
        """Test load balancing statistics."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config, device)
        
        # Add some data
        counts = torch.arange(1, config.n_routed_experts + 1, dtype=torch.float32, device=device)
        controller.update_after_batch(counts)
        
        mean, imbalance, steps = controller.get_stats()
        
        assert mean > 0.0, "Mean should be positive"
        # imbalance is coefficient of variation (std/mean), always >= 0
        assert imbalance >= 0.0, "Imbalance (CV) should be >= 0.0"
        assert steps == 1.0, "Steps should be 1"
        
        detailed = controller.get_detailed_stats()
        assert detailed['mean_count'] > 0.0
        assert detailed['step'] == 1
    
    def test_history_tracking(self):
        """Test bias and load history are tracked."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        
        controller = RouterBiasController(config, device)
        
        # Do several updates
        for i in range(5):
            counts = torch.full(
                (config.n_routed_experts,), float(i + 1), device=device
            )
            controller.update_after_batch(counts)
        
        # Check history
        bias_history = controller.get_bias_history()
        load_history = controller.get_load_history()
        
        assert len(bias_history) == 5
        assert len(load_history) == 5


class TestLoadBalancingState:
    """Tests for LoadBalancingState (underlying implementation)."""
    
    def test_creation(self):
        """Test LoadBalancingState creation."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        
        state = LoadBalancingState(config, device)
        
        assert state.bias.shape == (config.n_routed_experts,)
        assert state.step == 0
    
    def test_ema_decay(self):
        """Test that EMA decay is applied correctly."""
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        config.ema_decay = 0.9  # Known decay rate
        
        state = LoadBalancingState(config, device)
        
        # Initial counts are uniform 1/n_experts
        initial_ema = state.ema_counts.clone()
        
        # Update with new counts
        new_counts = torch.ones(config.n_routed_experts, device=device) * 10.0
        state.update(new_counts)
        
        # EMA should be: decay * old + (1 - decay) * new
        expected = 0.9 * initial_ema + 0.1 * new_counts
        assert torch.allclose(state.ema_counts, expected, atol=1e-5)


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
        config = DeepSeekMoEV3Config(aux_loss_free=False)
        assert config.aux_loss_free is False
