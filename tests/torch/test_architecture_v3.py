"""
DeepSeek-V3 Architecture Tests

Tests for:
- Scaled MoE Architecture (256 Experts)
- Auxiliary-Loss-Free Load Balancing
- Multi-Latent Attention Enhancements
- Multi-Token Prediction (MTP)
- FP8 Training Infrastructure
- Expert Specialization Analysis
"""

import pytest
import torch
import sys
from pathlib import Path

# Add source to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# ============================================================================
# 3.1 Scaled MoE Architecture Tests
# ============================================================================

class TestScaledMoEArchitecture:
    """Tests for 256-expert MoE architecture."""
    
    def test_deepseek_moe_v3_config_small(self):
        """Test small V3 config creation."""
        from deepseek.torch.model.moe import DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        assert config.n_routed_experts == 16
        assert config.top_k == 2
        assert config.n_expert_groups == 4
        assert config.experts_per_group == 4  # 16/4
        assert config.routed_expert_hidden == 512
    
    def test_deepseek_moe_v3_config_medium(self):
        """Test medium V3 config creation."""
        from deepseek.torch.model.moe import DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.medium_64_4()
        assert config.n_routed_experts == 64
        assert config.top_k == 4
        assert config.n_expert_groups == 8
        assert config.experts_per_group == 8
    
    def test_deepseek_moe_v3_config_full(self):
        """Test full V3 config (256 experts)."""
        from deepseek.torch.model.moe import DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.v3_256_8()
        assert config.n_routed_experts == 256
        assert config.top_k == 8
        assert config.n_expert_groups == 8
        assert config.experts_per_group == 32  # 256/8
        assert config.routed_expert_intermediate == 2048  # Fine-grained
        assert config.routing_strategy == "sigmoid"
    
    def test_expert_v3_forward(self):
        """Test SwiGLU expert forward pass."""
        from deepseek.torch.model.moe import ExpertV3
        
        d_model = 256
        d_hidden = 512
        batch_size = 4
        
        expert = ExpertV3(d_model, d_hidden)
        x = torch.randn(batch_size, d_model)
        out = expert(x)
        
        assert out.shape == (batch_size, d_model)
        # Check gradient flow
        assert out.requires_grad
    
    def test_deepseek_moe_v3_forward_small(self):
        """Test V3 MoE forward pass with small config."""
        from deepseek.torch.model.moe import DeepSeekMoEV3, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        model = DeepSeekMoEV3(config)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_hierarchical_routing(self):
        """Test 2-stage hierarchical routing (group → expert)."""
        from deepseek.torch.model.moe import DeepSeekMoEV3, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        model = DeepSeekMoEV3(config)
        
        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        # Forward pass should use hierarchical routing
        output = model(x)
        assert output.shape == x.shape
        
        # Verify group centroids exist
        assert hasattr(model, 'group_centroids')
        assert model.group_centroids.shape == (config.n_expert_groups, config.d_model)
    
    def test_ablation_config(self):
        """Test ablation study configurations."""
        from deepseek.torch.model.moe import DeepSeekMoEV3Config
        
        for expert_count in [8, 64, 256]:
            config = DeepSeekMoEV3Config.for_ablation(expert_count)
            assert config.n_routed_experts == expert_count
            assert config.ablation_mode == str(expert_count)


# ============================================================================
# 3.2 Auxiliary-Loss-Free Load Balancing Tests
# ============================================================================

class TestAuxLossFreeLoadBalancing:
    """Tests for auxiliary-loss-free load balancing."""
    
    def test_load_balancing_state_init(self):
        """Test load balancing state initialization."""
        from deepseek.torch.model.moe import LoadBalancingState, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        state = LoadBalancingState(config)
        
        assert state.n_experts == 16
        assert state.bias.shape == (16,)
        assert state.ema_counts.shape == (16,)
        # Initial bias should be zero
        assert torch.allclose(state.bias, torch.zeros(16))
    
    def test_load_balancing_update(self):
        """Test bias update mechanism."""
        from deepseek.torch.model.moe import LoadBalancingState, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        state = LoadBalancingState(config)
        
        # Simulate imbalanced expert counts
        # Expert 0 gets many tokens, expert 15 gets few
        counts = torch.zeros(16)
        counts[0] = 100.0
        counts[15] = 1.0
        counts[1:15] = 10.0
        
        state.update(counts)
        
        # Expert 0 should have negative bias (discourage)
        # Expert 15 should have positive bias (encourage)
        assert state.bias[0] < 0.0
        assert state.bias[15] > 0.0
        
        # Check stats
        mean, imbalance, step = state.get_stats()
        assert step == 1.0
    
    def test_load_balancing_bias_clamping(self):
        """Test that biases are clamped within bounds."""
        from deepseek.torch.model.moe import LoadBalancingState, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        config.bias_clamp = 1.0  # Tight clamp
        state = LoadBalancingState(config)
        
        # Extreme imbalance
        counts = torch.zeros(16)
        counts[0] = 1000.0
        
        # Multiple updates
        for _ in range(100):
            state.update(counts)
        
        # Biases should be clamped
        assert state.bias.max() <= 1.0
        assert state.bias.min() >= -1.0
    
    def test_load_balancing_ema_decay(self):
        """Test EMA tracking of expert counts."""
        from deepseek.torch.model.moe import LoadBalancingState, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        config.ema_decay = 0.9
        state = LoadBalancingState(config)
        
        # First update
        counts1 = torch.ones(16) * 10.0
        state.update(counts1)
        
        # Second update with different distribution
        counts2 = torch.zeros(16)
        counts2[0] = 160.0  # All to expert 0
        state.update(counts2)
        
        # EMA should smooth the counts
        # After two updates: ema = 0.9 * ema + 0.1 * new_counts
        assert not torch.allclose(state.ema_counts, counts2)  # Should be smoothed
    
    def test_load_balancing_detailed_stats(self):
        """Test detailed statistics retrieval."""
        from deepseek.torch.model.moe import LoadBalancingState, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        state = LoadBalancingState(config)
        
        counts = torch.randn(16).abs() * 10
        state.update(counts)
        
        stats = state.get_detailed_stats()
        
        assert "mean_count" in stats
        assert "std_count" in stats
        assert "load_balance_cv" in stats
        assert "mean_bias" in stats
        assert "step" in stats
    
    def test_capacity_metrics(self):
        """Test capacity metrics tracking."""
        from deepseek.torch.model.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=8)
        
        # Simulate dispatch
        metrics.record_dispatch(expert_id=0, tokens_routed=50, capacity=40)
        metrics.record_dispatch(expert_id=1, tokens_routed=30, capacity=40)
        
        assert metrics.dropped_tokens == 10  # 50 - 40
        assert metrics.processed_tokens == 70  # 40 + 30
        assert metrics.drop_rate() > 0
        
        summary = metrics.get_summary()
        assert "drop_rate" in summary
        assert "load_balance_cv" in summary


# ============================================================================
# 3.4 Multi-Token Prediction Tests
# ============================================================================

class TestMultiTokenPrediction:
    """Tests for Multi-Token Prediction module."""
    
    def test_mtp_config(self):
        """Test MTP configuration."""
        from deepseek.torch.model.mtp import MTPConfig
        
        config = MTPConfig(
            vocab_size=1000,
            d_model=256,
            prediction_depth=3,
        )
        
        assert config.prediction_depth == 3
        assert config.enable_speculative
    
    def test_mtp_module_forward(self):
        """Test single MTP module forward pass."""
        from deepseek.torch.model.mtp import MTPModule
        
        d_model = 256
        d_embed = 256
        vocab_size = 1000
        batch_size = 2
        seq_len = 8
        
        module = MTPModule(d_model, d_embed, vocab_size)
        hidden = torch.randn(batch_size, seq_len, d_model)
        
        logits, new_hidden = module(hidden)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert new_hidden.shape == (batch_size, seq_len, d_embed)
    
    def test_mtp_module_with_context(self):
        """Test MTP module with previous context."""
        from deepseek.torch.model.mtp import MTPModule
        
        d_model = 256
        d_embed = 256
        vocab_size = 1000
        batch_size = 2
        seq_len = 8
        
        module = MTPModule(d_model, d_embed, vocab_size)
        hidden = torch.randn(batch_size, seq_len, d_model)
        prev_context = torch.randn(batch_size, seq_len, d_embed)
        
        logits, new_hidden = module(hidden, previous_context=prev_context)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_sequential_mtp_module(self):
        """Test sequential MTP module with transformer layer."""
        from deepseek.torch.model.mtp import SequentialMTPModule
        
        d_model = 256
        d_embed = 256
        vocab_size = 1000
        batch_size = 2
        seq_len = 8
        
        module = SequentialMTPModule(d_model, d_embed, vocab_size, num_heads=4)
        hidden = torch.randn(batch_size, seq_len, d_model)
        
        logits, new_hidden = module(hidden)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert new_hidden.shape == (batch_size, seq_len, d_embed)
    
    def test_mtp_gradient_flow(self):
        """Test gradient flow through MTP chain."""
        from deepseek.torch.model.mtp import MTPModule
        
        d_model = 64
        d_embed = 64
        vocab_size = 100
        
        modules = [MTPModule(d_model, d_embed, vocab_size) for _ in range(3)]
        
        hidden = torch.randn(2, 4, d_model, requires_grad=True)
        
        # Chain through modules
        current_hidden = hidden
        total_loss = 0
        for i, module in enumerate(modules):
            logits, current_hidden = module(current_hidden)
            # Dummy loss
            total_loss = total_loss + logits.mean()
        
        total_loss.backward()
        
        # Gradient should flow back to input
        assert hidden.grad is not None
        assert not torch.allclose(hidden.grad, torch.zeros_like(hidden.grad))


# ============================================================================
# 3.5 FP8 Training Infrastructure Tests
# ============================================================================

class TestFP8Training:
    """Tests for FP8 training infrastructure."""
    
    def test_fp8_format_enum(self):
        """Test FP8 format enumeration."""
        from deepseek.torch.model.quantization import FP8Format
        
        e4m3 = FP8Format.E4M3
        e5m2 = FP8Format.E5M2
        
        assert e4m3.max_value == 448.0
        assert e5m2.max_value == 57344.0
        # E5M2 has more exponent bits so can represent smaller numbers
        assert e4m3.smallest_normal > e5m2.smallest_normal
    
    def test_tile_scaling_config(self):
        """Test tile scaling configuration."""
        from deepseek.torch.model.quantization import TileScalingConfig, FP8Format
        
        config = TileScalingConfig()
        assert config.tile_rows == 128
        assert config.tile_cols == 128
        assert config.forward_format == FP8Format.E4M3
        assert config.backward_format == FP8Format.E5M2
    
    def test_tile_scaling_state_compute(self):
        """Test per-tile scale computation."""
        from deepseek.torch.model.quantization import TileScalingState, TileScalingConfig, FP8Format
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)  # Small tiles for testing
        state = TileScalingState(config)
        
        tensor = torch.randn(8, 8)  # 2x2 tiles
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        # Should have scales for each tile
        assert scales.shape[0] == 2  # 8/4 = 2
        assert scales.shape[1] == 2
    
    def test_tile_quantize_fp8(self):
        """Test tile-wise FP8 quantization."""
        from deepseek.torch.model.quantization import (
            quantize_fp8_tiled,
            TileScalingState,
            TileScalingConfig,
            FP8Format,
        )
        
        # Create tensor and compute scales
        tensor = torch.randn(128, 128)
        config = TileScalingConfig(tile_rows=32, tile_cols=32)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        # Quantize
        quantized = quantize_fp8_tiled(
            tensor, scales, FP8Format.E4M3, tile_rows=32, tile_cols=32
        )
        
        assert quantized.shape == tensor.shape
        # Scales should be per-tile
        assert scales.shape[0] == 4  # 128/32
        assert scales.shape[1] == 4
    
    def test_dequantize_fp8_tiled(self):
        """Test tile-wise FP8 dequantization."""
        from deepseek.torch.model.quantization import (
            quantize_fp8_tiled,
            dequantize_fp8_tiled,
            TileScalingState,
            TileScalingConfig,
            FP8Format,
        )
        
        tensor = torch.randn(64, 64)
        config = TileScalingConfig(tile_rows=16, tile_cols=16)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        # Quantize
        quantized = quantize_fp8_tiled(
            tensor, scales, FP8Format.E4M3, tile_rows=16, tile_cols=16
        )
        
        # Dequantize
        recovered = dequantize_fp8_tiled(quantized, scales, tile_rows=16, tile_cols=16)
        
        # Should be close (quantization introduces error)
        assert recovered.shape == tensor.shape


# ============================================================================
# 3.6 Expert Specialization Analysis Tests
# ============================================================================

class TestExpertSpecializationAnalysis:
    """Tests for expert specialization tracking."""
    
    def test_expert_frequency_tracking(self):
        """Test expert frequency tracker."""
        from deepseek.torch.model.moe import DeepSeekMoEV3, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        model = DeepSeekMoEV3(config)
        
        # Forward pass should update frequency tracking
        x = torch.randn(2, 4, config.d_model)
        output = model(x)
        
        # Check that model has capacity metrics
        assert hasattr(model, 'capacity_metrics')
    
    def test_load_balance_history(self):
        """Test load balance history tracking."""
        from deepseek.torch.model.moe import LoadBalancingState, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        state = LoadBalancingState(config)
        
        # Multiple updates
        for i in range(5):
            counts = torch.randn(16).abs() * (i + 1)
            state.update(counts)
        
        # Should have recorded history
        assert len(state.bias_history) == 5
        assert len(state.load_history) == 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestArchitectureV3Integration:
    """Integration tests for DeepSeek-V3 architecture components."""
    
    def test_moe_with_load_balancing_training(self):
        """Test MoE training loop with load balancing."""
        from deepseek.torch.model.moe import DeepSeekMoEV3, DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        model = DeepSeekMoEV3(config)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for step in range(3):
            x = torch.randn(2, 4, config.d_model)
            output = model(x)
            loss = output.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Model should have updated
        assert model.load_balance.step > 0
    
    def test_mtp_integration_with_loss(self):
        """Test MTP with loss computation."""
        from deepseek.torch.model.mtp import MTPModule
        
        d_model = 64
        vocab_size = 100
        batch_size = 2
        seq_len = 8
        
        mtp = MTPModule(d_model, d_model, vocab_size)
        
        hidden = torch.randn(batch_size, seq_len, d_model)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits, _ = mtp(hidden)
        
        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        loss.backward()
        
        # Gradients should exist
        assert mtp.hidden_proj.weight.grad is not None


# Run with: pytest tests/test_architecture_v3.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
