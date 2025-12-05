"""
Tests for ANE-Optimized Transformer Layer

This module tests the ANEDeepSeekLayer and ANEDeepSeekLayerConfig classes.
"""

import torch

from deepseek.mlx.ane.model.layer import ANEDeepSeekLayer, ANEDeepSeekLayerConfig


class TestANEDeepSeekLayerConfig:
    """Tests for ANEDeepSeekLayerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANEDeepSeekLayerConfig()

        assert config.d_model == 4096
        assert config.num_heads == 32
        assert config.d_latent == 512
        assert config.d_rope == 64
        assert config.chunk_size == 128
        assert config.d_hidden == 11008
        assert config.num_experts == 32
        assert config.num_shared == 2
        assert config.top_k == 4
        assert config.use_moe is True
        assert config.use_fp16 is True
        assert config.use_int8_weights is True
        assert config.dropout == 0.0

    def test_tiny_config(self):
        """Test tiny preset configuration."""
        config = ANEDeepSeekLayerConfig.tiny()

        assert config.d_model == 256
        assert config.num_heads == 4
        assert config.d_latent == 64
        assert config.d_hidden == 512
        assert config.num_experts == 8
        assert config.top_k == 2

    def test_small_config(self):
        """Test small preset configuration."""
        config = ANEDeepSeekLayerConfig.small()

        assert config.d_model == 1024
        assert config.num_heads == 16
        assert config.d_hidden == 4096
        assert config.num_experts == 16

    def test_deepseek_v3_config(self):
        """Test DeepSeek-V3 style configuration."""
        config = ANEDeepSeekLayerConfig.deepseek_v3()

        assert config.d_model == 4096
        assert config.num_heads == 32
        assert config.num_experts == 32


class TestANEDeepSeekLayer:
    """Tests for ANEDeepSeekLayer."""

    def test_layer_creation(self):
        """Test layer creation with tiny config."""
        config = ANEDeepSeekLayerConfig.tiny()
        layer = ANEDeepSeekLayer(config, layer_idx=0)

        assert layer.config == config
        assert layer.layer_idx == 0
        assert hasattr(layer, 'attn_norm')
        assert hasattr(layer, 'ffn_norm')
        assert hasattr(layer, 'attention')
        assert hasattr(layer, 'ffn')

    def test_forward_pass(self):
        """Test forward pass through the layer."""
        config = ANEDeepSeekLayerConfig.tiny()
        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            output, kv_cache = layer(x)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        config = ANEDeepSeekLayerConfig.tiny()
        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = ~mask  # True = attend

        with torch.no_grad():
            output, _ = layer(x, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_dense_ffn_mode(self):
        """Test layer with dense FFN (no MoE)."""
        config = ANEDeepSeekLayerConfig.tiny()
        config.use_moe = False  # Disable MoE

        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            output, _ = layer(x)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_fp16_computation(self):
        """Test FP16 computation mode."""
        config = ANEDeepSeekLayerConfig.tiny()
        config.use_fp16 = True

        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            output, _ = layer(x)

        # Output should be FP16 when use_fp16=True
        assert output.dtype == torch.float16

    def test_memory_footprint(self):
        """Test memory footprint estimation."""
        config = ANEDeepSeekLayerConfig.tiny()
        layer = ANEDeepSeekLayer(config, layer_idx=0)

        footprint = layer.get_memory_footprint()

        assert 'attention_params' in footprint
        assert 'ffn_params' in footprint
        assert 'total_params' in footprint
        assert 'weight_size_mb' in footprint
        assert footprint['total_params'] > 0
        assert footprint['weight_size_mb'] > 0

    def test_different_layer_indices(self):
        """Test creating layers with different indices."""
        config = ANEDeepSeekLayerConfig.tiny()

        for idx in [0, 1, 5, 10]:
            layer = ANEDeepSeekLayer(config, layer_idx=idx)
            assert layer.layer_idx == idx


class TestLayerNumerical:
    """Numerical tests for transformer layer."""

    def test_residual_connection(self):
        """Test that residual connections are working."""
        config = ANEDeepSeekLayerConfig.tiny()
        config.use_fp16 = False  # Use FP32 for numerical stability

        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 1, 16
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            output, _ = layer(x)

        # Output should not be identical to input (due to attention and FFN)
        assert not torch.allclose(output, x, atol=1e-3)

        # But should have similar magnitude (residual prevents explosion)
        input_norm = x.norm()
        output_norm = output.norm()
        ratio = output_norm / input_norm

        assert 0.1 < ratio < 10.0, f"Output norm ratio {ratio} is extreme"

    def test_output_not_nan(self):
        """Test that output contains no NaN values."""
        config = ANEDeepSeekLayerConfig.tiny()
        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            output, _ = layer(x)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_deterministic_eval(self):
        """Test that eval mode is deterministic."""
        config = ANEDeepSeekLayerConfig.tiny()
        layer = ANEDeepSeekLayer(config, layer_idx=0)
        layer.eval()

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            output1, _ = layer(x)
            output2, _ = layer(x)

        assert torch.allclose(output1, output2, atol=1e-5)
