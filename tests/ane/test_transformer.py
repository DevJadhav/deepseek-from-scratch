"""
Tests for ANE-Optimized DeepSeek Transformer Model

This module tests the complete ANEDeepSeekModel and related components.
"""

import pytest
import torch

from deepseek.mlx.ane.model.transformer import (
    ANEDeepSeekConfig,
    ANEDeepSeekModel,
    ANEGenerationConfig,
    ANEDeepSeekForCausalLM,
    create_model_from_config,
)


class TestANEDeepSeekConfig:
    """Tests for ANEDeepSeekConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANEDeepSeekConfig()

        assert config.vocab_size == 102400
        assert config.num_layers == 32
        assert config.d_model == 4096
        assert config.num_heads == 32
        assert config.max_seq_len == 8192
        assert config.num_experts == 32
        assert config.use_moe is True
        assert config.num_dense_layers == 1

    def test_tiny_config(self):
        """Test tiny preset configuration."""
        config = ANEDeepSeekConfig.tiny()

        assert config.vocab_size == 32000
        assert config.num_layers == 4
        assert config.d_model == 256
        assert config.num_heads == 4
        assert config.max_seq_len == 2048
        assert config.num_experts == 8

    def test_small_config(self):
        """Test small preset configuration."""
        config = ANEDeepSeekConfig.small()

        assert config.vocab_size == 32000
        assert config.num_layers == 12
        assert config.d_model == 1024
        assert config.num_experts == 16

    def test_standard_config(self):
        """Test standard preset configuration."""
        config = ANEDeepSeekConfig.standard()

        assert config.vocab_size == 102400
        assert config.num_layers == 32
        assert config.d_model == 4096

    def test_deepseek_v3_config(self):
        """Test DeepSeek-V3 style configuration."""
        config = ANEDeepSeekConfig.deepseek_v3()

        assert config.num_layers == 48
        assert config.d_model == 5120
        assert config.num_experts == 64

    def test_get_layer_config(self):
        """Test getting layer-specific configuration."""
        config = ANEDeepSeekConfig.tiny()

        # First layer should be dense (no MoE)
        layer0_config = config.get_layer_config(0)
        assert layer0_config.use_moe is False

        # Later layers should use MoE
        layer1_config = config.get_layer_config(1)
        assert layer1_config.use_moe is True


class TestANEGenerationConfig:
    """Tests for ANEGenerationConfig."""

    def test_default_generation_config(self):
        """Test default generation configuration."""
        config = ANEGenerationConfig()

        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.do_sample is True
        assert config.use_cache is True

    def test_custom_generation_config(self):
        """Test custom generation configuration."""
        config = ANEGenerationConfig(
            max_new_tokens=200,
            temperature=0.7,
            top_k=100,
            top_p=0.95,
            do_sample=False,
        )

        assert config.max_new_tokens == 200
        assert config.temperature == 0.7
        assert config.top_k == 100
        assert config.top_p == 0.95
        assert config.do_sample is False

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError):
            ANEGenerationConfig(temperature=0)

        with pytest.raises(ValueError):
            ANEGenerationConfig(temperature=-1.0)

    def test_invalid_top_p(self):
        """Test that invalid top_p raises error."""
        with pytest.raises(ValueError):
            ANEGenerationConfig(top_p=1.5)

        with pytest.raises(ValueError):
            ANEGenerationConfig(top_p=-0.1)


class TestANEDeepSeekModel:
    """Tests for ANEDeepSeekModel."""

    def test_model_creation(self):
        """Test model creation with tiny config."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)

        assert model.config == config
        assert hasattr(model, 'embed_tokens')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'lm_head')
        assert len(model.layers) == config.num_layers

    def test_forward_pass(self):
        """Test forward pass through the model."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        seq_len = 32
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            with torch.no_grad():
                logits, _ = model(input_ids)
            assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        batch_size = 2
        for seq_len in [16, 32, 64]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            with torch.no_grad():
                logits, _ = model(input_ids)
            assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_memory_footprint(self):
        """Test memory footprint estimation."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)

        footprint = model.get_memory_footprint()

        assert 'embedding_params' in footprint
        assert 'layer_params' in footprint
        assert 'total_params' in footprint
        assert 'weight_size_mb' in footprint
        assert footprint['total_params'] > 0

    def test_get_num_params(self):
        """Test parameter counting."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)

        num_params = model.get_num_params()
        num_params_no_embed = model.get_num_params(count_embeddings=False)

        assert num_params > 0
        assert num_params_no_embed > 0
        assert num_params > num_params_no_embed


class TestANEDeepSeekModelGeneration:
    """Tests for text generation."""

    def test_greedy_generation(self):
        """Test greedy decoding generation."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        gen_config = ANEGenerationConfig(
            max_new_tokens=5,
            do_sample=False,  # Greedy
        )

        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        with torch.no_grad():
            output_ids = model.generate(input_ids, gen_config)

        # Should have input + generated tokens
        assert output_ids.shape[1] >= input_ids.shape[1]
        assert output_ids.shape[1] <= input_ids.shape[1] + gen_config.max_new_tokens

    def test_sampling_generation(self):
        """Test sampling-based generation."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        gen_config = ANEGenerationConfig(
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
            top_k=50,
        )

        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        with torch.no_grad():
            output_ids = model.generate(input_ids, gen_config)

        assert output_ids.shape[1] >= input_ids.shape[1]


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_model_from_config(self):
        """Test model creation from config name."""
        model = create_model_from_config("tiny")
        assert model is not None
        assert hasattr(model, 'forward')

    def test_create_invalid_config(self):
        """Test that invalid config name raises error."""
        with pytest.raises(ValueError):
            create_model_from_config("invalid_config_name")


class TestModelNumerical:
    """Numerical tests for the model."""

    def test_output_not_nan(self):
        """Test that output contains no NaN values."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 32))

        with torch.no_grad():
            logits, _ = model(input_ids)

        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"

    def test_logits_reasonable_range(self):
        """Test that logits are in a reasonable range."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 32))

        with torch.no_grad():
            logits, _ = model(input_ids)

        # Logits should typically be in a reasonable range
        assert logits.abs().max() < 1000, f"Logits too large: {logits.abs().max()}"

    def test_deterministic_eval(self):
        """Test that eval mode is deterministic."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 32))

        with torch.no_grad():
            logits1, _ = model(input_ids)
            logits2, _ = model(input_ids)

        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(config)
        model.eval()

        input_ids1 = torch.randint(0, config.vocab_size, (1, 32))
        input_ids2 = torch.randint(0, config.vocab_size, (1, 32))

        # Make sure they're actually different
        while torch.equal(input_ids1, input_ids2):
            input_ids2 = torch.randint(0, config.vocab_size, (1, 32))

        with torch.no_grad():
            logits1, _ = model(input_ids1)
            logits2, _ = model(input_ids2)

        assert not torch.allclose(logits1, logits2, atol=1e-3)
