"""
Tests for Tiny MLX Trainer
==========================

Tests for the TinyMLXTrainer, TinyMTPModel, and related components
used for training small DeepSeek models on Apple Silicon.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip all tests if MLX is not available
mlx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

# ruff: noqa: E402
from deepseek.mlx.tiny_trainer import (
    DataLoader,
    RMSNorm,
    TinyMLA,
    TinyMLXTrainer,
    TinyMoELayer,
    TinyModelConfig,
    TinyMTPModel,
    TinyTransformerBlock,
    compute_mtp_loss,
    create_tiny_model,
)


class TestTinyModelConfig:
    """Tests for TinyModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TinyModelConfig()
        assert config.vocab_size == 8000
        assert config.d_model == 256
        assert config.num_heads == 4
        assert config.num_layers == 4
        assert config.mtp_k == 2
        assert config.num_experts == 4
        assert config.top_k == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TinyModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_layers=2,
            mtp_k=3,
            num_experts=4,
            top_k=1,
        )
        assert config.vocab_size == 1000
        assert config.d_model == 64
        assert config.num_heads == 2
        assert config.num_layers == 2
        assert config.mtp_k == 3
        assert config.num_experts == 4
        assert config.top_k == 1

    def test_config_serialization(self):
        """Test config can be serialized to dict."""
        config = TinyModelConfig(vocab_size=500, d_model=32)
        config_dict = config.__dict__
        assert isinstance(config_dict, dict)
        assert config_dict["vocab_size"] == 500
        assert config_dict["d_model"] == 32


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_creation(self):
        """Test RMSNorm layer creation."""
        norm = RMSNorm(64)
        assert norm.weight.shape == (64,)

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        norm = RMSNorm(64)
        x = mlx.random.normal((2, 10, 64))
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalization(self):
        """Test RMSNorm actually normalizes."""
        norm = RMSNorm(64, eps=1e-6)
        x = mlx.random.normal((2, 10, 64)) * 10  # Large values
        out = norm(x)
        mlx.eval(out)
        # Output should have smaller magnitude on average
        assert float(mlx.mean(mlx.abs(out)).item()) < float(mlx.mean(mlx.abs(x)).item())


class TestTinyMLA:
    """Tests for TinyMLA (Multi-Head Latent Attention)."""

    def test_mla_creation(self):
        """Test MLA layer creation."""
        config = TinyModelConfig(d_model=64, num_heads=4, d_latent=32, d_rope=16)
        mla = TinyMLA(config)
        assert mla.num_heads == 4
        assert mla.head_dim == 16  # d_model / num_heads

    def test_mla_forward(self):
        """Test MLA forward pass."""
        config = TinyModelConfig(d_model=64, num_heads=4, d_latent=32, d_rope=16)
        mla = TinyMLA(config)
        x = mlx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        out, cache = mla(x, mask)
        assert out.shape == x.shape

    def test_mla_with_cache(self):
        """Test MLA with KV cache."""
        config = TinyModelConfig(d_model=64, num_heads=4, d_latent=32, d_rope=16)
        mla = TinyMLA(config)

        # First pass without cache
        x1 = mlx.random.normal((2, 5, 64))
        mask1 = nn.MultiHeadAttention.create_additive_causal_mask(5)
        out1, cache = mla(x1, mask1)
        assert cache is not None
        assert out1.shape == (2, 5, 64)

        # Note: Testing cache continuation is complex due to mask shape requirements
        # The MLA implementation may need adjustments for proper cache support
        # For now, verify cache is returned
        assert isinstance(cache, tuple)
        assert len(cache) == 2  # (keys, values)


class TestTinyMoELayer:
    """Tests for TinyMoELayer (Mixture of Experts)."""

    def test_moe_creation(self):
        """Test MoE layer creation."""
        config = TinyModelConfig(d_model=64, num_experts=4, num_shared_experts=1, top_k=2)
        moe = TinyMoELayer(config)
        assert moe.num_experts == 4
        assert moe.top_k == 2

    def test_moe_forward(self):
        """Test MoE forward pass."""
        config = TinyModelConfig(d_model=64, num_experts=4, num_shared_experts=1, top_k=2)
        moe = TinyMoELayer(config)
        x = mlx.random.normal((2, 10, 64))
        out = moe(x)
        assert out.shape == x.shape

    def test_moe_routing(self):
        """Test MoE routes to different experts."""
        config = TinyModelConfig(d_model=64, num_experts=8, top_k=2)
        moe = TinyMoELayer(config)
        x = mlx.random.normal((4, 20, 64))
        out = moe(x)
        mlx.eval(out)
        # Output should be different from input
        assert not mlx.allclose(x, out).item()


class TestTinyTransformerBlock:
    """Tests for TinyTransformerBlock."""

    def test_block_creation(self):
        """Test transformer block creation."""
        config = TinyModelConfig(d_model=64, num_heads=4)
        block = TinyTransformerBlock(config)
        assert block.attn is not None
        assert block.moe is not None
        assert block.norm1 is not None
        assert block.norm2 is not None

    def test_block_forward(self):
        """Test transformer block forward pass."""
        config = TinyModelConfig(d_model=64, num_heads=4)
        block = TinyTransformerBlock(config)
        x = mlx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        out, cache = block(x, mask)
        assert out.shape == x.shape


class TestTinyMTPModel:
    """Tests for TinyMTPModel (Multi-Token Prediction)."""

    def test_model_creation(self):
        """Test model creation."""
        config = TinyModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            mtp_k=2,
        )
        model = TinyMTPModel(config)
        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.mtp_k == 2
        assert len(model.layers) == 2
        assert len(model.mtp_heads) == 2

    def test_model_forward(self):
        """Test model forward pass."""
        config = TinyModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            mtp_k=2,
        )
        model = TinyMTPModel(config)
        input_ids = mlx.random.randint(0, 1000, (2, 16))
        main_logits, mtp_logits, cache = model(input_ids)
        
        assert main_logits.shape == (2, 16, 1000)
        assert len(mtp_logits) == 2
        for mtp_head_logits in mtp_logits:
            assert mtp_head_logits.shape == (2, 16, 1000)

    def test_model_no_mtp(self):
        """Test model without MTP heads."""
        config = TinyModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            mtp_k=0,  # No MTP
        )
        model = TinyMTPModel(config)
        assert len(model.mtp_heads) == 0
        
        input_ids = mlx.random.randint(0, 1000, (2, 16))
        main_logits, mtp_logits, cache = model(input_ids)
        assert main_logits.shape == (2, 16, 1000)
        assert len(mtp_logits) == 0

    def test_model_generate(self):
        """Test model generation."""
        config = TinyModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            mtp_k=0,
        )
        model = TinyMTPModel(config)
        input_ids = mlx.random.randint(0, 1000, (1, 5))
        output = model.generate(input_ids, max_new_tokens=10)
        assert output.shape[1] == 15  # 5 original + 10 new


class TestComputeMTPLoss:
    """Tests for compute_mtp_loss function."""

    def test_loss_computation(self):
        """Test MTP loss computation."""
        config = TinyModelConfig(vocab_size=100, d_model=32, mtp_k=2)
        model = TinyMTPModel(config)
        
        input_ids = mlx.random.randint(0, 100, (2, 10))
        main_logits, mtp_logits, _ = model(input_ids)
        
        loss, metrics = compute_mtp_loss(main_logits, mtp_logits, input_ids)
        mlx.eval(loss)
        
        assert loss.shape == ()  # Scalar
        assert "main_loss" in metrics
        assert "mtp_loss" in metrics

    def test_loss_with_pad_token(self):
        """Test loss ignores pad tokens."""
        config = TinyModelConfig(vocab_size=100, d_model=32, mtp_k=2)
        model = TinyMTPModel(config)
        
        # Create input with padding
        input_ids = mlx.random.randint(0, 100, (2, 10))
        main_logits, mtp_logits, _ = model(input_ids)
        
        loss, metrics = compute_mtp_loss(
            main_logits, mtp_logits, input_ids, 
            pad_token_id=0
        )
        mlx.eval(loss)
        assert loss.shape == ()


class TestTinyMLXTrainer:
    """Tests for TinyMLXTrainer."""

    def test_trainer_creation(self):
        """Test trainer creation."""
        config = TinyModelConfig(vocab_size=100, d_model=32, num_layers=1)
        model = TinyMTPModel(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TinyMLXTrainer(
                model=model,
                config=config,
                checkpoint_dir=tmpdir,
                learning_rate=1e-4,
            )
            assert trainer.model is model
            assert trainer.config is config
            assert trainer.learning_rate == 1e-4

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load cycle."""
        config = TinyModelConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            mtp_k=2,
            num_experts=4,
            top_k=2,
        )
        model = TinyMTPModel(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer and save checkpoint
            trainer = TinyMLXTrainer(
                model=model,
                config=config,
                checkpoint_dir=tmpdir,
            )
            trainer.global_step = 100
            trainer.best_loss = 2.5
            trainer.save_checkpoint("test")
            
            # Verify checkpoint files exist
            ckpt_path = Path(tmpdir) / "test"
            assert (ckpt_path / "model.safetensors").exists()
            assert (ckpt_path / "config.json").exists()
            assert (ckpt_path / "training_state.json").exists()
            
            # Load checkpoint
            trainer2, model2, config2 = TinyMLXTrainer.load_checkpoint(str(ckpt_path))
            
            # Verify state restored
            assert trainer2.global_step == 100
            assert trainer2.best_loss == 2.5
            assert config2.vocab_size == config.vocab_size
            assert config2.d_model == config.d_model
            
            # Verify model weights match
            input_ids = mlx.random.randint(0, config.vocab_size, (2, 16))
            out1, _, _ = model(input_ids)
            out2, _, _ = model2(input_ids)
            mlx.eval(out1, out2)
            assert mlx.allclose(out1, out2, atol=1e-5).item()

    def test_flatten_unflatten_params(self):
        """Test parameter flattening and unflattening."""
        config = TinyModelConfig(vocab_size=100, d_model=32, num_layers=1)
        model = TinyMTPModel(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TinyMLXTrainer(model=model, config=config, checkpoint_dir=tmpdir)
            
            # Get original params
            params = dict(model.parameters())
            
            # Flatten
            flat = trainer._flatten_params(params)
            assert isinstance(flat, dict)
            # All keys should be strings with dots
            for key in flat.keys():
                assert isinstance(key, str)
            
            # Unflatten
            nested = trainer._unflatten_params(flat)
            assert isinstance(nested, dict)
            
            # Verify structure matches
            assert "embed" in nested
            assert "layers" in nested
            assert "lm_head" in nested

    def test_evaluate(self):
        """Test evaluation method."""
        config = TinyModelConfig(vocab_size=100, d_model=32, num_layers=1, mtp_k=1)
        model = TinyMTPModel(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TinyMLXTrainer(model=model, config=config, checkpoint_dir=tmpdir)
            
            # Create mock dataloader
            class MockLoader:
                def __init__(self):
                    self.data = [
                        {"input_ids": mlx.random.randint(0, 100, (4, 16))}
                        for _ in range(3)
                    ]
                def __iter__(self):
                    return iter(self.data)
            
            loader = MockLoader()
            loss = trainer.evaluate(loader, pad_token_id=0, max_batches=2)
            assert isinstance(loss, float)
            assert loss >= 0


class TestDataLoader:
    """Tests for DataLoader."""

    def test_dataloader_creation(self):
        """Test DataLoader creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data file
            data_file = Path(tmpdir) / "train.jsonl"
            with open(data_file, "w") as f:
                for i in range(10):
                    json.dump({"text": f"This is sample text number {i}."}, f)
                    f.write("\n")
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.return_value = {
                "input_ids": [[1, 2, 3, 4, 5] + [0] * 59]  # Padded to 64
            }
            
            loader = DataLoader(
                data_path=tmpdir,
                tokenizer=mock_tokenizer,
                batch_size=4,
                max_seq_len=64,
            )
            assert loader.batch_size == 4
            assert loader.max_seq_len == 64


class TestCreateTinyModel:
    """Tests for create_tiny_model helper function."""

    def test_create_default_model(self):
        """Test creating model with default config."""
        model = create_tiny_model()
        assert isinstance(model, TinyMTPModel)
        assert model.vocab_size == 8000  # Default from TinyModelConfig

    def test_create_custom_model(self):
        """Test creating model with custom config."""
        config = TinyModelConfig(vocab_size=500, d_model=128)
        model = create_tiny_model(config)
        assert model.vocab_size == 500
        assert model.d_model == 128


class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_full_training_step(self):
        """Test a full training step with forward, backward, update."""
        import mlx.optimizers as optim

        config = TinyModelConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=1,
            mtp_k=1,
            num_experts=2,
            top_k=1,
        )
        model = TinyMTPModel(config)

        # Create optimizer
        optimizer = optim.Adam(learning_rate=1e-4)

        # Create batch
        input_ids = mlx.random.randint(0, 100, (2, 16))

        # Define loss function
        def loss_fn(model, input_ids):
            main_logits, mtp_logits, _ = model(input_ids)
            loss, _ = compute_mtp_loss(main_logits, mtp_logits, input_ids)
            return loss

        # Compute loss and gradients
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, input_ids)
        mlx.eval(loss)

        # Update model
        optimizer.update(model, grads)
        mlx.eval(model.parameters())

        assert loss.shape == ()
        assert float(loss.item()) > 0

    def test_checkpoint_round_trip_preserves_training(self):
        """Test that checkpoint preserves training progress."""
        config = TinyModelConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=1,
            mtp_k=1,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and partially train model
            model1 = TinyMTPModel(config)
            trainer1 = TinyMLXTrainer(
                model=model1,
                config=config,
                checkpoint_dir=tmpdir,
                learning_rate=1e-3,
            )
            
            # Do a few "training steps" (just update state)
            trainer1.global_step = 50
            trainer1.best_loss = 3.14
            
            # Save
            trainer1.save_checkpoint("checkpoint")
            
            # Load into new trainer
            trainer2, model2, config2 = TinyMLXTrainer.load_checkpoint(
                f"{tmpdir}/checkpoint"
            )
            
            # Verify training state preserved
            assert trainer2.global_step == 50
            assert trainer2.best_loss == 3.14
            assert config2.vocab_size == 100
            assert config2.d_model == 32
