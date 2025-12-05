"""
Tests for Modal GPU training utilities.

Tests cover:
- Model building functions
- Data loading utilities
- DeepSpeed configuration
- Model size configs
"""

import pytest
from pathlib import Path


class TestBuildModel:
    """Tests for model building utilities."""

    def test_build_model_returns_module(self):
        """Test that build_model returns a torch.nn.Module."""
        pytest.importorskip("torch")
        from deepseek.cloud.modal.training_utils import build_model

        config = {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
        }

        model = build_model(config)
        import torch.nn as nn

        assert isinstance(model, nn.Module)

    def test_build_model_has_forward(self):
        """Test that built model has forward method."""
        pytest.importorskip("torch")
        from deepseek.cloud.modal.training_utils import build_model

        config = {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
        }

        model = build_model(config)
        assert hasattr(model, "forward")
        assert callable(model.forward)

    def test_build_model_forward_pass(self):
        """Test that model can do a forward pass."""
        torch = pytest.importorskip("torch")
        from deepseek.cloud.modal.training_utils import build_model

        config = {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
        }

        model = build_model(config)
        model.eval()

        # Create dummy input
        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids)

        assert output.shape == (2, 16, 1000)


class TestModelSizeConfig:
    """Tests for model size configuration."""

    def test_get_tiny_config(self):
        """Test tiny model configuration."""
        from deepseek.cloud.modal.training_utils import get_model_size_config

        config = get_model_size_config("tiny")

        assert "hidden_size" in config
        assert "num_layers" in config
        assert "num_attention_heads" in config
        assert config["hidden_size"] == 256
        assert config["num_layers"] == 4

    def test_get_small_config(self):
        """Test small model configuration."""
        from deepseek.cloud.modal.training_utils import get_model_size_config

        config = get_model_size_config("small")

        assert config["hidden_size"] == 512
        assert config["num_layers"] == 8

    def test_get_medium_config(self):
        """Test medium model configuration."""
        from deepseek.cloud.modal.training_utils import get_model_size_config

        config = get_model_size_config("medium")

        assert config["hidden_size"] == 1024
        assert config["num_layers"] == 16

    def test_get_large_config(self):
        """Test large model configuration."""
        from deepseek.cloud.modal.training_utils import get_model_size_config

        config = get_model_size_config("large")

        assert config["hidden_size"] == 2048
        assert config["num_layers"] == 24

    def test_unknown_size_returns_tiny(self):
        """Test that unknown size returns tiny config."""
        from deepseek.cloud.modal.training_utils import get_model_size_config

        config = get_model_size_config("unknown_size")
        tiny_config = get_model_size_config("tiny")

        assert config == tiny_config


class TestDefaultTrainingConfig:
    """Tests for default training configuration."""

    def test_get_default_training_config(self):
        """Test default training configuration."""
        from deepseek.cloud.modal.training_utils import get_default_training_config

        config = get_default_training_config()

        assert "batch_size" in config
        assert "learning_rate" in config
        assert "max_steps" in config
        assert "warmup_steps" in config
        assert "use_amp" in config

    def test_get_training_config_custom_steps(self):
        """Test training config with custom max_steps."""
        from deepseek.cloud.modal.training_utils import get_default_training_config

        config = get_default_training_config(max_steps=5000)

        assert config["max_steps"] == 5000
        assert config["warmup_steps"] <= 500  # Should be fraction of max_steps


class TestDeepSpeedConfig:
    """Tests for DeepSpeed configuration builder."""

    def test_build_deepspeed_config_zero2(self):
        """Test DeepSpeed config with ZeRO stage 2."""
        from deepseek.cloud.modal.training_utils import build_deepspeed_config

        training_config = {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 4,
            "use_amp": True,
        }

        distributed_config = {
            "data_parallel_size": 4,
            "zero_stage": 2,
        }

        config = build_deepspeed_config(training_config, distributed_config)

        assert config["train_batch_size"] == 32  # 8 * 4
        assert config["gradient_accumulation_steps"] == 4
        assert config["optimizer"]["type"] == "AdamW"
        assert config["zero_optimization"]["stage"] == 2
        assert config["fp16"]["enabled"] is True

    def test_build_deepspeed_config_zero3(self):
        """Test DeepSpeed config with ZeRO stage 3."""
        from deepseek.cloud.modal.training_utils import build_deepspeed_config

        training_config = {
            "batch_size": 4,
            "learning_rate": 1e-4,
        }

        distributed_config = {
            "data_parallel_size": 8,
            "zero_stage": 3,
        }

        config = build_deepspeed_config(training_config, distributed_config)

        assert config["zero_optimization"]["stage"] == 3
        assert config["zero_optimization"]["offload_param"]["device"] == "cpu"


class TestParquetDataset:
    """Tests for ParquetDataset class."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return data_dir

    def test_dataset_handles_missing_path(self, tmp_path):
        """Test dataset handles non-existent path gracefully."""
        from deepseek.cloud.modal.training_utils import ParquetDataset

        dataset = ParquetDataset(tmp_path / "nonexistent")
        assert len(dataset) == 0

    def test_dataset_loads_jsonl(self, temp_data_dir):
        """Test dataset can load JSONL files."""
        import json

        # Create test JSONL file
        jsonl_file = temp_data_dir / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"Sample text {i}"}) + "\n")

        from deepseek.cloud.modal.training_utils import ParquetDataset

        dataset = ParquetDataset(temp_data_dir)
        assert len(dataset) == 10


class TestCollateBatch:
    """Tests for batch collation."""

    def test_collate_batch_pads_sequences(self):
        """Test that collate_batch properly pads sequences."""
        torch = pytest.importorskip("torch")
        from deepseek.cloud.modal.training_utils import collate_batch

        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])},
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])},
        ]

        result = collate_batch(batch)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        # Second sequence should be padded
        assert result["input_ids"][1, 2].item() == 0
        assert result["attention_mask"][1, 2].item() == 0
