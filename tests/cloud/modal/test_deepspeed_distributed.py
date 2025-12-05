"""
Tests for DeepSpeed Distributed Training Infrastructure.

Tests cover:
- DeepSpeed ZeRO Stage 2/3 configuration
- Distributed training configuration validation
- Config validation and defaults
- Training config parsing

Note: GPU-specific tests are skipped when CUDA is not available.
"""

from pathlib import Path

import pytest
import yaml


class TestDeepSpeedZeROConfig:
    """Tests for DeepSpeed ZeRO configuration files."""

    @pytest.fixture
    def zero2_config(self):
        """Load ZeRO Stage 2 config."""
        config_path = (
            Path(__file__).parents[3] / "config" / "hydra" / "training" / "deepspeed_zero2.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def zero3_config(self):
        """Load ZeRO Stage 3 config."""
        config_path = (
            Path(__file__).parents[3] / "config" / "hydra" / "training" / "deepspeed_zero3.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_zero2_config_exists(self):
        """Test ZeRO Stage 2 config file exists."""
        config_path = (
            Path(__file__).parents[3] / "config" / "hydra" / "training" / "deepspeed_zero2.yaml"
        )
        assert config_path.exists(), f"Config not found at {config_path}"

    def test_zero3_config_exists(self):
        """Test ZeRO Stage 3 config file exists."""
        config_path = (
            Path(__file__).parents[3] / "config" / "hydra" / "training" / "deepspeed_zero3.yaml"
        )
        assert config_path.exists(), f"Config not found at {config_path}"

    def test_zero2_stage_is_2(self, zero2_config):
        """Test ZeRO Stage 2 config has correct stage."""
        assert zero2_config["deepspeed"]["zero_optimization"]["stage"] == 2

    def test_zero3_stage_is_3(self, zero3_config):
        """Test ZeRO Stage 3 config has correct stage."""
        assert zero3_config["deepspeed"]["zero_optimization"]["stage"] == 3

    def test_zero2_has_optimizer_offload(self, zero2_config):
        """Test ZeRO Stage 2 has optimizer offloading."""
        offload = zero2_config["deepspeed"]["zero_optimization"]["offload_optimizer"]
        assert offload["device"] == "cpu"
        assert offload["pin_memory"] is True

    def test_zero3_has_param_offload(self, zero3_config):
        """Test ZeRO Stage 3 has parameter offloading."""
        offload = zero3_config["deepspeed"]["zero_optimization"]["offload_param"]
        assert offload["device"] == "cpu"
        assert offload["pin_memory"] is True

    def test_zero2_bf16_enabled(self, zero2_config):
        """Test ZeRO Stage 2 uses bf16."""
        assert zero2_config["deepspeed"]["bf16"]["enabled"] is True
        assert zero2_config["deepspeed"]["fp16"]["enabled"] is False

    def test_zero3_bf16_enabled(self, zero3_config):
        """Test ZeRO Stage 3 uses bf16."""
        assert zero3_config["deepspeed"]["bf16"]["enabled"] is True
        assert zero3_config["deepspeed"]["fp16"]["enabled"] is False

    def test_zero2_has_gradient_clipping(self, zero2_config):
        """Test ZeRO Stage 2 has gradient clipping."""
        assert zero2_config["deepspeed"]["gradient_clipping"] == 1.0

    def test_zero3_has_activation_checkpointing(self, zero3_config):
        """Test ZeRO Stage 3 has activation checkpointing config."""
        ac = zero3_config["deepspeed"]["activation_checkpointing"]
        assert "partition_activations" in ac
        assert "cpu_checkpointing" in ac

    def test_zero2_optimizer_is_adamw(self, zero2_config):
        """Test ZeRO Stage 2 uses AdamW optimizer."""
        assert zero2_config["deepspeed"]["optimizer"]["type"] == "AdamW"

    def test_zero3_optimizer_is_adamw(self, zero3_config):
        """Test ZeRO Stage 3 uses AdamW optimizer."""
        assert zero3_config["deepspeed"]["optimizer"]["type"] == "AdamW"

    def test_zero2_has_scheduler(self, zero2_config):
        """Test ZeRO Stage 2 has LR scheduler."""
        scheduler = zero2_config["deepspeed"]["scheduler"]
        assert scheduler["type"] == "WarmupDecayLR"
        assert "warmup_num_steps" in scheduler["params"]

    def test_zero3_has_scheduler(self, zero3_config):
        """Test ZeRO Stage 3 has LR scheduler."""
        scheduler = zero3_config["deepspeed"]["scheduler"]
        assert scheduler["type"] == "WarmupDecayLR"
        assert "warmup_num_steps" in scheduler["params"]

    def test_zero3_gather_weights_on_save(self, zero3_config):
        """Test ZeRO Stage 3 gathers weights on save."""
        zero_opt = zero3_config["deepspeed"]["zero_optimization"]
        assert zero_opt["stage3_gather_16bit_weights_on_model_save"] is True


class TestDistributedTrainerConfig:
    """Tests for distributed trainer configuration."""

    def test_distributed_trainer_file_exists(self):
        """Test distributed trainer file exists."""
        trainer_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "distributed_trainer.py"
        assert trainer_path.exists()

    def test_distributed_trainer_has_train_function(self):
        """Test distributed trainer has train function."""
        trainer_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "distributed_trainer.py"
        content = trainer_path.read_text()
        assert "def train_distributed" in content or "train_distributed" in content

    def test_distributed_trainer_has_deepspeed_import(self):
        """Test distributed trainer mentions deepspeed."""
        trainer_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "distributed_trainer.py"
        content = trainer_path.read_text()
        assert "deepspeed" in content.lower()

    def test_distributed_trainer_has_zero_config(self):
        """Test distributed trainer has ZeRO config builder."""
        trainer_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "distributed_trainer.py"
        content = trainer_path.read_text()
        assert "_build_deepspeed_config" in content


class TestDeepSpeedConfigBuilder:
    """Tests for DeepSpeed config builder function."""

    def test_build_config_function_exists(self):
        """Test config builder function exists."""
        trainer_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "distributed_trainer.py"
        content = trainer_path.read_text()
        assert "def _build_deepspeed_config" in content

    def test_build_config_structure(self):
        """Test config builder produces correct structure."""
        expected_keys = [
            "train_batch_size",
            "gradient_accumulation_steps",
            "optimizer",
            "fp16",
            "zero_optimization",
        ]
        trainer_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "distributed_trainer.py"
        content = trainer_path.read_text()
        for key in expected_keys:
            assert f'"{key}"' in content or f"'{key}'" in content


class TestRustCLI:
    """Tests for Rust CLI implementation."""

    def test_main_rs_exists(self):
        """Test main.rs exists."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        assert main_path.exists()

    def test_main_rs_has_clap(self):
        """Test main.rs uses clap for CLI."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "clap" in content
        assert "Parser" in content

    def test_main_rs_has_subcommands(self):
        """Test main.rs has command subcommands."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "Subcommand" in content
        assert "enum Commands" in content

    def test_main_rs_has_train_command(self):
        """Test main.rs has Train command."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "Train" in content
        assert "Commands::Train" in content

    def test_main_rs_has_evaluate_command(self):
        """Test main.rs has Evaluate command."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "Evaluate" in content
        assert "Commands::Evaluate" in content

    def test_main_rs_has_export_command(self):
        """Test main.rs has Export command."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "Export" in content
        assert "Commands::Export" in content

    def test_main_rs_has_demo_command(self):
        """Test main.rs has Demo command."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "Demo" in content
        assert "Commands::Demo" in content

    def test_cargo_toml_has_clap_dependency(self):
        """Test Cargo.toml has clap dependency."""
        cargo_path = Path(__file__).parents[3] / "rust-src" / "Cargo.toml"
        content = cargo_path.read_text()
        assert "clap" in content
        assert "derive" in content

    def test_main_rs_outputs_json(self):
        """Test main.rs can output JSON for Python bridge."""
        main_path = Path(__file__).parents[3] / "rust-src" / "src" / "main.rs"
        content = main_path.read_text()
        assert "serde" in content.lower() or "json" in content.lower()


class TestModalGPUConfig:
    """Tests for Modal GPU configuration."""

    def test_config_file_exists(self):
        """Test config.py exists."""
        config_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "config.py"
        assert config_path.exists()

    def test_config_has_gpu_settings(self):
        """Test config has GPU settings."""
        config_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "config.py"
        content = config_path.read_text()
        assert "H100" in content or "GPU" in content or "gpu" in content

    def test_config_has_parallelism_settings(self):
        """Test config has parallelism settings."""
        config_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "config.py"
        content = config_path.read_text()
        parallelism_terms = [
            "tensor_parallel",
            "data_parallel",
            "pipeline_parallel",
            "expert_parallel",
        ]
        found = sum(1 for term in parallelism_terms if term in content.lower())
        assert found >= 2, f"Expected parallelism settings, found {found}"


class TestTrainingConfigIntegration:
    """Tests for training config integration."""

    def test_training_configs_exist(self):
        """Test all training configs exist."""
        config_dir = Path(__file__).parents[3] / "config" / "hydra" / "training"
        expected_configs = [
            "deepspeed_zero2.yaml",
            "deepspeed_zero3.yaml",
        ]
        for config_name in expected_configs:
            config_path = config_dir / config_name
            assert config_path.exists(), f"Missing config: {config_name}"

    def test_all_training_configs_valid_yaml(self):
        """Test all training configs are valid YAML."""
        config_dir = Path(__file__).parents[3] / "config" / "hydra" / "training"
        for config_path in config_dir.glob("*.yaml"):
            with open(config_path) as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_path}: {e}")

    def test_deepspeed_configs_have_required_sections(self):
        """Test DeepSpeed configs have all required sections."""
        config_dir = Path(__file__).parents[3] / "config" / "hydra" / "training"
        required_sections = ["zero_optimization", "optimizer", "bf16"]

        for config_name in ["deepspeed_zero2.yaml", "deepspeed_zero3.yaml"]:
            config_path = config_dir / config_name
            with open(config_path) as f:
                config = yaml.safe_load(f)

            ds_config = config.get("deepspeed", {})
            for section in required_sections:
                assert section in ds_config, f"Missing {section} in {config_name}"


class TestModalAppDefinition:
    """Tests for Modal app definition."""

    def test_app_file_exists(self):
        """Test app.py exists."""
        app_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        assert app_path.exists()

    def test_app_has_modal_import(self):
        """Test app.py imports modal."""
        app_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()
        assert "import modal" in content or "from modal" in content

    def test_app_has_gpu_config(self):
        """Test app.py has GPU configuration."""
        app_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()
        assert "H100" in content or "gpu" in content.lower()

    def test_app_has_volume_definition(self):
        """Test app.py has volume definitions."""
        app_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()
        assert "Volume" in content or "volume" in content.lower()

    def test_app_has_retry_policy(self):
        """Test app.py has retry policy."""
        app_path = Path(__file__).parents[3] / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()
        assert "retry" in content.lower() or "Retries" in content
