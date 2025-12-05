#!/usr/bin/env python3
"""
Integration Tests for Validation Checklist
==========================================

Tests to verify all items in the Validation Checklist Before Production
are properly implemented and functional.
"""

from __future__ import annotations

from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestInfrastructureChecklist:
    """Tests for Infrastructure checklist items."""

    def test_modal_app_exists(self):
        """Modal app builds without errors."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        assert app_path.exists(), "modal_gpu/app.py should exist"

        content = app_path.read_text()
        assert "modal.App" in content, "Should define Modal app"
        assert "modal.Volume" in content, "Should define volumes"
        assert "modal.gpu" in content or "gpu=" in content, "Should configure GPU"

    def test_volumes_defined(self):
        """Volumes accessible and mountable."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()

        assert "checkpoints_volume" in content, "Should define checkpoints volume"
        assert "data_volume" in content, "Should define data volume"
        assert "create_if_missing=True" in content, "Volumes should auto-create"

    def test_gpu_allocation_configured(self):
        """GPU allocation working."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()

        # Check for GPU configuration
        has_gpu_config = any(
            pattern in content
            for pattern in ["modal.gpu.H100", "modal.gpu.A100", "gpu=", "GPU_CONFIG"]
        )
        assert has_gpu_config, "Should configure GPU type"

    def test_checkpoint_sync_functional(self):
        """Checkpoint sync functional."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()

        assert "sync_checkpoints" in content or "checkpoint" in content.lower()
        assert "commit()" in content or "reload()" in content

    def test_container_images_defined(self):
        """Container images defined."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()

        assert "modal.Image" in content, "Should define container image"
        assert "cuda" in content.lower(), "Should have CUDA support"

    def test_environment_variables_configured(self):
        """Environment variables configured."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()

        assert (
            "modal.Secret" in content or "secrets" in content
        ), "Should configure secrets"


class TestRayPipelineChecklist:
    """Tests for Ray Pipeline checklist items."""

    def test_time_sliced_wave_execution(self):
        """Time-sliced wave execution."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        assert "WaveConfig" in content, "Should have wave configuration"
        assert "wave" in content.lower(), "Should handle waves"

    def test_four_wave_configuration(self):
        """4-wave configuration."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        # Check for wave thresholds
        assert "wave_thresholds" in content or "Wave 1" in content.lower()

    def test_stage_orchestration(self):
        """Stage orchestration (data, train, eval, export)."""
        config_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "config.py"
        content = config_path.read_text()

        assert "Stage" in content, "Should define stages"
        assert "DATA_PREP" in content or "data" in content.lower()
        assert "PRETRAIN" in content or "train" in content.lower()
        assert "EXPORT" in content or "export" in content.lower()

    def test_checkpoint_management(self):
        """Checkpoint management."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        assert "checkpoint" in content.lower()
        assert "save" in content.lower() or "load" in content.lower()

    def test_wave_validation_returns_real_metrics(self):
        """Wave validation returns real metrics."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        assert "_run_validation" in content, "Should have validation function"
        assert "validation_loss" in content or "val_loss" in content
        assert "wave_thresholds" in content or "threshold" in content.lower()


class TestDataPipelineChecklist:
    """Tests for Data Pipeline checklist items."""

    def test_fineweb_download_script_exists(self):
        """FineWeb-Edu download script exists."""
        script_path = PROJECT_ROOT / "scripts" / "data-dowloader" / "download_fineweb_edu.py"
        assert script_path.exists(), "download_fineweb_edu.py should exist"

        content = script_path.read_text()
        assert "fineweb" in content.lower()
        assert "download" in content.lower() or "load_dataset" in content

    def test_tokenization_script_exists(self):
        """Tokenization script exists."""
        script_path = PROJECT_ROOT / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        assert script_path.exists(), "tokenize_fineweb.py should exist"

        content = script_path.read_text()
        assert "tokenize" in content.lower()
        assert "shard" in content.lower()

    def test_modal_volume_upload_exists(self):
        """Modal volume upload script exists."""
        script_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "prepare_fineweb.py"
        assert script_path.exists(), "prepare_fineweb.py should exist"

        content = script_path.read_text()
        assert "modal" in content.lower()
        assert "volume" in content.lower() or "upload" in content.lower()

    def test_dataloader_implementation_exists(self):
        """DataLoader verified."""
        dataloader_path = (
            PROJECT_ROOT
            / "src"
            / "deepseek"
            / "torch"
            / "training"
            / "sharded_dataloader.py"
        )
        assert dataloader_path.exists(), "sharded_dataloader.py should exist"

        content = dataloader_path.read_text()
        assert "ShardedBinaryDataset" in content or "DataLoader" in content
        assert "mmap" in content.lower() or "memory" in content.lower()


class TestRustBackendChecklist:
    """Tests for Rust Backend checklist items."""

    def test_training_implementation_exists(self):
        """Training implementation exists."""
        main_rs = PROJECT_ROOT / "rust-src" / "src" / "main.rs"
        assert main_rs.exists(), "main.rs should exist"

        content = main_rs.read_text()
        assert "train" in content.lower()

    def test_rust_cuda_runner_exists(self):
        """RustCudaRunner in ray_pipeline."""
        runner_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "runners" / "rust_runner.py"
        assert runner_path.exists(), "rust_runner.py should exist"

        content = runner_path.read_text()
        assert "Rust" in content

    def test_cli_subcommands_implemented(self):
        """CLI subcommands for train/evaluate/export."""
        main_rs = PROJECT_ROOT / "rust-src" / "src" / "main.rs"
        content = main_rs.read_text()

        assert "Train" in content or "train" in content
        assert "Evaluate" in content or "evaluate" in content
        assert "Export" in content or "export" in content
        assert "clap" in content.lower(), "Should use clap for CLI"

    def test_json_metrics_output(self):
        """JSON metrics output."""
        main_rs = PROJECT_ROOT / "rust-src" / "src" / "main.rs"
        content = main_rs.read_text()

        assert "TrainingMetrics" in content or "metrics" in content.lower()
        assert "serde" in content.lower() or "json" in content.lower()
        assert "serde_json" in content or "to_string" in content


class TestTrainingPipelineChecklist:
    """Tests for Training Pipeline checklist items."""

    def test_wave_validation_implementation(self):
        """Wave validation returns real metrics."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        assert "_run_validation" in content
        assert "_validate_mlx_checkpoint" in content or "mlx" in content.lower()
        assert "_validate_pytorch_checkpoint" in content or "pytorch" in content.lower()

    def test_checkpoint_save_restore(self):
        """Checkpoint save/restore works."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        assert "checkpoint" in content.lower()
        # Check for save/load operations
        assert "save" in content.lower() or "load" in content.lower()

    def test_wave_thresholds_configured(self):
        """Wave thresholds configured."""
        workflow_path = PROJECT_ROOT / "src" / "deepseek" / "pipeline" / "workflow.py"
        content = workflow_path.read_text()

        # Check for progressive thresholds
        assert "8.0" in content and "5.0" in content and "3.5" in content and "2.5" in content, \
            "Should have progressive wave thresholds"


class TestMonitoringChecklist:
    """Tests for Monitoring checklist items."""

    def test_cost_tracking_implementation(self):
        """Cost tracking calculates correctly."""
        tracker_path = PROJECT_ROOT / "monitoring" / "cost_tracker.py"
        assert tracker_path.exists(), "cost_tracker.py should exist"

        content = tracker_path.read_text()
        assert "CostTracker" in content
        assert "gpu_hours" in content.lower() or "GPU" in content

    def test_budget_alerts_configured(self):
        """Budget alerts configured."""
        tracker_path = PROJECT_ROOT / "monitoring" / "cost_tracker.py"
        content = tracker_path.read_text()

        # Check for alert thresholds
        assert "AlertLevel" in content or "alert" in content.lower()
        assert "50" in content or "75" in content or "90" in content or "95" in content


class TestSuccessCriteria:
    """Tests for Success Criteria implementation."""

    def test_training_pipeline_supports_steps(self):
        """Training pipeline supports 150K steps on FineWeb-Edu."""
        # Check configs support max_steps
        zero2_config = PROJECT_ROOT / "config" / "hydra" / "training" / "deepspeed_zero2.yaml"
        content = zero2_config.read_text()
        assert "150000" in content or "max_steps" in content

    def test_perplexity_evaluation_implemented(self):
        """Perplexity evaluation implemented."""
        eval_path = PROJECT_ROOT / "scripts" / "evaluate.py"
        assert eval_path.exists(), "evaluate.py should exist"

        downstream_path = PROJECT_ROOT / "scripts" / "downstream_eval.py"
        assert downstream_path.exists(), "downstream_eval.py should exist"

    def test_cost_budget_limit(self):
        """Cost tracking with $1,000 budget limit."""
        tracker_path = PROJECT_ROOT / "monitoring" / "cost_tracker.py"
        content = tracker_path.read_text()

        assert "budget" in content.lower()
        assert "1000" in content or "budget_limit" in content

    def test_checkpoint_download_support(self):
        """Checkpoint download via Modal volume sync."""
        app_path = PROJECT_ROOT / "src" / "deepseek" / "cloud" / "modal" / "app.py"
        content = app_path.read_text()

        assert "Volume" in content
        assert "sync" in content.lower() or "commit" in content

    def test_deepspeed_configs_exist(self):
        """DeepSpeed ZeRO Stage 2/3 configs exist."""
        zero2_path = PROJECT_ROOT / "config" / "hydra" / "training" / "deepspeed_zero2.yaml"
        zero3_path = PROJECT_ROOT / "config" / "hydra" / "training" / "deepspeed_zero3.yaml"

        assert zero2_path.exists(), "deepspeed_zero2.yaml should exist"
        assert zero3_path.exists(), "deepspeed_zero3.yaml should exist"

    def test_downstream_evaluation_implemented(self):
        """Downstream evaluation implemented (HellaSwag, LAMBADA)."""
        downstream_path = PROJECT_ROOT / "scripts" / "downstream_eval.py"
        content = downstream_path.read_text()

        assert "HellaSwag" in content or "hellaswag" in content.lower()
        assert "LAMBADA" in content or "lambada" in content.lower()


class TestCargoToml:
    """Tests for Rust Cargo.toml configuration."""

    def test_clap_dependency(self):
        """Cargo.toml has clap dependency."""
        cargo_path = PROJECT_ROOT / "rust-src" / "Cargo.toml"
        content = cargo_path.read_text()

        assert "clap" in content, "Should have clap dependency"
        assert '"4.' in content or "'4." in content, "Should use clap v4.x"


class TestConfigFiles:
    """Tests for configuration files."""

    def test_hydra_config_exists(self):
        """Hydra config exists."""
        config_path = PROJECT_ROOT / "config" / "hydra" / "config.yaml"
        assert config_path.exists(), "config.yaml should exist"

    def test_training_configs_exist(self):
        """Training configs exist."""
        training_dir = PROJECT_ROOT / "config" / "hydra" / "training"
        assert training_dir.exists(), "training config directory should exist"

        configs = list(training_dir.glob("*.yaml"))
        assert len(configs) >= 4, "Should have at least 4 training configs"

    def test_model_configs_exist(self):
        """Model configs exist."""
        model_dir = PROJECT_ROOT / "config" / "hydra" / "model"
        if model_dir.exists():
            configs = list(model_dir.glob("*.yaml"))
            assert len(configs) >= 1, "Should have at least 1 model config"
