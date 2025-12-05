"""Tests for workflow module including validation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deepseek.pipeline.config import (
    DataConfig,
    ExportConfig,
    ModelConfig,
    PipelineConfig,
    TimeSlicedConfig,
    TrainingConfig,
    WaveBackend,
    WaveConfig,
)
from deepseek.pipeline.workflow import DeepSeekWorkflow


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create a mock checkpoint directory with valid structure."""
    ckpt_dir = temp_dir / "checkpoint"
    ckpt_dir.mkdir()

    # Create valid checkpoint structure
    (ckpt_dir / "model.pt").write_bytes(b"mock model weights")
    (ckpt_dir / "optimizer.pt").write_bytes(b"mock optimizer state")
    (ckpt_dir / "config.json").write_text(json.dumps({
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
    }))
    (ckpt_dir / "training_state.json").write_text(json.dumps({
        "step": 1000,
        "best_loss": 3.5,
        "last_loss": 3.6,
    }))

    return ckpt_dir


@pytest.fixture
def validation_data_dir(temp_dir):
    """Create mock validation data."""
    val_dir = temp_dir / "data" / "valid"
    val_dir.mkdir(parents=True)

    # Create JSONL validation data
    val_file = val_dir / "validation.jsonl"
    with open(val_file, "w") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"This is validation text number {i}. " * 10}) + "\n")

    return val_dir.parent


@pytest.fixture
def pipeline_config(temp_dir, validation_data_dir):
    """Create a pipeline config for testing."""
    return PipelineConfig(
        run_name="test-workflow",
        model=ModelConfig.tiny(),
        training=TrainingConfig(
            max_steps=100,
            batch_size=4,
        ),
        data=DataConfig(
            data_dir=str(validation_data_dir),
        ),
        export=ExportConfig(
            output_dir=str(temp_dir / "export"),
        ),
    )


class TestDeepSeekWorkflow:
    """Test cases for DeepSeekWorkflow."""

    def test_workflow_init(self, pipeline_config):
        """Test workflow initialization."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)
            assert workflow.config == pipeline_config

    def test_workflow_init_ray(self, pipeline_config):
        """Test workflow can be initialized."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)
            # Workflow should be created successfully
            assert workflow is not None
            assert workflow.config == pipeline_config


class TestValidation:
    """Test cases for wave validation functionality."""

    def test_run_validation_returns_loss(self, pipeline_config, checkpoint_dir):
        """Test that validation returns a loss value."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)

        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.PYTORCH_CUDA,
            start_step=0,
            end_step=1000,
        )

        # Mock the validation methods to avoid requiring actual model
        with patch.object(workflow, "_validate_pytorch_checkpoint", return_value=3.5):
            loss = workflow._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_dir),
                metadata={"input_data": str(pipeline_config.data.data_dir)},
            )

        assert isinstance(loss, float)
        assert loss == 3.5

    def test_validation_fallback_to_training_loss(self, pipeline_config, checkpoint_dir):
        """Test fallback to training loss when validation fails."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)

        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.PYTORCH_CUDA,
            start_step=0,
            end_step=1000,
        )

        # Mock validation to raise exception
        with patch.object(
            workflow,
            "_validate_pytorch_checkpoint",
            side_effect=Exception("Validation error"),
        ):
            metadata = {"input_data": str(pipeline_config.data.data_dir)}
            loss = workflow._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_dir),
                metadata=metadata,
            )

        # Should fallback to best_loss from training_state.json
        assert loss == 3.5
        assert metadata["wave_1_val_loss"] == 3.5

    def test_validation_records_in_metadata(self, pipeline_config, checkpoint_dir):
        """Test that validation loss is recorded in metadata."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)

        wave = WaveConfig(
            wave_id=2,
            backend=WaveBackend.PYTORCH_CUDA,
            start_step=1000,
            end_step=2000,
        )

        metadata = {"input_data": str(pipeline_config.data.data_dir)}

        with patch.object(workflow, "_validate_pytorch_checkpoint", return_value=4.2):
            workflow._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_dir),
                metadata=metadata,
            )

        assert "wave_2_val_loss" in metadata
        assert metadata["wave_2_val_loss"] == 4.2


class TestValidationBackends:
    """Test validation for different backends."""

    def test_mlx_validation_called_for_mlx_backend(self, pipeline_config, checkpoint_dir):
        """Test MLX validation is called for MLX backend."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)

        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.MLX,
            start_step=0,
            end_step=1000,
        )

        with patch.object(workflow, "_validate_mlx_checkpoint", return_value=3.0) as mock:
            workflow._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_dir),
                metadata={},
            )
            mock.assert_called_once()

    def test_rust_validation_called_for_rust_backend(self, pipeline_config, checkpoint_dir):
        """Test Rust validation is called for Rust backend."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)

        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.RUST,
            start_step=0,
            end_step=1000,
        )

        with patch.object(workflow, "_validate_rust_checkpoint", return_value=3.0) as mock:
            workflow._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_dir),
                metadata={},
            )
            mock.assert_called_once()

    def test_pytorch_validation_called_for_python_backend(self, pipeline_config, checkpoint_dir):
        """Test PyTorch validation is called for Python backend."""
        with patch("ray.init"):
            workflow = DeepSeekWorkflow(pipeline_config)

        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.PYTORCH_CUDA,
            start_step=0,
            end_step=1000,
        )

        with patch.object(workflow, "_validate_pytorch_checkpoint", return_value=3.0) as mock:
            workflow._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_dir),
                metadata={},
            )
            mock.assert_called_once()


class TestTimeSlicedExecution:
    """Test time-sliced wave execution."""

    def test_wave_config_creation(self):
        """Test WaveConfig creation."""
        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.RUST,
            start_step=0,
            end_step=5000,
        )
        assert wave.wave_id == 1
        assert wave.backend == WaveBackend.RUST
        assert wave.num_steps == 5000

    def test_time_sliced_config_default(self):
        """Test TimeSlicedConfig creation with defaults."""
        time_config = TimeSlicedConfig(enabled=False)
        assert time_config.enabled is False
        assert time_config.num_waves == 4
        assert time_config.steps_per_wave == 5000

    def test_time_sliced_config_generates_waves_when_enabled(self):
        """Test that enabling time-slicing generates default waves."""
        time_config = TimeSlicedConfig(enabled=True)
        assert len(time_config.waves) == 4
        # Default waves: PYTORCH_CUDA -> MLX -> RUST -> CPU
        assert time_config.waves[0].backend == WaveBackend.PYTORCH_CUDA
        assert time_config.waves[1].backend == WaveBackend.MLX
        assert time_config.waves[2].backend == WaveBackend.RUST
        assert time_config.waves[3].backend == WaveBackend.CPU

    def test_wave_step_ranges(self):
        """Test that wave step ranges are correct."""
        time_config = TimeSlicedConfig(enabled=True)
        waves = time_config.waves

        # Wave 1: 0-5000
        assert waves[0].start_step == 0
        assert waves[0].end_step == 5000

        # Wave 2: 5000-10000
        assert waves[1].start_step == 5000
        assert waves[1].end_step == 10000

        # Wave 3: 10000-15000
        assert waves[2].start_step == 10000
        assert waves[2].end_step == 15000

        # Wave 4: 15000-20000
        assert waves[3].start_step == 15000
        assert waves[3].end_step == 20000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
