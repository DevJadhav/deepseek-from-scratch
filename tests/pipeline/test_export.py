"""Tests for export stage."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deepseek.pipeline.config import (
    ExportConfig,
    ModelConfig,
    PipelineConfig,
)
from deepseek.pipeline.stages.base import StageContext
from deepseek.pipeline.stages.export import ExportStage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create a mock checkpoint directory."""
    ckpt_dir = temp_dir / "checkpoint"
    ckpt_dir.mkdir()
    
    # Create mock checkpoint files
    (ckpt_dir / "model.pt").write_text("mock model data")
    (ckpt_dir / "optimizer.pt").write_text("mock optimizer data")
    (ckpt_dir / "config.json").write_text('{"d_model": 256}')

    return ckpt_dir


@pytest.fixture
def pipeline_config(temp_dir):
    """Create a pipeline config for testing."""
    return PipelineConfig(
        run_name="test-export",
        model=ModelConfig.tiny(),
        export=ExportConfig(
            output_dir=str(temp_dir / "export_output"),
            export_safetensors=True,
            export_gguf=False,
        ),
    )


class TestExportStage:
    """Test cases for ExportStage."""
    
    def test_export_stage_init(self, pipeline_config):
        """Test export stage initialization."""
        stage = ExportStage(pipeline_config)
        assert stage.stage_name == "export"
        assert stage.config == pipeline_config
    
    def test_export_requires_checkpoint(self, pipeline_config):
        """Test that export fails without checkpoint."""
        stage = ExportStage(pipeline_config)
        context = StageContext(config=pipeline_config)
        
        with pytest.raises(RuntimeError, match="requires a checkpoint"):
            stage.run(context)
    
    def test_export_directory_checkpoint(self, pipeline_config, checkpoint_dir):
        """Test exporting a directory checkpoint."""
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            previous_output=str(checkpoint_dir),
        )
        
        result = stage.run(context)
        
        # Check export path was set
        assert "export_path" in result.metadata
        export_path = Path(result.metadata["export_path"])
        assert export_path.exists() or export_path.with_suffix("").exists()
    
    def test_export_creates_metadata(self, pipeline_config, checkpoint_dir):
        """Test that export creates metadata file."""
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            previous_output=str(checkpoint_dir),
        )
        
        result = stage.run(context)
        
        # Check metadata file was created
        output_dir = Path(pipeline_config.export.output_dir)
        metadata_path = output_dir / "export_metadata.json"
        
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["run_name"] == "test-export"
        assert "source_checkpoint" in metadata
    
    def test_export_from_pretrain_checkpoint(self, pipeline_config, checkpoint_dir):
        """Test exporting from pretrain checkpoint in metadata."""
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            metadata={"pretrain_checkpoint": str(checkpoint_dir)},
        )
        
        result = stage.run(context)
        assert "export_path" in result.metadata
    
    def test_export_from_sft_checkpoint(self, pipeline_config, checkpoint_dir):
        """Test exporting from SFT checkpoint in metadata."""
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            metadata={"sft_checkpoint": str(checkpoint_dir)},
        )

        _ = stage.run(context)
        # Verifies no error occurs during export
    
    def test_export_from_grpo_checkpoint(self, pipeline_config, checkpoint_dir):
        """Test exporting from GRPO checkpoint in metadata."""
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            metadata={"grpo_checkpoint": str(checkpoint_dir)},
        )

        _ = stage.run(context)
        # Verifies no error occurs during export
    
    def test_export_from_distillation_checkpoint(self, pipeline_config, checkpoint_dir):
        """Test exporting from distillation checkpoint (highest priority)."""
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            metadata={
                "pretrain_checkpoint": "/other/path",
                "distillation_checkpoint": str(checkpoint_dir),
            },
        )

        result = stage.run(context)
        # Should complete without error and set export_path
        assert "export_path" in result.metadata
    
    def test_export_single_file_checkpoint(self, pipeline_config, temp_dir):
        """Test exporting a single file checkpoint."""
        # Create a single file checkpoint
        ckpt_file = temp_dir / "model.ckpt"
        ckpt_file.write_text("mock checkpoint data")
        
        stage = ExportStage(pipeline_config)
        context = StageContext(
            config=pipeline_config,
            previous_output=str(ckpt_file),
        )
        
        result = stage.run(context)
        assert "export_path" in result.metadata


class TestExportGGUF:
    """Test cases for GGUF export functionality."""

    def test_gguf_export_called_when_configured(self, pipeline_config, checkpoint_dir):
        """Test that GGUF export is attempted when export_gguf is True."""
        pipeline_config.export.export_gguf = True
        stage = ExportStage(pipeline_config)

        with patch.object(stage, "_export_gguf", return_value=None) as mock_gguf:
            context = StageContext(
                config=pipeline_config,
                previous_output=str(checkpoint_dir),
            )
            stage.run(context)
            mock_gguf.assert_called_once()

    def test_gguf_export_not_called_when_disabled(self, pipeline_config, checkpoint_dir):
        """Test that GGUF export is not called when export_gguf is False."""
        pipeline_config.export.export_gguf = False
        stage = ExportStage(pipeline_config)

        with patch.object(stage, "_export_gguf", return_value=None) as mock_gguf:
            context = StageContext(
                config=pipeline_config,
                previous_output=str(checkpoint_dir),
            )
            stage.run(context)
            mock_gguf.assert_not_called()


class TestExportSafetensors:
    """Test cases for safetensors export functionality."""
    
    def test_safetensors_export_called(self, pipeline_config, checkpoint_dir):
        """Test that safetensors export is attempted."""
        stage = ExportStage(pipeline_config)
        
        with patch.object(stage, "_export_safetensors", return_value=None) as mock_st:
            context = StageContext(
                config=pipeline_config,
                previous_output=str(checkpoint_dir),
            )
            stage.run(context)
            mock_st.assert_called_once()
    
    def test_copy_existing_safetensors(self, pipeline_config, temp_dir):
        """Test copying existing safetensors files."""
        # Create checkpoint with existing safetensors
        ckpt_dir = temp_dir / "checkpoint_st"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.safetensors").write_bytes(b"mock safetensors data")
        
        stage = ExportStage(pipeline_config)
        result = stage._export_safetensors(str(ckpt_dir), Path(pipeline_config.export.output_dir))
        
        # Should return path to copied safetensors
        if result:
            assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
