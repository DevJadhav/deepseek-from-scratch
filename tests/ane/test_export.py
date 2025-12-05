"""
Tests for CoreML Export Pipeline

This module tests the CoreML export utilities for ANE-optimized models.
"""

import torch

from deepseek.mlx.ane.export.coreml_export import (
    ANEExportConfig,
    ComputeUnit,
    CoreMLExporter,
    CoreMLOptimizationConfig,
    TracingWrapper,
)
from deepseek.mlx.ane.model.transformer import ANEDeepSeekConfig, ANEDeepSeekModel


class TestComputeUnit:
    """Tests for ComputeUnit enum."""

    def test_compute_unit_values(self):
        """Test ComputeUnit enum values."""
        assert ComputeUnit.ALL.value == "ALL"
        assert ComputeUnit.CPU_AND_GPU.value == "CPU_AND_GPU"
        assert ComputeUnit.CPU_AND_NE.value == "CPU_AND_NE"
        assert ComputeUnit.CPU_ONLY.value == "CPU_ONLY"

    def test_to_coreml_mapping(self):
        """Test conversion to coreml string."""
        assert ComputeUnit.ALL.to_coreml() == "ALL"
        assert ComputeUnit.CPU_ONLY.to_coreml() == "CPU_ONLY"


class TestCoreMLOptimizationConfig:
    """Tests for CoreMLOptimizationConfig."""

    def test_default_config(self):
        """Test default optimization configuration."""
        config = CoreMLOptimizationConfig()

        assert config.fuse_matmul_add is True
        assert config.fuse_conv_bn is True
        assert config.fold_constants is True
        assert config.quantize_weights is False
        assert config.float16_inference is True

    def test_custom_config(self):
        """Test custom optimization configuration."""
        config = CoreMLOptimizationConfig(
            quantize_weights=True,
            float16_inference=False,
        )

        assert config.quantize_weights is True
        assert config.float16_inference is False


class TestANEExportConfig:
    """Tests for ANEExportConfig."""

    def test_default_config(self):
        """Test default export configuration."""
        config = ANEExportConfig()

        assert config.batch_size == 1
        assert config.sequence_length == 128
        assert config.compute_units == ComputeUnit.ALL
        assert config.model_name == "DeepSeekANE"
        assert config.validate_numerics is True
        assert config.numeric_tolerance == 1e-3

    def test_custom_config(self):
        """Test custom export configuration."""
        config = ANEExportConfig(
            batch_size=2,
            sequence_length=256,
            compute_units=ComputeUnit.CPU_AND_NE,
            model_name="CustomModel",
        )

        assert config.batch_size == 2
        assert config.sequence_length == 256
        assert config.compute_units == ComputeUnit.CPU_AND_NE
        assert config.model_name == "CustomModel"

    def test_optimization_default_creation(self):
        """Test that optimization config is created by default."""
        config = ANEExportConfig()

        assert config.optimization is not None
        assert hasattr(config.optimization, 'fuse_matmul_add')


class TestTracingWrapper:
    """Tests for TracingWrapper."""

    def test_wrapper_creation(self):
        """Test creating a tracing wrapper."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)

        export_config = ANEExportConfig()
        wrapper = TracingWrapper(model, export_config)

        assert wrapper.model is model
        assert wrapper.config is export_config

    def test_wrapper_forward(self):
        """Test forward pass through wrapper."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        export_config = ANEExportConfig(sequence_length=32)
        wrapper = TracingWrapper(model, export_config)
        wrapper.eval()

        input_ids = torch.randint(0, model_config.vocab_size, (1, 32))

        with torch.no_grad():
            output = wrapper(input_ids)

        assert output.shape == (1, 32, model_config.vocab_size)


class TestCoreMLExporter:
    """Tests for CoreMLExporter."""

    def test_exporter_creation(self):
        """Test creating an exporter."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)

        export_config = ANEExportConfig()
        exporter = CoreMLExporter(model, export_config)

        assert exporter.model is model
        assert exporter.config is export_config

    def test_tracing(self):
        """Test model tracing."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        export_config = ANEExportConfig(sequence_length=32)
        exporter = CoreMLExporter(model, export_config)

        traced = exporter.trace()

        assert traced is not None
        assert isinstance(traced, torch.jit.ScriptModule)

    def test_traced_model_forward(self):
        """Test forward pass through traced model."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        export_config = ANEExportConfig(sequence_length=32)
        exporter = CoreMLExporter(model, export_config)

        traced = exporter.trace()

        input_ids = torch.randint(0, model_config.vocab_size, (1, 32))

        with torch.no_grad():
            output = traced(input_ids)

        assert output.shape == (1, 32, model_config.vocab_size)

    def test_traced_matches_original(self):
        """Test that traced model output matches original."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        export_config = ANEExportConfig(sequence_length=32)
        exporter = CoreMLExporter(model, export_config)

        traced = exporter.trace()

        input_ids = torch.randint(0, model_config.vocab_size, (1, 32))

        with torch.no_grad():
            original_output, _ = model(input_ids, use_cache=False)
            traced_output = traced(input_ids)

        # Should be very close (tracing can introduce small differences)
        assert torch.allclose(original_output, traced_output, atol=1e-4)


class TestExportNumerical:
    """Numerical tests for export pipeline."""

    def test_tracing_preserves_numerics(self):
        """Test that tracing preserves numerical accuracy."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        export_config = ANEExportConfig(sequence_length=32)
        exporter = CoreMLExporter(model, export_config)

        traced = exporter.trace()

        # Test multiple inputs
        for _ in range(3):
            input_ids = torch.randint(0, model_config.vocab_size, (1, 32))

            with torch.no_grad():
                original, _ = model(input_ids, use_cache=False)
                traced_out = traced(input_ids)

            max_diff = (original - traced_out).abs().max().item()
            assert max_diff < 1e-4, f"Max diff {max_diff} exceeds threshold"

    def test_tracing_deterministic(self):
        """Test that traced model is deterministic."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        export_config = ANEExportConfig(sequence_length=32)
        exporter = CoreMLExporter(model, export_config)

        traced = exporter.trace()

        input_ids = torch.randint(0, model_config.vocab_size, (1, 32))

        with torch.no_grad():
            out1 = traced(input_ids)
            out2 = traced(input_ids)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_sequence_lengths(self):
        """Test tracing with different sequence lengths."""
        model_config = ANEDeepSeekConfig.tiny()
        model = ANEDeepSeekModel(model_config)
        model.eval()

        for seq_len in [16, 32, 64]:
            export_config = ANEExportConfig(sequence_length=seq_len)
            exporter = CoreMLExporter(model, export_config)

            traced = exporter.trace()

            input_ids = torch.randint(0, model_config.vocab_size, (1, seq_len))

            with torch.no_grad():
                output = traced(input_ids)

            assert output.shape == (1, seq_len, model_config.vocab_size)
