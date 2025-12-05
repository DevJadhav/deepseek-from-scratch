"""
Tests for ANE Quantization Module

Tests all ANE quantization components:
- Weight Quantization (INT8, INT4) via weight_quant.py
- Activation Quantization (FP16, INT8) via activation_quant.py
- Mixed Precision Management via mixed_precision.py
- Dynamic Quantization with Calibration via dynamic_quant.py
"""

import pytest
import torch
import torch.nn as nn

from deepseek.mlx.ane.quantization.weight_quant import (
    WeightQuantConfig,
    WeightQuantType,
    QuantizedWeight,
    quantize_weight_int8,
    quantize_weight_int4,
    dequantize_weight_int8,
    dequantize_weight_int4,
)
from deepseek.mlx.ane.quantization.activation_quant import (
    ActivationQuantConfig,
    ActivationQuantType,
    ActivationStats,
    QuantizedActivation,
    ActivationObserver,
    ANEActivationQuantizer,
    quantize_activation_fp16,
    quantize_activation_int8,
    dequantize_activation,
)
from deepseek.mlx.ane.quantization.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionManager,
    LayerPrecision,
    LayerType,
)
from deepseek.mlx.ane.quantization.dynamic_quant import (
    CalibrationConfig,
    CalibrationMethod,
    DynamicQuantizer,
    HistogramObserver,
    calibrate_model,
)


# ============================================================================
# Phase 6.1: Weight Quantization Tests
# ============================================================================

class TestPhase6WeightQuantConfig:
    """Tests for Phase 6 weight quantization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WeightQuantConfig()
        assert config.quant_type == WeightQuantType.INT8_PER_CHANNEL
        assert config.symmetric is True
        assert config.block_size == 128

    def test_int4_config(self):
        """Test INT4 configuration."""
        config = WeightQuantConfig(
            quant_type=WeightQuantType.INT4_PER_BLOCK,
            block_size=64,
        )
        assert config.quant_type == WeightQuantType.INT4_PER_BLOCK
        assert config.block_size == 64


class TestPhase6QuantizedWeight:
    """Tests for Phase 6 quantized weight data class."""

    def test_int8_quantized_weight(self):
        """Test INT8 quantized weight creation."""
        weight = torch.randn(64, 32)
        config = WeightQuantConfig(quant_type=WeightQuantType.INT8_PER_CHANNEL)
        qw = quantize_weight_int8(weight, config=config)

        assert qw.data.dtype == torch.int8
        assert qw.scale.shape == (64,)
        assert qw.is_symmetric is True

    def test_dequantize_int8(self):
        """Test INT8 dequantization."""
        weight = torch.randn(64, 32)
        qw = quantize_weight_int8(weight)
        dequant = dequantize_weight_int8(qw)

        # Check reconstruction is close
        error = (weight - dequant.float()).abs().mean()
        assert error < 0.1  # Allow some quantization error

    def test_memory_bytes(self):
        """Test memory bytes calculation."""
        weight = torch.randn(256, 128)
        qw = quantize_weight_int8(weight)

        mem = qw.memory_bytes()
        assert mem > 0
        # INT8 data + FP32 scales
        expected_approx = 256 * 128 + 256 * 4
        assert abs(mem - expected_approx) < 100  # Allow some margin

    def test_compression_ratio(self):
        """Test compression ratio property."""
        weight = torch.randn(256, 128)
        qw = quantize_weight_int8(weight)

        ratio = qw.compression_ratio
        # FP16 to INT8 should compress ~2x (minus scale overhead)
        assert ratio > 1.5


class TestPhase6WeightQuantization:
    """Tests for Phase 6 weight quantization functions."""

    def test_int8_per_channel_quantization(self):
        """Test INT8 per-channel weight quantization."""
        weight = torch.randn(64, 32)
        qw = quantize_weight_int8(weight, axis=0)

        assert qw.data.dtype == torch.int8
        assert qw.scale.shape == (64,)  # Per-channel scale

    def test_int8_quantize_dequantize_accuracy(self):
        """Test that INT8 quantize/dequantize preserves values reasonably."""
        weight = torch.randn(64, 32)
        qw = quantize_weight_int8(weight)
        dequant = dequantize_weight_int8(qw)

        # Check reconstruction error
        error = (weight - dequant.float()).abs().mean()
        assert error < 0.1  # Allow some quantization error

    def test_int4_per_block_quantization(self):
        """Test INT4 per-block weight quantization."""
        weight = torch.randn(128, 256)
        qw = quantize_weight_int4(weight, block_size=64)

        assert qw.config.quant_type == WeightQuantType.INT4_PER_BLOCK
        # INT4 is packed (2 values per byte)
        assert qw.data.dtype == torch.int8

    def test_int4_quantize_dequantize(self):
        """Test INT4 quantize/dequantize cycle."""
        weight = torch.randn(128, 128)
        qw = quantize_weight_int4(weight, block_size=64)
        dequant = dequantize_weight_int4(qw)

        # INT4 has larger error than INT8
        error = (weight - dequant.float()).abs().mean()
        assert error < 0.3


class TestPhase6QuantizeLinearLayer:
    """Tests for Phase 6 linear layer quantization."""

    def test_quantize_linear_weights(self):
        """Test quantizing linear layer weights."""
        layer = nn.Linear(64, 32)

        qw = quantize_weight_int8(layer.weight.data)

        assert qw is not None
        assert qw.data.shape[0] == 32  # output features
        assert qw.data.shape[1] == 64  # input features


# ============================================================================
# Phase 6.2: Activation Quantization Tests
# ============================================================================

class TestPhase6ActivationQuantConfig:
    """Tests for Phase 6 activation quantization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ActivationQuantConfig()
        assert config.quant_type == ActivationQuantType.FP16

    def test_int8_config(self):
        """Test INT8 configuration."""
        config = ActivationQuantConfig(
            quant_type=ActivationQuantType.INT8_PER_TENSOR,
            symmetric=True,
        )
        assert config.quant_type == ActivationQuantType.INT8_PER_TENSOR
        assert config.symmetric is True


class TestPhase6ActivationStats:
    """Tests for Phase 6 activation statistics."""

    def test_stats_update(self):
        """Test statistics update."""
        stats = ActivationStats()

        stats.update(torch.tensor([1.0, 2.0, 3.0]))
        stats.update(torch.tensor([-1.0, 0.0, 5.0]))

        assert stats.min_val == -1.0
        assert stats.max_val == 5.0
        assert stats.num_batches == 2


class TestPhase6ActivationObserver:
    """Tests for Phase 6 activation observer."""

    def test_observer_forward(self):
        """Test observer forward pass."""
        config = ActivationQuantConfig(quant_type=ActivationQuantType.INT8_PER_TENSOR)
        observer = ActivationObserver(config)
        observer.train()

        x = torch.randn(32, 64)
        y = observer(x)

        # Should pass through unchanged
        torch.testing.assert_close(x, y)
        assert observer.num_batches > 0

    def test_observer_qparams(self):
        """Test observer qparams computation."""
        config = ActivationQuantConfig(symmetric=True)
        observer = ActivationObserver(config)
        observer.train()

        for _ in range(10):
            _ = observer(torch.randn(32, 64))

        scale, zp = observer.calculate_qparams()
        assert scale > 0
        assert zp is None  # Symmetric

    def test_observer_reset(self):
        """Test observer reset."""
        config = ActivationQuantConfig()
        observer = ActivationObserver(config)
        observer.train()

        _ = observer(torch.randn(32, 64))
        observer.reset()

        assert observer.num_batches == 0


class TestPhase6ActivationQuantizer:
    """Tests for Phase 6 ANE activation quantizer."""

    def test_fp16_quantization(self):
        """Test FP16 activation quantization."""
        quantizer = ANEActivationQuantizer()

        x = torch.randn(2, 64, dtype=torch.float32)
        qx = quantizer.quantize(x)

        assert qx.quant_type == ActivationQuantType.FP16

    def test_int8_quantization(self):
        """Test INT8 activation quantization."""
        config = ActivationQuantConfig(
            quant_type=ActivationQuantType.INT8_PER_TENSOR,
            symmetric=True,
        )
        quantizer = ANEActivationQuantizer(config=config)

        x = torch.randn(2, 64)
        qx = quantizer.quantize(x)

        assert qx.data.dtype == torch.int8

    def test_quantize_dequantize(self):
        """Test quantize and dequantize cycle."""
        config = ActivationQuantConfig(quant_type=ActivationQuantType.INT8_PER_TENSOR)
        quantizer = ANEActivationQuantizer(config=config)

        x = torch.randn(2, 64)
        qx = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qx)

        # Check reconstruction is close
        error = (x - x_reconstructed.float()).abs().mean()
        assert error < 0.5  # Allow quantization error

    def test_critical_vs_non_critical(self):
        """Test critical vs non-critical path quantization."""
        quantizer = ANEActivationQuantizer()

        x = torch.randn(2, 64)

        # Critical should use FP16
        qx_critical = quantizer.quantize_critical(x)
        assert qx_critical.quant_type == ActivationQuantType.FP16

        # Non-critical should use INT8
        qx_non_critical = quantizer.quantize_non_critical(x)
        assert qx_non_critical.quant_type == ActivationQuantType.INT8_PER_TENSOR


class TestPhase6QuantizeActivationFunctions:
    """Tests for Phase 6 activation quantization functions."""

    def test_quantize_fp16(self):
        """Test quantize_activation_fp16 function."""
        x = torch.randn(4, 128, dtype=torch.float32)
        qx = quantize_activation_fp16(x)

        assert qx.data.dtype == torch.float16
        assert qx.quant_type == ActivationQuantType.FP16

    def test_quantize_int8(self):
        """Test quantize_activation_int8 function."""
        x = torch.randn(4, 128)
        qx = quantize_activation_int8(x, symmetric=True)

        assert qx.data.dtype == torch.int8

    def test_dequantize(self):
        """Test dequantize_activation function."""
        x = torch.randn(4, 128)
        qx = quantize_activation_int8(x)
        dequant = dequantize_activation(qx)

        assert dequant.shape == x.shape


# ============================================================================
# Phase 6.1: Mixed Precision Tests
# ============================================================================

class TestPhase6LayerType:
    """Tests for Phase 6 layer type enum."""

    def test_layer_types(self):
        """Test layer type enum values."""
        assert LayerType.EMBEDDING.value == "embedding"
        assert LayerType.ATTENTION_QKV.value == "attention_qkv"
        assert LayerType.FFN_GATE.value == "ffn_gate"
        assert LayerType.EXPERT.value == "expert"


class TestPhase6LayerPrecision:
    """Tests for Phase 6 layer precision dataclass."""

    def test_create_precision(self):
        """Test creating layer precision."""
        precision = LayerPrecision(
            weight_quant=WeightQuantConfig(quant_type=WeightQuantType.INT8_PER_CHANNEL),
            activation_quant=ActivationQuantConfig(quant_type=ActivationQuantType.FP16),
            is_critical=True,
            layer_type=LayerType.ATTENTION_QKV,
        )

        assert precision.is_critical is True
        assert precision.layer_type == LayerType.ATTENTION_QKV


class TestPhase6MixedPrecisionConfig:
    """Tests for Phase 6 mixed precision configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = MixedPrecisionConfig()
        assert config.embedding_precision is not None
        assert config.attention_precision is not None
        assert config.ffn_precision is not None

    def test_accuracy_focused_config(self):
        """Test accuracy-focused configuration."""
        config = MixedPrecisionConfig.accuracy_focused()
        assert config.keep_critical_fp16 is True

    def test_efficiency_focused_config(self):
        """Test efficiency-focused configuration."""
        config = MixedPrecisionConfig.efficiency_focused()
        assert config.default_weight_quant == WeightQuantType.INT4_PER_BLOCK


class TestPhase6MixedPrecisionManager:
    """Tests for Phase 6 mixed precision manager."""

    def test_create_manager(self):
        """Test creating mixed precision manager."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        assert manager is not None

    def test_classify_layer(self):
        """Test layer classification."""
        manager = MixedPrecisionManager()

        # Test various layer names
        assert manager.classify_layer("model.embed_tokens") == LayerType.EMBEDDING
        assert manager.classify_layer("model.layers.0.self_attn.q_proj") == LayerType.ATTENTION_QKV
        assert manager.classify_layer("model.layers.0.mlp.gate_proj") == LayerType.FFN_GATE
        assert manager.classify_layer("model.layers.0.norm") == LayerType.LAYERNORM
        assert manager.classify_layer("lm_head") == LayerType.LM_HEAD

    def test_get_layer_precision(self):
        """Test getting precision for specific layer."""
        manager = MixedPrecisionManager()

        precision = manager.get_layer_precision("model.layers.0.self_attn.q_proj")
        assert precision.layer_type == LayerType.ATTENTION_QKV

    def test_should_quantize_weights(self):
        """Test should_quantize_weights method."""
        manager = MixedPrecisionManager()

        # Embeddings should not be quantized (FP16)
        assert manager.should_quantize_weights("embed_tokens") is False

        # Attention should be quantized (INT8)
        assert manager.should_quantize_weights("self_attn.q_proj") is True

    def test_model_precision_summary(self):
        """Test getting model precision summary."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.attn_q = nn.Linear(64, 64)
                self.ffn_gate = nn.Linear(64, 256)

        model = SimpleModel()
        manager = MixedPrecisionManager()

        summary = manager.get_model_precision_summary(model)
        assert "by_type" in summary
        assert "layers" in summary
        assert len(summary["layers"]) > 0

    def test_estimate_memory_savings(self):
        """Test memory savings estimation."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1024, 1024)

        model = SimpleModel()
        manager = MixedPrecisionManager()

        estimates = manager.estimate_memory_savings(model)
        assert 'baseline_bytes' in estimates
        assert 'quantized_bytes' in estimates
        assert 'compression_ratio' in estimates
        assert estimates['compression_ratio'] > 1.0


# ============================================================================
# Phase 6.2: Dynamic Quantization Tests
# ============================================================================

class TestPhase6CalibrationConfig:
    """Tests for Phase 6 calibration configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = CalibrationConfig()
        assert config.method == CalibrationMethod.MINMAX
        assert config.num_bits == 8

    def test_percentile_config(self):
        """Test percentile configuration."""
        config = CalibrationConfig(
            method=CalibrationMethod.PERCENTILE,
            percentile=99.9,
        )
        assert config.method == CalibrationMethod.PERCENTILE
        assert config.percentile == 99.9


class TestPhase6HistogramObserver:
    """Tests for Phase 6 histogram observer."""

    def test_observe_tensor(self):
        """Test observing a tensor."""
        config = CalibrationConfig(num_bins=256)
        observer = HistogramObserver(config)
        observer.train()

        x = torch.randn(100)
        _ = observer(x)

        assert observer.num_samples > 0

    def test_compute_qparams_minmax(self):
        """Test computing qparams with minmax method."""
        config = CalibrationConfig(method=CalibrationMethod.MINMAX)
        observer = HistogramObserver(config)
        observer.train()

        for _ in range(10):
            _ = observer(torch.randn(100))

        scale, zp = observer.compute_qparams()
        assert scale > 0

    def test_observer_reset(self):
        """Test resetting observer."""
        config = CalibrationConfig()
        observer = HistogramObserver(config)
        observer.train()

        _ = observer(torch.randn(100))
        observer.reset()

        assert observer.num_samples == 0


class TestPhase6DynamicQuantizer:
    """Tests for Phase 6 dynamic quantizer."""

    def test_create_quantizer(self):
        """Test creating dynamic quantizer."""
        config = CalibrationConfig()
        quantizer = DynamicQuantizer(config)

        assert quantizer is not None

    def test_calibration_mode(self):
        """Test calibration mode."""
        quantizer = DynamicQuantizer()
        quantizer.register_layer("test_layer")
        quantizer.enable_calibration()

        x = torch.randn(32, 64)
        y = quantizer(x, name="test_layer")

        # In calibration mode, should pass through
        torch.testing.assert_close(x, y)

    def test_inference_mode(self):
        """Test inference mode after calibration."""
        quantizer = DynamicQuantizer()
        quantizer.register_layer("test_layer")

        # Calibrate
        quantizer.enable_calibration()
        for _ in range(10):
            _ = quantizer(torch.randn(32, 64), name="test_layer")
        quantizer.disable_calibration()

        # Enable inference
        quantizer.enable_inference()

        x = torch.randn(32, 64)
        y = quantizer(x, name="test_layer")

        # Should be quantized/dequantized
        assert y.shape == x.shape

    def test_get_qparams(self):
        """Test getting quantization parameters."""
        quantizer = DynamicQuantizer()
        quantizer.register_layer("layer1")
        quantizer.register_layer("layer2")

        quantizer.enable_calibration()
        for _ in range(5):
            _ = quantizer(torch.randn(16, 32), name="layer1")
            _ = quantizer(torch.randn(16, 64), name="layer2")
        quantizer.disable_calibration()

        qparams = quantizer.get_qparams()

        assert "layer1" in qparams
        assert "layer2" in qparams

    def test_save_load_calibration(self, tmp_path):
        """Test saving and loading calibration."""
        quantizer = DynamicQuantizer()
        quantizer.register_layer("test")

        quantizer.enable_calibration()
        _ = quantizer(torch.randn(32, 64), name="test")
        quantizer.disable_calibration()

        # Save
        save_path = str(tmp_path / "calibration.pt")
        quantizer.save_calibration(save_path)

        # Load into new quantizer
        quantizer2 = DynamicQuantizer()
        quantizer2.load_calibration(save_path)

        assert "test" in quantizer2.get_qparams()


class TestPhase6CalibrateModel:
    """Tests for Phase 6 model calibration function."""

    def test_calibrate_simple_model(self):
        """Test calibrating a simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Create simple dataloader
        data = [torch.randn(8, 64) for _ in range(10)]

        config = CalibrationConfig(num_batches=5)
        quantizer = calibrate_model(model, data, config)

        assert quantizer is not None


# ============================================================================
# Phase 6 Integration Tests
# ============================================================================

class TestPhase6QuantizationIntegration:
    """Integration tests for Phase 6 quantization components."""

    def test_full_quantization_pipeline(self):
        """Test full quantization pipeline."""
        # Create model
        class TransformerBlock(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn_qkv = nn.Linear(dim, dim * 3)
                self.attn_out = nn.Linear(dim, dim)
                self.norm2 = nn.LayerNorm(dim)
                self.ffn1 = nn.Linear(dim, dim * 4)
                self.ffn2 = nn.Linear(dim * 4, dim)

            def forward(self, x):
                h = self.norm1(x)
                qkv = self.attn_qkv(h)  # noqa: F841
                h = self.attn_out(h)
                x = x + h
                h = self.norm2(x)
                h = torch.relu(self.ffn1(h))
                h = self.ffn2(h)
                return x + h

        model = TransformerBlock()

        # Apply mixed precision analysis
        manager = MixedPrecisionManager()
        summary = manager.get_model_precision_summary(model)
        assert len(summary["layers"]) > 0

        # Quantize weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                qw = quantize_weight_int8(module.weight.data)
                assert qw is not None

        # Test activation quantization
        quantizer = ANEActivationQuantizer()
        x = torch.randn(2, 8, 64)
        qx = quantizer.quantize(x)
        assert qx is not None

    def test_quantization_preserves_output_shape(self):
        """Test that quantization preserves output shapes."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        x = torch.randn(4, 64)
        original_output = model(x)

        # Quantize and dequantize weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                qw = quantize_weight_int8(module.weight.data)
                dequant = dequantize_weight_int8(qw)
                module.weight.data = dequant.float()

        # Shape should be preserved
        new_output = model(x)
        assert new_output.shape == original_output.shape

    def test_end_to_end_weight_activation_quantization(self):
        """Test combined weight and activation quantization."""
        # Create simple model
        model = nn.Linear(64, 32)

        # Quantize weights
        qw = quantize_weight_int8(model.weight.data)

        # Quantize activations
        quantizer = ANEActivationQuantizer(
            config=ActivationQuantConfig(quant_type=ActivationQuantType.INT8_PER_TENSOR)
        )

        # Run inference with quantized components
        x = torch.randn(4, 64)
        qx = quantizer.quantize(x)
        dequant_x = quantizer.dequantize(qx)

        # Verify output
        with torch.no_grad():
            dequant_weight = dequantize_weight_int8(qw)
            # Ensure both tensors are float32 for matmul
            output = torch.matmul(dequant_x.float(), dequant_weight.float().t())

        assert output.shape == (4, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
