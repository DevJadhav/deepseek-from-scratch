"""Tests for INT4/INT8 inference quantization in MLX."""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

if MLX_AVAILABLE:
    from src.deepseek.mlx.quantization import (
        QuantizedLinearInt4,
        QuantizedLinearInt8,
        quantize_model_int4,
        quantize_model_int8,
    )


pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


class TestQuantizedLinearInt8:
    """Tests for INT8 quantized linear layer."""

    def test_from_float_basic(self):
        """Test basic INT8 quantization from float."""
        # Create a simple linear layer
        linear = nn.Linear(64, 32)
        
        # Quantize
        quantized = QuantizedLinearInt8.from_float(linear)
        
        # Check shapes
        assert quantized.weight_int8.shape == (32, 64)
        assert quantized.scale.shape == (32,)

    def test_from_float_with_bias(self):
        """Test INT8 quantization preserves bias."""
        linear = nn.Linear(64, 32)
        
        quantized = QuantizedLinearInt8.from_float(linear)
        
        # Bias should be preserved
        assert quantized.bias is not None
        assert quantized.bias.shape == (32,)

    def test_forward_shape(self):
        """Test forward pass produces correct shape."""
        linear = nn.Linear(64, 32)
        quantized = QuantizedLinearInt8.from_float(linear)
        
        x = mx.random.normal((2, 10, 64))
        out = quantized(x)
        
        assert out.shape == (2, 10, 32)

    def test_quantization_range(self):
        """Test INT8 values are in valid range."""
        linear = nn.Linear(128, 64)
        # Set large weights to test clamping
        linear.weight = mx.random.normal((64, 128)) * 10
        
        quantized = QuantizedLinearInt8.from_float(linear)
        
        # All values should be in int8 range
        weight_flat = quantized.weight_int8.reshape(-1)
        assert mx.all(weight_flat >= -128)
        assert mx.all(weight_flat <= 127)

    def test_dequantize_reconstruction(self):
        """Test dequantization approximately reconstructs original."""
        linear = nn.Linear(64, 32)
        
        original_weight = linear.weight.astype(mx.float32)
        quantized = QuantizedLinearInt8.from_float(linear)
        
        # Dequantize
        dequant_weight = quantized._dequantize()
        
        # Relative error should be small
        error = mx.abs(original_weight - dequant_weight.astype(mx.float32))
        rel_error = mx.mean(error) / mx.mean(mx.abs(original_weight))
        
        assert float(rel_error) < 0.1  # Less than 10% relative error


class TestQuantizedLinearInt4:
    """Tests for INT4 quantized linear layer."""

    def test_from_float_basic(self):
        """Test basic INT4 quantization from float."""
        linear = nn.Linear(256, 64)
        
        quantized = QuantizedLinearInt4.from_float(linear, group_size=128)
        
        # Check packed weight shape (2 values per byte)
        assert quantized.weight_packed.shape[0] == 64
        assert quantized.weight_packed.shape[1] == 128  # 256/2

    def test_group_size_affects_scales(self):
        """Test group size affects number of scale groups."""
        linear = nn.Linear(256, 64)
        
        q128 = QuantizedLinearInt4.from_float(linear, group_size=128)
        q64 = QuantizedLinearInt4.from_float(linear, group_size=64)
        
        # With group_size=128, 256 features = 2 groups
        assert q128.scales.shape == (64, 2)
        # With group_size=64, 256 features = 4 groups
        assert q64.scales.shape == (64, 4)

    def test_forward_shape(self):
        """Test forward pass produces correct shape."""
        linear = nn.Linear(256, 64)
        quantized = QuantizedLinearInt4.from_float(linear, group_size=128)
        
        x = mx.random.normal((2, 10, 256))
        out = quantized(x)
        
        assert out.shape == (2, 10, 64)

    def test_nibble_packing(self):
        """Test nibbles are packed correctly (2 per byte)."""
        linear = nn.Linear(128, 32)
        quantized = QuantizedLinearInt4.from_float(linear, group_size=128)
        
        # Packed weight should be half the size
        assert quantized.weight_packed.shape[1] == 64  # 128/2

    def test_scales_zeros_range(self):
        """Test scales and zeros are computed correctly."""
        linear = nn.Linear(128, 32)
        quantized = QuantizedLinearInt4.from_float(linear, group_size=128)
        
        # Scales should be positive
        assert mx.all(quantized.scales > 0)
        # Zeros should be in reasonable range (0-15 for asymmetric)
        assert mx.all(quantized.zeros >= 0)
        assert mx.all(quantized.zeros <= 15)


class TestModelQuantization:
    """Tests for full model quantization."""

    def test_quantize_model_int8(self):
        """Test INT8 quantization of a simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 16)
            
            def __call__(self, x):
                x = mx.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel()
        quantized_model = quantize_model_int8(model)
        
        # Check layers were replaced
        assert isinstance(quantized_model.fc1, QuantizedLinearInt8)
        assert isinstance(quantized_model.fc2, QuantizedLinearInt8)

    def test_quantize_model_int4(self):
        """Test INT4 quantization of a simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 128)
                self.fc2 = nn.Linear(128, 64)
            
            def __call__(self, x):
                x = mx.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel()
        quantized_model = quantize_model_int4(model, group_size=64)
        
        # Check layers were replaced
        assert isinstance(quantized_model.fc1, QuantizedLinearInt4)
        assert isinstance(quantized_model.fc2, QuantizedLinearInt4)


class TestQuantizationNumerics:
    """Tests for quantization numerical accuracy."""

    def test_int8_output_similarity(self):
        """Test INT8 output is similar to float output."""
        linear = nn.Linear(64, 32)
        quantized = QuantizedLinearInt8.from_float(linear)
        
        x = mx.random.normal((4, 64))
        
        # Float output
        float_out = linear(x)
        # Quantized output
        quant_out = quantized(x)
        
        # Outputs should be similar
        diff = mx.abs(float_out - quant_out)
        rel_diff = mx.mean(diff) / (mx.mean(mx.abs(float_out)) + 1e-6)
        
        assert float(rel_diff) < 0.2  # Less than 20% relative error

    def test_int4_output_similarity(self):
        """Test INT4 output is similar to float output."""
        linear = nn.Linear(256, 64)
        quantized = QuantizedLinearInt4.from_float(linear, group_size=128)
        
        x = mx.random.normal((4, 256))
        
        # Float output
        float_out = linear(x)
        # Quantized output
        quant_out = quantized(x)
        
        # INT4 has more error than INT8
        diff = mx.abs(float_out - quant_out)
        rel_diff = mx.mean(diff) / (mx.mean(mx.abs(float_out)) + 1e-6)
        
        assert float(rel_diff) < 0.5  # Less than 50% relative error (INT4 is lossy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
