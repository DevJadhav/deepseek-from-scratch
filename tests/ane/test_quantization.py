"""
Tests for ANE Quantization Utilities

Tests for INT8/INT4 quantization and dequantization functions.
"""

import pytest
import torch

from deepseek.mlx.ane.utils.quantization import (
    QuantizationConfig,
    QuantizationType,
    QuantizedTensor,
    compute_compression_ratio,
    compute_scale_per_block,
    compute_scale_per_channel,
    dequantize_int4_block,
    dequantize_int8_per_channel,
    get_quantized_size_bytes,
    quantize_int4_block,
    quantize_int8_per_channel,
)


class TestComputeScalePerChannel:
    """Tests for per-channel scale computation."""

    def test_symmetric_scale(self):
        """Test symmetric scale computation."""
        x = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        scale, zero_point = compute_scale_per_channel(x, axis=0)

        assert scale.shape == (2,)
        assert zero_point is None  # Symmetric quantization
        assert scale[0] == 3.0 / 127  # max of row 0 is 3.0
        assert scale[1] == 6.0 / 127  # max of row 1 is 6.0

    def test_asymmetric_scale(self):
        """Test asymmetric scale computation."""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scale, zero_point = compute_scale_per_channel(x, axis=0, symmetric=False)

        assert scale.shape == (2,)
        assert zero_point is not None
        assert zero_point.shape == (2,)


class TestComputeScalePerBlock:
    """Tests for per-block scale computation."""

    def test_block_scale(self):
        """Test per-block scale computation."""
        x = torch.randn(256, 256)
        scale = compute_scale_per_block(x, block_size=128)

        # 256/128 = 2 blocks per dimension
        assert scale.shape == (2, 2)
        assert (scale > 0).all()

    def test_non_divisible_size(self):
        """Test with non-divisible tensor size."""
        x = torch.randn(300, 300)
        scale = compute_scale_per_block(x, block_size=128)

        # 300/128 = 3 blocks (with padding)
        assert scale.shape == (3, 3)


class TestQuantizeInt8PerChannel:
    """Tests for INT8 per-channel quantization."""

    def test_basic_quantization(self):
        """Test basic INT8 quantization."""
        x = torch.randn(64, 128)
        quantized = quantize_int8_per_channel(x, axis=0)

        assert quantized.data.dtype == torch.int8
        assert quantized.scale.shape == (64,)
        assert quantized.quant_type == QuantizationType.INT8_PER_CHANNEL

    def test_quantize_dequantize_roundtrip(self):
        """Test that quantize-dequantize preserves values approximately."""
        x = torch.randn(32, 64)
        quantized = quantize_int8_per_channel(x, axis=0)
        dequantized = dequantize_int8_per_channel(quantized, axis=0)

        # Check shapes match
        assert dequantized.shape == x.shape

        # Check values are close (within quantization error)
        max_error = (x - dequantized.float()).abs().max()
        assert max_error < 0.1  # Reasonable quantization error

    def test_quantization_range(self):
        """Test that quantized values are in INT8 range."""
        x = torch.randn(64, 128) * 10  # Larger values
        quantized = quantize_int8_per_channel(x, axis=0)

        assert quantized.data.min() >= -127
        assert quantized.data.max() <= 127


class TestQuantizeInt4Block:
    """Tests for INT4 block-wise quantization."""

    def test_basic_int4_quantization(self):
        """Test basic INT4 quantization."""
        x = torch.randn(128, 128)
        quantized = quantize_int4_block(x, block_size=64)

        # INT4 is packed into int8 (2 values per byte)
        assert quantized.data.dtype == torch.int8
        assert quantized.quant_type == QuantizationType.INT4_PER_BLOCK

    def test_int4_roundtrip(self):
        """Test INT4 quantize-dequantize roundtrip."""
        x = torch.randn(128, 128)
        quantized = quantize_int4_block(x, block_size=64)
        dequantized = dequantize_int4_block(quantized, block_size=64)

        # Check shapes match
        assert dequantized.shape == x.shape

        # INT4 has larger quantization error
        max_error = (x - dequantized.float()).abs().max()
        assert max_error < 0.5  # INT4 has larger error than INT8

    def test_memory_reduction(self):
        """Test that INT4 provides memory reduction."""
        x = torch.randn(256, 256)
        quantized = quantize_int4_block(x, block_size=128)

        # Original: 256*256*4 bytes (FP32) = 262144 bytes
        # Packed INT4: ~256*256/2 bytes = ~32768 bytes (plus scales)
        original_bytes = x.numel() * 4
        packed_bytes = quantized.data.numel()

        assert packed_bytes < original_bytes / 6  # At least 6x compression


class TestQuantizedTensor:
    """Tests for QuantizedTensor class."""

    def test_dataclass_access(self):
        """Test dataclass field access."""
        data = torch.zeros(10, dtype=torch.int8)
        scale = torch.ones(10)
        qt = QuantizedTensor(
            data=data,
            scale=scale,
            zero_point=None,
            original_shape=(10,),
            quant_type=QuantizationType.INT8_PER_CHANNEL,
        )

        assert qt.data is data
        assert qt.scale is scale
        assert qt.zero_point is None
        assert qt.quant_type == QuantizationType.INT8_PER_CHANNEL
        assert qt.is_symmetric is True


class TestQuantizationConfig:
    """Tests for QuantizationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QuantizationConfig()

        assert config.quant_type == QuantizationType.INT8_PER_CHANNEL
        assert config.block_size == 128
        assert config.symmetric is True

    def test_custom_block_size(self):
        """Test custom block size configuration."""
        config = QuantizationConfig(block_size=64)
        assert config.block_size == 64


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_quantized_size_bytes_int8(self):
        """Test size calculation for INT8."""
        size = get_quantized_size_bytes(
            (1024, 1024),
            QuantizationType.INT8_PER_CHANNEL
        )

        # Data: 1024*1024 bytes, Scale: 1024*4 bytes
        expected = 1024 * 1024 + 1024 * 4
        assert size == expected

    def test_get_quantized_size_bytes_int4(self):
        """Test size calculation for INT4."""
        size = get_quantized_size_bytes(
            (1024, 1024),
            QuantizationType.INT4_PER_BLOCK,
            block_size=128
        )

        # Data: 1024*1024/2 bytes, Scale: 8*8*4 bytes
        expected_data = (1024 * 1024 + 1) // 2
        expected_scale = 8 * 8 * 4  # 8 blocks per dimension
        assert size == expected_data + expected_scale

    def test_compression_ratio_int8(self):
        """Test compression ratio for INT8."""
        ratio = compute_compression_ratio(torch.float32, QuantizationType.INT8_PER_CHANNEL)
        # FP32 (32 bits) -> INT8 (8.5 bits approximate with scale overhead)
        assert 3.5 < ratio < 4.0

    def test_compression_ratio_int4(self):
        """Test compression ratio for INT4."""
        ratio = compute_compression_ratio(torch.float32, QuantizationType.INT4_PER_BLOCK)
        # FP32 (32 bits) -> INT4 (4.5 bits approximate with scale overhead)
        assert 6.5 < ratio < 8.0

    def test_compression_ratio_none(self):
        """Test compression ratio for no quantization."""
        ratio = compute_compression_ratio(torch.float32, QuantizationType.NONE)
        assert ratio == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
