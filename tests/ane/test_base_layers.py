"""
Tests for ANE Base Layers

Tests for ANERMSNorm, ANELinear, and ANEEmbedding layers.
"""

import pytest
import torch

from deepseek.mlx.ane.layers.base import ANEEmbedding, ANELinear, ANERMSNorm
from deepseek.mlx.ane.utils.quantization import QuantizationType


class TestANERMSNorm:
    """Tests for ANERMSNorm layer."""

    def test_forward_shape(self):
        """Test that forward pass preserves shape."""
        norm = ANERMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        y = norm(x)

        assert y.shape == x.shape

    def test_normalization_property(self):
        """Test that output has unit RMS (approximately)."""
        norm = ANERMSNorm(dim=64, elementwise_affine=False)
        x = torch.randn(2, 16, 64) * 5  # Scale up input
        y = norm(x)

        # RMS of output should be close to 1
        rms = (y.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_fp16_conversion(self):
        """Test FP16 conversion for ANE efficiency."""
        norm = ANERMSNorm(dim=64, use_fp16=True)
        x = torch.randn(2, 16, 64)
        y = norm(x)

        # Output should be same dtype as input
        assert y.dtype == x.dtype

    def test_weight_initialization(self):
        """Test weight is initialized to ones."""
        norm = ANERMSNorm(dim=64)
        # Weight is padded to multiple of 16
        assert norm.weight.shape[0] >= 64
        # Original values should be ones
        assert torch.allclose(norm.weight[:64], torch.ones(64))

    def test_without_elementwise_affine(self):
        """Test norm without learned scale."""
        norm = ANERMSNorm(dim=64, elementwise_affine=False)
        assert norm.weight is None

        x = torch.randn(2, 16, 64)
        y = norm(x)
        assert y.shape == x.shape


class TestANELinear:
    """Tests for ANELinear layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shape."""
        linear = ANELinear(in_features=64, out_features=128)
        x = torch.randn(2, 16, 64)
        y = linear(x)

        assert y.shape == (2, 16, 128)

    def test_forward_no_bias(self):
        """Test forward without bias."""
        linear = ANELinear(in_features=64, out_features=128, bias=False)
        assert linear.bias is None

        x = torch.randn(2, 16, 64)
        y = linear(x)
        assert y.shape == (2, 16, 128)

    def test_quantize_weights_int8(self):
        """Test INT8 weight quantization."""
        linear = ANELinear(
            in_features=64,
            out_features=128,
            quant_type=QuantizationType.INT8_PER_CHANNEL,
        )

        # Initially not quantized
        assert not linear._quantized

        # Quantize
        linear.quantize_weights()

        # Now quantized
        assert linear._quantized
        assert linear.weight_quantized is not None
        assert linear.weight_quantized.dtype == torch.int8
        assert linear.weight_scale is not None

    def test_quantize_weights_int4(self):
        """Test INT4 weight quantization."""
        linear = ANELinear(
            in_features=128,
            out_features=256,
            quant_type=QuantizationType.INT4_PER_BLOCK,
            tile_size=64,
        )

        linear.quantize_weights()

        assert linear._quantized
        assert linear.weight_quantized is not None
        # INT4 is packed into int8
        assert linear.weight_quantized.dtype == torch.int8

    def test_forward_after_quantization(self):
        """Test forward pass works after quantization."""
        linear = ANELinear(in_features=64, out_features=128)
        x = torch.randn(2, 16, 64)

        # Get output before quantization
        y_before = linear(x).clone()

        # Quantize
        linear.quantize_weights()

        # Get output after quantization
        y_after = linear(x)

        # Shapes should match
        assert y_after.shape == y_before.shape

        # Values should be close (within quantization error)
        # Note: INT8 quantization introduces small errors
        max_error = (y_before - y_after).abs().max()
        assert max_error < 1.0  # Allow some quantization error

    def test_tiled_matmul(self):
        """Test tiled matmul for large dimensions."""
        linear = ANELinear(
            in_features=256,
            out_features=512,
            tile_size=64,  # Small tiles to trigger tiling
        )
        x = torch.randn(2, 16, 256)
        y = linear(x)

        assert y.shape == (2, 16, 512)

    def test_fp16_mode(self):
        """Test FP16 computation mode."""
        linear = ANELinear(in_features=64, out_features=128, use_fp16=True)
        x = torch.randn(2, 16, 64)
        y = linear(x)

        # Output should be same dtype as input
        assert y.dtype == x.dtype


class TestANEEmbedding:
    """Tests for ANEEmbedding layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shape."""
        embedding = ANEEmbedding(num_embeddings=1000, embedding_dim=64)
        input_ids = torch.randint(0, 1000, (2, 16))
        y = embedding(input_ids)

        assert y.shape == (2, 16, 64)

    def test_padding_idx(self):
        """Test padding index behavior."""
        embedding = ANEEmbedding(
            num_embeddings=1000,
            embedding_dim=64,
            padding_idx=0,
        )

        # Padding idx embedding should be zeros
        assert torch.allclose(
            embedding.weight[0],
            torch.zeros(64),
        )

        # Other embeddings should not be zeros
        assert not torch.allclose(
            embedding.weight[1],
            torch.zeros(64),
        )

    def test_fp16_output(self):
        """Test FP16 output mode."""
        embedding = ANEEmbedding(
            num_embeddings=1000,
            embedding_dim=64,
            use_fp16=True,
        )
        input_ids = torch.randint(0, 1000, (2, 16))
        y = embedding(input_ids)

        # Output should be FP16 when use_fp16=True
        assert y.dtype == torch.float16

    def test_fp32_output(self):
        """Test FP32 output mode."""
        embedding = ANEEmbedding(
            num_embeddings=1000,
            embedding_dim=64,
            use_fp16=False,
        )
        input_ids = torch.randint(0, 1000, (2, 16))
        y = embedding(input_ids)

        # Output should be FP32 when use_fp16=False
        assert y.dtype == torch.float32

    def test_different_input_shapes(self):
        """Test with different input shapes."""
        embedding = ANEEmbedding(num_embeddings=1000, embedding_dim=64)

        # 1D input
        ids_1d = torch.randint(0, 1000, (16,))
        y_1d = embedding(ids_1d)
        assert y_1d.shape == (16, 64)

        # 2D input
        ids_2d = torch.randint(0, 1000, (2, 16))
        y_2d = embedding(ids_2d)
        assert y_2d.shape == (2, 16, 64)

        # 3D input
        ids_3d = torch.randint(0, 1000, (2, 4, 16))
        y_3d = embedding(ids_3d)
        assert y_3d.shape == (2, 4, 16, 64)


class TestLayerIntegration:
    """Integration tests for ANE layers working together."""

    def test_embedding_to_linear(self):
        """Test embedding output flows to linear."""
        embedding = ANEEmbedding(num_embeddings=1000, embedding_dim=64)
        linear = ANELinear(in_features=64, out_features=128)

        input_ids = torch.randint(0, 1000, (2, 16))
        x = embedding(input_ids)
        y = linear(x)

        assert y.shape == (2, 16, 128)

    def test_linear_to_norm(self):
        """Test linear output flows to norm."""
        linear = ANELinear(in_features=64, out_features=128)
        norm = ANERMSNorm(dim=128)

        x = torch.randn(2, 16, 64)
        y = linear(x)
        z = norm(y)

        assert z.shape == (2, 16, 128)

    def test_full_pipeline(self):
        """Test embedding -> linear -> norm pipeline."""
        embedding = ANEEmbedding(num_embeddings=1000, embedding_dim=64)
        linear = ANELinear(in_features=64, out_features=128)
        norm = ANERMSNorm(dim=128)

        input_ids = torch.randint(0, 1000, (2, 16))
        x = embedding(input_ids)
        y = linear(x)
        z = norm(y)

        assert z.shape == (2, 16, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
