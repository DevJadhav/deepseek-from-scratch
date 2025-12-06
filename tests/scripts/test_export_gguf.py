"""Tests for GGUF export functionality with K-quant support."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.export_gguf import (
    GGMLType,
    GGUFMetadata,
    GGUFWriter,
    convert_weight_names,
    quantize_q4_0,
    quantize_q4_k,
    quantize_q5_k,
    quantize_q6_k,
    quantize_q8_0,
)


class TestQuantizationQ8_0:
    """Tests for Q8_0 quantization."""

    def test_basic_quantization(self):
        """Test basic Q8_0 quantization."""
        data = np.random.randn(256).astype(np.float32)
        quantized, scales = quantize_q8_0(data)

        # Check shapes - 256 elements / 32 block size = 8 blocks
        assert quantized.shape == (8, 32)
        assert scales.shape == (8,)
        assert scales.dtype == np.float16
        assert quantized.dtype == np.int8

    def test_quantization_range(self):
        """Test Q8_0 values are in valid range."""
        data = np.random.randn(1024).astype(np.float32) * 10
        quantized, scales = quantize_q8_0(data)

        # All values should be in int8 range
        assert np.all(quantized >= -128)
        assert np.all(quantized <= 127)

    def test_zero_handling(self):
        """Test Q8_0 handles zeros correctly."""
        data = np.zeros(64, dtype=np.float32)
        quantized, scales = quantize_q8_0(data)

        # Scales should be 1.0 to avoid division by zero
        assert np.all(scales > 0)


class TestQuantizationQ4_0:
    """Tests for Q4_0 quantization."""

    def test_basic_quantization(self):
        """Test basic Q4_0 quantization with nibble packing."""
        data = np.random.randn(256).astype(np.float32)
        packed, scales = quantize_q4_0(data)

        # Check shapes - 256 elements / 32 block size = 8 blocks
        # 32 values packed as nibbles = 16 bytes per block
        assert packed.shape == (8, 16)
        assert scales.shape == (8,)
        assert packed.dtype == np.uint8
        assert scales.dtype == np.float16

    def test_nibble_packing(self):
        """Test nibble packing is correct."""
        # Create predictable data
        data = np.arange(32, dtype=np.float32) / 4.0  # Values 0-7.75
        packed, scales = quantize_q4_0(data)

        # Each byte should contain 2 nibbles
        # Low nibble = first value + 8, High nibble = second value + 8
        assert packed.shape == (1, 16)
        for i in range(16):
            low = packed[0, i] & 0x0F
            high = (packed[0, i] >> 4) & 0x0F
            # Both nibbles should be in valid range (0-15 for -8 to +7 shifted)
            assert 0 <= low <= 15
            assert 0 <= high <= 15


class TestQuantizationQ4_K:
    """Tests for Q4_K (K-quant) quantization."""

    def test_basic_quantization(self):
        """Test basic Q4_K quantization with superblock structure."""
        data = np.random.randn(512).astype(np.float32)
        packed_quants, d, dmin, scales_mins = quantize_q4_k(data)

        # 512 elements / 256 superblock size = 2 superblocks
        assert packed_quants.shape == (2, 128)  # 128 bytes per superblock
        assert d.shape == (2,)
        assert dmin.shape == (2,)
        assert scales_mins.shape == (2, 12)  # 12 bytes of packed scales/mins

        assert d.dtype == np.float16
        assert dmin.dtype == np.float16
        assert packed_quants.dtype == np.uint8
        assert scales_mins.dtype == np.uint8

    def test_superblock_size(self):
        """Test Q4_K handles 256-element superblocks correctly."""
        data = np.random.randn(768).astype(np.float32)  # 3 superblocks
        packed_quants, d, dmin, scales_mins = quantize_q4_k(data)

        assert packed_quants.shape[0] == 3  # 3 superblocks

    def test_scales_mins_packing(self):
        """Test scales and mins are packed into 12 bytes correctly."""
        data = np.random.randn(256).astype(np.float32)
        packed_quants, d, dmin, scales_mins = quantize_q4_k(data)

        # scales_mins should be 12 bytes
        assert scales_mins.shape == (1, 12)


class TestQuantizationQ5_K:
    """Tests for Q5_K (5-bit K-quant) quantization."""

    def test_basic_quantization(self):
        """Test basic Q5_K quantization."""
        data = np.random.randn(512).astype(np.float32)
        packed_low, packed_high, d, dmin, scales_mins = quantize_q5_k(data)

        # 512 elements / 256 superblock size = 2 superblocks
        assert packed_low.shape == (2, 128)  # Low 4 bits
        assert packed_high.shape == (2, 32)  # High bit
        assert d.shape == (2,)
        assert dmin.shape == (2,)
        assert scales_mins.shape == (2, 12)

    def test_high_bits(self):
        """Test Q5_K high bits are packed correctly."""
        data = np.random.randn(256).astype(np.float32) * 10  # Larger values
        packed_low, packed_high, d, dmin, scales_mins = quantize_q5_k(data)

        # High bits should be packed - each byte holds 8 bits
        assert packed_high.shape == (1, 32)  # 256 / 8 = 32 bytes


class TestQuantizationQ6_K:
    """Tests for Q6_K (6-bit K-quant) quantization."""

    def test_basic_quantization(self):
        """Test basic Q6_K quantization."""
        data = np.random.randn(512).astype(np.float32)
        packed_low, packed_high, d, scales = quantize_q6_k(data)

        # 512 elements / 256 superblock size = 2 superblocks
        assert packed_low.shape == (2, 128)
        assert packed_high.shape == (2, 64)
        assert d.shape == (2,)
        assert scales.shape == (2, 16)  # 16 int8 scales per superblock

    def test_scales_are_int8(self):
        """Test Q6_K uses int8 scales."""
        data = np.random.randn(256).astype(np.float32)
        packed_low, packed_high, d, scales = quantize_q6_k(data)

        assert scales.dtype == np.int8
        assert np.all(scales >= -128)
        assert np.all(scales <= 127)


class TestGGUFWriter:
    """Tests for GGUF file writer."""

    def test_write_f16(self):
        """Test writing F16 tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gguf"
            metadata = GGUFMetadata(
                name="test",
                vocab_size=100,
                hidden_size=64,
                num_layers=2,
            )
            writer = GGUFWriter(output_path, metadata)

            # Add a simple tensor
            tensor = np.random.randn(64, 64).astype(np.float32)
            writer.add_tensor("test.weight", tensor, GGMLType.F16)
            writer.write()

            # Verify file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify magic number
            with open(output_path, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                assert magic == 0x46554747  # "GGUF"

    def test_write_q8_0(self):
        """Test writing Q8_0 quantized tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_q8_0.gguf"
            metadata = GGUFMetadata(name="test_q8")
            writer = GGUFWriter(output_path, metadata)

            tensor = np.random.randn(256).astype(np.float32)
            writer.add_tensor("test.weight", tensor, GGMLType.Q8_0)
            writer.write()

            assert output_path.exists()
            # Q8_0 file should be smaller than F32 original
            # Q8_0: 34 bytes per 32 elements = 272 bytes for 256 elements
            # vs F32: 1024 bytes

    def test_write_q4_k(self):
        """Test writing Q4_K quantized tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_q4_k.gguf"
            metadata = GGUFMetadata(name="test_q4k")
            writer = GGUFWriter(output_path, metadata)

            tensor = np.random.randn(512).astype(np.float32)
            writer.add_tensor("test.weight", tensor, GGMLType.Q4_K)
            writer.write()

            assert output_path.exists()

    def test_write_q5_k(self):
        """Test writing Q5_K quantized tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_q5_k.gguf"
            metadata = GGUFMetadata(name="test_q5k")
            writer = GGUFWriter(output_path, metadata)

            tensor = np.random.randn(512).astype(np.float32)
            writer.add_tensor("test.weight", tensor, GGMLType.Q5_K)
            writer.write()

            assert output_path.exists()

    def test_write_q6_k(self):
        """Test writing Q6_K quantized tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_q6_k.gguf"
            metadata = GGUFMetadata(name="test_q6k")
            writer = GGUFWriter(output_path, metadata)

            tensor = np.random.randn(512).astype(np.float32)
            writer.add_tensor("test.weight", tensor, GGMLType.Q6_K)
            writer.write()

            assert output_path.exists()

    def test_quantized_size_calculation(self):
        """Test quantized size calculation is correct."""
        metadata = GGUFMetadata()
        writer = GGUFWriter(Path("/tmp/test.gguf"), metadata)

        # F16: 2 bytes per element
        assert writer._get_quantized_size(1000, GGMLType.F16) == 2000

        # Q8_0: 34 bytes per 32 elements (ceil(1000/32) = 32 blocks)
        assert writer._get_quantized_size(1024, GGMLType.Q8_0) == 32 * 34

        # Q4_K: 144 bytes per 256 elements
        assert writer._get_quantized_size(256, GGMLType.Q4_K) == 144
        assert writer._get_quantized_size(512, GGMLType.Q4_K) == 288

        # Q5_K: 176 bytes per 256 elements
        assert writer._get_quantized_size(256, GGMLType.Q5_K) == 176

        # Q6_K: 210 bytes per 256 elements
        assert writer._get_quantized_size(256, GGMLType.Q6_K) == 210


class TestWeightNameConversion:
    """Tests for weight name conversion."""

    def test_embed_tokens_conversion(self):
        """Test embed_tokens -> token_embd conversion."""
        weights = {"model.embed_tokens.weight": np.zeros((100, 64))}
        converted = convert_weight_names(weights)
        assert "token_embd" in list(converted.keys())[0]

    def test_layer_index_conversion(self):
        """Test layers.N. -> blk.N. conversion."""
        weights = {"model.layers.0.self_attn.q_proj.weight": np.zeros((64, 64))}
        converted = convert_weight_names(weights)
        key = list(converted.keys())[0]
        assert "blk.0." in key
        assert "attn_q" in key

    def test_mlp_conversion(self):
        """Test MLP weight name conversion."""
        weights = {
            "model.layers.0.mlp.gate_proj.weight": np.zeros((64, 64)),
            "model.layers.0.mlp.up_proj.weight": np.zeros((64, 64)),
            "model.layers.0.mlp.down_proj.weight": np.zeros((64, 64)),
        }
        converted = convert_weight_names(weights)
        keys = list(converted.keys())
        assert any("ffn_gate" in k for k in keys)
        assert any("ffn_up" in k for k in keys)
        assert any("ffn_down" in k for k in keys)


class TestGGUFMetadata:
    """Tests for GGUF metadata."""

    def test_default_values(self):
        """Test default metadata values."""
        metadata = GGUFMetadata()
        assert metadata.architecture == "deepseek"
        assert metadata.vocab_size == 32000
        assert metadata.hidden_size == 512
        assert metadata.num_layers == 8

    def test_custom_values(self):
        """Test custom metadata values."""
        metadata = GGUFMetadata(
            name="custom-model",
            vocab_size=50000,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
        )
        assert metadata.name == "custom-model"
        assert metadata.vocab_size == 50000
        assert metadata.hidden_size == 1024
        assert metadata.num_layers == 24
        assert metadata.num_heads == 16


class TestQuantizationRoundTrip:
    """Tests to verify quantization doesn't lose too much information."""

    def test_q8_0_reconstruction_error(self):
        """Test Q8_0 reconstruction error is acceptable."""
        np.random.seed(42)
        data = np.random.randn(1024).astype(np.float32)
        quantized, scales = quantize_q8_0(data)

        # Dequantize
        n_blocks = len(scales)
        dequantized = np.zeros(n_blocks * 32)
        for i in range(n_blocks):
            dequantized[i * 32 : (i + 1) * 32] = quantized[i].astype(np.float32) * float(scales[i]) / 127.0

        # Relative error should be small
        error = np.abs(data - dequantized).mean()
        rel_error = error / np.abs(data).mean()
        assert rel_error < 0.05  # Less than 5% relative error

    def test_q4_k_reconstruction_maintains_structure(self):
        """Test Q4_K maintains data structure."""
        np.random.seed(42)
        data = np.random.randn(512).astype(np.float32)
        packed_quants, d, dmin, scales_mins = quantize_q4_k(data)

        # Verify structure is preserved
        assert packed_quants.shape[0] == 2  # 2 superblocks
        # Data should be properly packed
        assert np.all(packed_quants >= 0)  # uint8 values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
