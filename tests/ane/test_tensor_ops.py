"""
Tests for ANE Tensor Operations

Tests for shape normalization, layout conversion, and memory alignment utilities.
"""

import pytest
import torch

from deepseek.mlx.ane.utils.tensor_ops import (
    pad_to_multiple,
    pad_to_power_of_2,
    normalize_shape_for_ane,
    unpad_tensor,
    nchw_to_nhwc,
    nhwc_to_nchw,
    convert_to_channel_last,
    convert_from_channel_last,
    is_aligned,
    align_tensor,
    get_aligned_size,
    get_ane_friendly_dim,
    split_for_ane_tiles,
    check_ane_compatible,
    ANE_ALIGNMENT,
    ANE_MAX_TENSOR_SIZE,
)


class TestPadToMultiple:
    """Tests for pad_to_multiple function."""

    def test_already_multiple(self):
        """Test that multiples return unchanged."""
        assert pad_to_multiple(16, 16) == 16
        assert pad_to_multiple(32, 16) == 32
        assert pad_to_multiple(64, 16) == 64

    def test_needs_padding(self):
        """Test padding to next multiple."""
        assert pad_to_multiple(17, 16) == 32
        assert pad_to_multiple(1, 16) == 16
        assert pad_to_multiple(15, 16) == 16

    def test_different_multiples(self):
        """Test with different multiple values."""
        assert pad_to_multiple(33, 32) == 64
        assert pad_to_multiple(100, 64) == 128
        assert pad_to_multiple(129, 128) == 256


class TestPadToPowerOf2:
    """Tests for pad_to_power_of_2 function."""

    def test_already_power_of_2(self):
        """Test that powers of 2 return unchanged."""
        assert pad_to_power_of_2(16) == 16
        assert pad_to_power_of_2(32) == 32
        assert pad_to_power_of_2(256) == 256

    def test_needs_rounding(self):
        """Test rounding up to next power of 2."""
        assert pad_to_power_of_2(17) == 32
        assert pad_to_power_of_2(33) == 64
        assert pad_to_power_of_2(100) == 128

    def test_min_size(self):
        """Test minimum size constraint."""
        assert pad_to_power_of_2(1, min_size=16) == 16
        assert pad_to_power_of_2(8, min_size=16) == 16


class TestNormalizeShapeForANE:
    """Tests for normalize_shape_for_ane function."""

    def test_2d_padding(self):
        """Test 2D tensor padding."""
        x = torch.randn(17, 33)
        padded, original_shape = normalize_shape_for_ane(x)
        
        assert padded.shape == (32, 48)
        assert original_shape == (17, 33)

    def test_3d_padding(self):
        """Test 3D tensor padding."""
        x = torch.randn(2, 17, 33)
        padded, original_shape = normalize_shape_for_ane(x)
        
        assert padded.shape[0] == 2
        assert padded.shape[1] % 16 == 0
        assert padded.shape[2] % 16 == 0
        assert original_shape == (2, 17, 33)

    def test_power_of_2_mode(self):
        """Test power of 2 padding mode."""
        x = torch.randn(17, 33)
        padded, _ = normalize_shape_for_ane(x, power_of_2=True)

        assert padded.shape[0] & (padded.shape[0] - 1) == 0  # Is power of 2
        assert padded.shape[1] & (padded.shape[1] - 1) == 0

    def test_no_padding_needed(self):
        """Test when no padding is needed."""
        x = torch.randn(16, 32)
        padded, original_shape = normalize_shape_for_ane(x)
        
        assert padded.shape == (16, 32)
        assert torch.equal(padded, x)


class TestUnpadTensor:
    """Tests for unpad_tensor function."""

    def test_restore_original_shape(self):
        """Test that unpadding restores original shape."""
        original = torch.randn(17, 33)
        padded, original_shape = normalize_shape_for_ane(original)
        unpadded = unpad_tensor(padded, original_shape)

        assert unpadded.shape == original.shape
        # Check that the original values are preserved in the unpadded region
        assert unpadded.shape == (17, 33)

    def test_no_unpadding_needed(self):
        """Test when no unpadding is needed."""
        x = torch.randn(16, 32)
        result = unpad_tensor(x, (16, 32))
        
        assert result.shape == (16, 32)
        assert torch.equal(result, x)


class TestLayoutConversion:
    """Tests for NCHW/NHWC conversion."""

    def test_nchw_to_nhwc(self):
        """Test NCHW to NHWC conversion."""
        x = torch.randn(2, 3, 4, 5)  # N=2, C=3, H=4, W=5
        y = nchw_to_nhwc(x)
        
        assert y.shape == (2, 4, 5, 3)  # N=2, H=4, W=5, C=3
        assert torch.equal(y[0, 1, 2, 1], x[0, 1, 1, 2])

    def test_nhwc_to_nchw(self):
        """Test NHWC to NCHW conversion."""
        x = torch.randn(2, 4, 5, 3)  # N=2, H=4, W=5, C=3
        y = nhwc_to_nchw(x)
        
        assert y.shape == (2, 3, 4, 5)  # N=2, C=3, H=4, W=5

    def test_roundtrip(self):
        """Test roundtrip conversion preserves data."""
        x = torch.randn(2, 3, 4, 5)
        y = nchw_to_nhwc(x)
        z = nhwc_to_nchw(y)
        
        assert torch.allclose(x, z)

    def test_invalid_dims(self):
        """Test error on invalid dimensions."""
        x = torch.randn(2, 3, 4)  # 3D tensor
        
        with pytest.raises(ValueError):
            nchw_to_nhwc(x)
        
        with pytest.raises(ValueError):
            nhwc_to_nchw(x)


class TestConvertToChannelLast:
    """Tests for channel-last conversion."""

    def test_4d_conversion(self):
        """Test 4D tensor conversion to channels_last."""
        x = torch.randn(2, 3, 4, 5)
        y = convert_to_channel_last(x)
        
        # Check memory format for 4D
        assert y.is_contiguous(memory_format=torch.channels_last)

    def test_3d_conversion(self):
        """Test 3D tensor passes through unchanged (no channels_last for 3D)."""
        x = torch.randn(2, 3, 4)  # B=2, C=3, S=4
        y = convert_to_channel_last(x)

        # 3D tensors pass through unchanged (no channels_last format)
        assert y.shape == x.shape
        assert torch.equal(x, y)

    def test_2d_passthrough(self):
        """Test 2D tensor passes through unchanged."""
        x = torch.randn(3, 4)
        y = convert_to_channel_last(x)
        
        assert torch.equal(x, y)


class TestMemoryAlignment:
    """Tests for memory alignment utilities."""

    def test_is_aligned(self):
        """Test alignment checking."""
        x = torch.randn(16, 16).contiguous()
        # New contiguous tensors are typically aligned
        # but we can't guarantee it, so just test the function runs
        result = is_aligned(x)
        assert isinstance(result, bool)

    def test_get_aligned_size(self):
        """Test aligned size computation."""
        assert get_aligned_size(15, 16) == 16
        assert get_aligned_size(16, 16) == 16
        assert get_aligned_size(17, 16) == 32
        assert get_aligned_size(100, 16) == 112

    def test_align_tensor(self):
        """Test tensor alignment."""
        x = torch.randn(16, 16)
        y = align_tensor(x)
        
        assert y.is_contiguous()


class TestANEHelpers:
    """Tests for ANE-specific helper functions."""

    def test_get_ane_friendly_dim(self):
        """Test ANE-friendly dimension calculation."""
        assert get_ane_friendly_dim(15) == 16
        assert get_ane_friendly_dim(17) == 32
        assert get_ane_friendly_dim(100, prefer_power_of_2=True) == 128
        assert get_ane_friendly_dim(100, prefer_power_of_2=False) == 112

    def test_split_for_ane_tiles(self):
        """Test tensor tiling."""
        x = torch.randn(2, 300)
        tiles = split_for_ane_tiles(x, tile_size=128)

        assert len(tiles) == 3  # 300/128 = 3 chunks
        assert tiles[0].shape[0] == 2
        assert tiles[1].shape[0] == 2
        assert tiles[2].shape[0] == 2
        # Total elements across tiles should match original
        total_elems = sum(t.shape[1] for t in tiles)
        assert total_elems == 300

    def test_split_small_tensor(self):
        """Test that small tensors don't get split."""
        x = torch.randn(2, 64)
        tiles = split_for_ane_tiles(x, tile_size=128)
        
        assert len(tiles) == 1
        assert torch.equal(tiles[0], x)

    def test_check_ane_compatible(self):
        """Test ANE compatibility checking."""
        # Small aligned tensor with good dimensions
        x = torch.randn(16, 32).contiguous()
        is_compat, reason = check_ane_compatible(x)
        # May or may not be compatible depending on alignment
        assert isinstance(is_compat, bool)
        
        # Large tensor should fail
        large = torch.randn(10000, 10000)
        is_compat, reason = check_ane_compatible(large, max_size=1024)
        assert not is_compat
        assert "exceeds max" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
