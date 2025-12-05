"""
Tests for ANE Activation Functions

Tests for ANESiLU, ANEGELU, ANESwiGLU, and ANEFusedSwiGLU.
"""

import pytest
import torch
import torch.nn.functional as F

from deepseek.mlx.ane.layers.activations import ANEFusedSwiGLU, ANEGELU, ANESiLU, ANESwiGLU


class TestANESiLU:
    """Tests for ANESiLU activation."""

    def test_forward_shape(self):
        """Test that forward pass preserves shape."""
        silu = ANESiLU()
        x = torch.randn(2, 16, 64)
        y = silu(x)

        assert y.shape == x.shape

    def test_matches_pytorch_silu(self):
        """Test that output matches PyTorch SiLU."""
        silu = ANESiLU(use_fp16=False)
        x = torch.randn(2, 16, 64)

        y_ane = silu(x)
        y_pytorch = F.silu(x)

        assert torch.allclose(y_ane, y_pytorch, atol=1e-5)

    def test_fp16_mode(self):
        """Test FP16 computation mode."""
        silu = ANESiLU(use_fp16=True)
        x = torch.randn(2, 16, 64)
        y = silu(x)

        # Output should preserve input dtype
        assert y.dtype == x.dtype

    def test_negative_values(self):
        """Test behavior with negative input values."""
        silu = ANESiLU()
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = silu(x)

        # SiLU(-x) ≈ -x * sigmoid(-x)
        # Check that negative inputs produce smaller negative outputs
        assert y[0] < 0  # Large negative
        assert y[2] == 0.0  # Zero stays zero
        assert y[4] > 0  # Positive stays positive


class TestANEGELU:
    """Tests for ANEGELU activation."""

    def test_forward_shape(self):
        """Test that forward pass preserves shape."""
        gelu = ANEGELU()
        x = torch.randn(2, 16, 64)
        y = gelu(x)

        assert y.shape == x.shape

    def test_matches_pytorch_gelu_approx(self):
        """Test that output matches PyTorch GELU with tanh approximation."""
        gelu = ANEGELU(use_fp16=False, approximate=True)
        x = torch.randn(2, 16, 64)

        y_ane = gelu(x)
        y_pytorch = F.gelu(x, approximate="tanh")

        assert torch.allclose(y_ane, y_pytorch, atol=1e-5)

    def test_exact_mode(self):
        """Test exact GELU mode."""
        # Use fp16=False to avoid precision loss in conversion
        gelu = ANEGELU(use_fp16=False, approximate=False)
        x = torch.randn(2, 16, 64)
        y = gelu(x)

        y_exact = F.gelu(x, approximate="none")
        assert torch.allclose(y, y_exact, atol=1e-4)

    def test_fp16_mode(self):
        """Test FP16 computation mode."""
        gelu = ANEGELU(use_fp16=True)
        x = torch.randn(2, 16, 64)
        y = gelu(x)

        assert y.dtype == x.dtype


class TestANESwiGLU:
    """Tests for ANESwiGLU activation (separate projections)."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shape."""
        # use_fp16=False to avoid dtype mismatch with FP32 weights
        swiglu = ANESwiGLU(in_features=64, hidden_features=128, out_features=64, use_fp16=False)
        x = torch.randn(2, 16, 64)
        y = swiglu(x)

        assert y.shape == (2, 16, 64)

    def test_default_out_features(self):
        """Test that out_features defaults to in_features."""
        swiglu = ANESwiGLU(in_features=64, hidden_features=128, use_fp16=False)
        x = torch.randn(2, 16, 64)
        y = swiglu(x)

        assert y.shape == (2, 16, 64)

    def test_different_out_features(self):
        """Test with different output features."""
        swiglu = ANESwiGLU(in_features=64, hidden_features=128, out_features=32, use_fp16=False)
        x = torch.randn(2, 16, 64)
        y = swiglu(x)

        assert y.shape == (2, 16, 32)

    def test_fp16_mode(self):
        """Test FP16 computation mode with FP16 input and weights."""
        swiglu = ANESwiGLU(in_features=64, hidden_features=128, use_fp16=True)
        # Convert weights to FP16 to match the conversion in forward pass
        swiglu = swiglu.half()
        x = torch.randn(2, 16, 64, dtype=torch.float16)
        y = swiglu(x)

        assert y.dtype == torch.float16

    def test_has_three_projections(self):
        """Test that SwiGLU has gate, up, and down projections."""
        swiglu = ANESwiGLU(in_features=64, hidden_features=128)

        assert hasattr(swiglu, "gate_proj")
        assert hasattr(swiglu, "up_proj")
        assert hasattr(swiglu, "down_proj")

        # Check projection dimensions
        assert swiglu.gate_proj.in_features == 64
        assert swiglu.gate_proj.out_features == 128
        assert swiglu.up_proj.in_features == 64
        assert swiglu.up_proj.out_features == 128
        assert swiglu.down_proj.in_features == 128
        assert swiglu.down_proj.out_features == 64


class TestANEFusedSwiGLU:
    """Tests for ANEFusedSwiGLU activation (fused gate+up projection)."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shape."""
        swiglu = ANEFusedSwiGLU(in_features=64, hidden_features=128, out_features=64, use_fp16=False)
        x = torch.randn(2, 16, 64)
        y = swiglu(x)

        assert y.shape == (2, 16, 64)

    def test_default_out_features(self):
        """Test that out_features defaults to in_features."""
        swiglu = ANEFusedSwiGLU(in_features=64, hidden_features=128, use_fp16=False)
        x = torch.randn(2, 16, 64)
        y = swiglu(x)

        assert y.shape == (2, 16, 64)

    def test_fused_projection(self):
        """Test that gate+up is a single fused projection."""
        swiglu = ANEFusedSwiGLU(in_features=64, hidden_features=128)

        # Should have fused gate_up_proj instead of separate
        assert hasattr(swiglu, "gate_up_proj")
        assert hasattr(swiglu, "down_proj")

        # Fused projection outputs 2*hidden_features
        assert swiglu.gate_up_proj.out_features == 256  # 2 * 128

    def test_matches_separate_swiglu(self):
        """Test that fused version produces similar output to separate."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)

        # Create both versions
        separate = ANESwiGLU(in_features=64, hidden_features=128, use_fp16=False)
        fused = ANEFusedSwiGLU(in_features=64, hidden_features=128, use_fp16=False)

        # Copy weights from separate to fused
        with torch.no_grad():
            fused.gate_up_proj.weight.data[:128] = separate.gate_proj.weight.data
            fused.gate_up_proj.weight.data[128:] = separate.up_proj.weight.data
            fused.down_proj.weight.data = separate.down_proj.weight.data

        y_separate = separate(x)
        y_fused = fused(x)

        assert torch.allclose(y_separate, y_fused, atol=1e-5)

    def test_fp16_mode(self):
        """Test FP16 computation mode with FP16 input and weights."""
        swiglu = ANEFusedSwiGLU(in_features=64, hidden_features=128, use_fp16=True)
        # Convert weights to FP16 to match the conversion in forward pass
        swiglu = swiglu.half()
        x = torch.randn(2, 16, 64, dtype=torch.float16)
        y = swiglu(x)

        assert y.dtype == torch.float16


class TestActivationIntegration:
    """Integration tests for activations in a neural network context."""

    def test_silu_in_ffn(self):
        """Test SiLU in a simple FFN."""
        ffn = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            ANESiLU(use_fp16=False),  # Disable FP16 to match weight dtype
            torch.nn.Linear(128, 64),
        )
        x = torch.randn(2, 16, 64)
        y = ffn(x)

        assert y.shape == x.shape

    def test_gelu_in_ffn(self):
        """Test GELU in a simple FFN."""
        ffn = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            ANEGELU(use_fp16=False),  # Disable FP16 to match weight dtype
            torch.nn.Linear(128, 64),
        )
        x = torch.randn(2, 16, 64)
        y = ffn(x)

        assert y.shape == x.shape

    def test_swiglu_as_ffn(self):
        """Test SwiGLU as complete FFN replacement."""
        # SwiGLU acts as a complete FFN block
        ffn = ANESwiGLU(in_features=64, hidden_features=128, out_features=64, use_fp16=False)
        x = torch.randn(2, 16, 64)
        y = ffn(x)

        assert y.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
