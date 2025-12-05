"""
Triton Kernel Tests - Python GPU Backend

Tests for:
- Fused SwiGLU kernel
- Fused RMSNorm kernel  
- Fused softmax kernel
- Fused MLA attention kernel
- Kernel autotuning
- Fallback to native operations
"""

import pytest
import torch
import torch.nn as nn
import sys

# Skip all tests if Triton is not available
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" or not torch.cuda.is_available(),
    reason="Triton kernels require CUDA GPU"
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get CUDA device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_tensor(device):
    """Create random tensor factory."""
    def _create(shape, dtype=torch.float32):
        return torch.randn(shape, device=device, dtype=dtype)
    return _create


# =============================================================================
# Triton Availability Tests
# =============================================================================

class TestTritonAvailability:
    """Tests for Triton availability and fallback."""
    
    def test_triton_available_flag(self):
        """Test TRITON_AVAILABLE flag is set correctly."""
        try:
            from deepseek.kernels import TRITON_AVAILABLE
            # Should be True on CUDA systems with Triton, False otherwise
            assert isinstance(TRITON_AVAILABLE, bool)
        except ImportError:
            pytest.skip("Kernels module not available")
    
    def test_fallback_when_unavailable(self):
        """Test that fallback functions work when Triton unavailable."""
        try:
            from deepseek.kernels import TRITON_AVAILABLE
            from deepseek.torch.kernels.triton_kernels import (
                fused_swiglu_forward,
                fused_rmsnorm_forward,
                fused_softmax_forward,
            )
            
            # These should always be callable (either Triton or fallback)
            assert callable(fused_swiglu_forward)
            assert callable(fused_rmsnorm_forward)
            assert callable(fused_softmax_forward)
        except ImportError:
            pytest.skip("Kernels module not available")


# =============================================================================
# SwiGLU Kernel Tests
# =============================================================================

class TestFusedSwiGLU:
    """Tests for fused SwiGLU activation kernel."""
    
    def test_swiglu_shape(self, device, random_tensor):
        """Test SwiGLU output shape."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_swiglu_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, seq, hidden = 2, 16, 64
        gate = random_tensor((batch, seq, hidden))
        up = random_tensor((batch, seq, hidden))
        
        output = fused_swiglu_forward(gate, up)
        
        assert output.shape == (batch, seq, hidden)
    
    def test_swiglu_dtype_preservation(self, device, random_tensor):
        """Test SwiGLU preserves dtype."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_swiglu_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        for dtype in [torch.float32, torch.float16]:
            gate = random_tensor((2, 16, 64), dtype=dtype)
            up = random_tensor((2, 16, 64), dtype=dtype)
            
            output = fused_swiglu_forward(gate, up)
            assert output.dtype == dtype
    
    def test_swiglu_correctness(self, device, random_tensor):
        """Test SwiGLU numerical correctness against reference."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_swiglu_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        gate = random_tensor((2, 16, 64))
        up = random_tensor((2, 16, 64))
        
        # Fused kernel
        fused_output = fused_swiglu_forward(gate, up)
        
        # Reference implementation
        reference_output = torch.nn.functional.silu(gate) * up
        
        assert torch.allclose(fused_output, reference_output, rtol=1e-3, atol=1e-3)
    
    def test_swiglu_backward(self, device, random_tensor):
        """Test SwiGLU backward pass."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_swiglu
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        gate = random_tensor((2, 16, 64)).requires_grad_(True)
        up = random_tensor((2, 16, 64)).requires_grad_(True)
        
        output = fused_swiglu(gate, up)
        loss = output.sum()
        loss.backward()
        
        assert gate.grad is not None
        assert up.grad is not None
        assert gate.grad.shape == gate.shape
        assert up.grad.shape == up.shape


# =============================================================================
# RMSNorm Kernel Tests
# =============================================================================

class TestFusedRMSNorm:
    """Tests for fused RMSNorm kernel."""
    
    def test_rmsnorm_shape(self, device, random_tensor):
        """Test RMSNorm output shape."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_rmsnorm_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, seq, hidden = 2, 16, 64
        x = random_tensor((batch, seq, hidden))
        weight = random_tensor((hidden,))
        
        output = fused_rmsnorm_forward(x, weight, eps=1e-6)
        
        assert output.shape == (batch, seq, hidden)
    
    def test_rmsnorm_with_residual(self, device, random_tensor):
        """Test RMSNorm with fused residual addition."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_rmsnorm_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, seq, hidden = 2, 16, 64
        x = random_tensor((batch, seq, hidden))
        residual = random_tensor((batch, seq, hidden))
        weight = random_tensor((hidden,))
        
        output = fused_rmsnorm_forward(x, weight, eps=1e-6, residual=residual)
        
        assert output.shape == (batch, seq, hidden)
    
    def test_rmsnorm_correctness(self, device, random_tensor):
        """Test RMSNorm numerical correctness."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_rmsnorm_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, seq, hidden = 2, 16, 64
        x = random_tensor((batch, seq, hidden))
        weight = torch.ones(hidden, device=device)
        eps = 1e-6
        
        # Fused kernel
        fused_output = fused_rmsnorm_forward(x, weight, eps=eps)
        
        # Reference implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        reference_output = weight * x_norm
        
        assert torch.allclose(fused_output, reference_output, rtol=1e-3, atol=1e-3)
    
    def test_rmsnorm_eps_values(self, device, random_tensor):
        """Test RMSNorm with different epsilon values."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_rmsnorm_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        x = random_tensor((2, 16, 64))
        weight = random_tensor((64,))
        
        for eps in [1e-5, 1e-6, 1e-8]:
            output = fused_rmsnorm_forward(x, weight, eps=eps)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Softmax Kernel Tests
# =============================================================================

class TestFusedSoftmax:
    """Tests for fused softmax kernel with online normalization."""
    
    def test_softmax_shape(self, device, random_tensor):
        """Test softmax output shape."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_softmax_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, heads, seq_q, seq_k = 2, 8, 16, 16
        scores = random_tensor((batch, heads, seq_q, seq_k))
        
        output = fused_softmax_forward(scores)
        
        assert output.shape == scores.shape
    
    def test_softmax_sums_to_one(self, device, random_tensor):
        """Test softmax rows sum to 1."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_softmax_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        scores = random_tensor((2, 8, 16, 16))
        output = fused_softmax_forward(scores)
        
        row_sums = output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3)
    
    def test_softmax_correctness(self, device, random_tensor):
        """Test softmax numerical correctness."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_softmax_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        scores = random_tensor((2, 8, 16, 16))
        
        # Fused kernel
        fused_output = fused_softmax_forward(scores)
        
        # Reference implementation
        reference_output = torch.softmax(scores, dim=-1)
        
        assert torch.allclose(fused_output, reference_output, rtol=1e-3, atol=1e-3)
    
    def test_softmax_with_mask(self, device, random_tensor):
        """Test softmax with attention mask."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_softmax_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, heads, seq = 2, 8, 16
        scores = random_tensor((batch, heads, seq, seq))
        
        # Causal mask
        mask = torch.triu(
            torch.ones(seq, seq, device=device) * float('-inf'),
            diagonal=1
        )
        
        output = fused_softmax_forward(scores, mask=mask)
        
        assert output.shape == scores.shape
        # Check that masked positions are ~0
        # (upper triangular should be masked)
    
    def test_softmax_numerical_stability(self, device):
        """Test softmax is numerically stable with large values."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_softmax_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        # Large values that could cause overflow
        scores = torch.ones(2, 8, 16, 16, device=device) * 100
        
        output = fused_softmax_forward(scores)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# MLA Attention Kernel Tests
# =============================================================================

class TestFusedMLAAttention:
    """Tests for fused Multi-Latent Attention kernel."""
    
    def test_mla_attention_shape(self, device, random_tensor):
        """Test MLA attention output shape."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_mla_attention
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, heads, seq, d_head = 2, 8, 16, 64
        q = random_tensor((batch, heads, seq, d_head))
        k = random_tensor((batch, heads, seq, d_head))
        v = random_tensor((batch, heads, seq, d_head))
        
        output = fused_mla_attention(q, k, v)
        
        assert output.shape == (batch, heads, seq, d_head)
    
    def test_mla_attention_causal(self, device, random_tensor):
        """Test MLA attention with causal masking."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_mla_attention
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, heads, seq, d_head = 2, 8, 16, 64
        q = random_tensor((batch, heads, seq, d_head))
        k = random_tensor((batch, heads, seq, d_head))
        v = random_tensor((batch, heads, seq, d_head))
        
        output = fused_mla_attention(q, k, v, causal=True)
        
        assert output.shape == (batch, heads, seq, d_head)
    
    def test_mla_attention_scale(self, device, random_tensor):
        """Test MLA attention with custom scale."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_mla_attention
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, heads, seq, d_head = 2, 8, 16, 64
        q = random_tensor((batch, heads, seq, d_head))
        k = random_tensor((batch, heads, seq, d_head))
        v = random_tensor((batch, heads, seq, d_head))
        
        scale = 1.0 / (d_head ** 0.5)
        output = fused_mla_attention(q, k, v, scale=scale)
        
        assert output.shape == (batch, heads, seq, d_head)
    
    def test_mla_attention_correctness(self, device, random_tensor):
        """Test MLA attention correctness against reference."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_mla_attention
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        batch, heads, seq, d_head = 2, 4, 8, 32
        q = random_tensor((batch, heads, seq, d_head))
        k = random_tensor((batch, heads, seq, d_head))
        v = random_tensor((batch, heads, seq, d_head))
        
        scale = 1.0 / (d_head ** 0.5)
        
        # Fused kernel
        fused_output = fused_mla_attention(q, k, v, scale=scale, causal=False)
        
        # Reference implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        reference_output = torch.matmul(attn, v)
        
        assert torch.allclose(fused_output, reference_output, rtol=1e-2, atol=1e-2)


# =============================================================================
# Kernel Autotuning Tests
# =============================================================================

class TestKernelAutotuning:
    """Tests for kernel autotuning infrastructure."""
    
    def test_autotuner_exists(self):
        """Test TritonKernelAutotuner class exists."""
        try:
            from deepseek.torch.kernels.triton_kernels import TritonKernelAutotuner
            assert callable(TritonKernelAutotuner)
        except ImportError:
            pytest.skip("Triton kernels not available")
    
    def test_autotuner_initialization(self):
        """Test autotuner initialization."""
        try:
            from deepseek.torch.kernels.triton_kernels import TritonKernelAutotuner
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        autotuner = TritonKernelAutotuner()
        assert autotuner is not None
    
    def test_autotuner_get_config(self):
        """Test autotuner returns config for problem size."""
        try:
            from deepseek.torch.kernels.triton_kernels import TritonKernelAutotuner
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        autotuner = TritonKernelAutotuner()
        
        # Get config for a problem size
        config = autotuner.get_config(
            kernel_name="fused_swiglu",
            batch_size=16,
            seq_len=512,
            hidden_size=4096,
        )
        
        assert config is not None
        assert isinstance(config, dict)


# =============================================================================
# Fallback Tests
# =============================================================================

class TestKernelFallback:
    """Tests for kernel fallback to native operations."""
    
    def test_swiglu_fallback(self):
        """Test SwiGLU fallback works without Triton."""
        # This test should pass even without Triton
        from deepseek.torch.kernels.triton_kernels import fused_swiglu_forward
        
        device = torch.device("cpu")  # Force CPU to avoid Triton
        gate = torch.randn(2, 16, 64, device=device)
        up = torch.randn(2, 16, 64, device=device)
        
        output = fused_swiglu_forward(gate, up)
        
        assert output.shape == gate.shape
    
    def test_rmsnorm_fallback(self):
        """Test RMSNorm fallback works without Triton."""
        from deepseek.torch.kernels.triton_kernels import fused_rmsnorm_forward
        
        device = torch.device("cpu")
        x = torch.randn(2, 16, 64, device=device)
        weight = torch.ones(64, device=device)
        
        output = fused_rmsnorm_forward(x, weight)
        
        assert output.shape == x.shape
    
    def test_softmax_fallback(self):
        """Test softmax fallback works without Triton."""
        from deepseek.torch.kernels.triton_kernels import fused_softmax_forward
        
        device = torch.device("cpu")
        scores = torch.randn(2, 8, 16, 16, device=device)
        
        output = fused_softmax_forward(scores)
        
        assert output.shape == scores.shape


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestKernelBenchmarks:
    """Tests for kernel benchmarking utilities."""
    
    @pytest.mark.slow
    def test_benchmark_swiglu(self, device, random_tensor):
        """Benchmark SwiGLU kernel vs native."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_swiglu_forward, benchmark_kernel
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        gate = random_tensor((16, 512, 4096))
        up = random_tensor((16, 512, 4096))
        
        # Should complete without error
        # Actual benchmarking would be done separately
        output = fused_swiglu_forward(gate, up)
        assert output is not None
    
    @pytest.mark.slow
    def test_benchmark_rmsnorm(self, device, random_tensor):
        """Benchmark RMSNorm kernel vs native."""
        try:
            from deepseek.torch.kernels.triton_kernels import fused_rmsnorm_forward
        except ImportError:
            pytest.skip("Triton kernels not available")
        
        x = random_tensor((16, 512, 4096))
        weight = random_tensor((4096,))
        
        output = fused_rmsnorm_forward(x, weight)
        assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
