"""
GPU Optimization Tests - MLX Backend

Tests for:
- Metal acceleration verification
- BF16/Float16 support
- MLX memory profiling
- Activation checkpointing
- Lazy evaluation optimization
"""

import pytest
import sys

# Skip all tests if not on macOS or MLX not available
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX tests require macOS with Apple Silicon"
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_mlx_model():
    """Create a simple MLX model for testing."""
    if not MLX_AVAILABLE:
        pytest.skip("MLX not available")
    
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(1000, hidden_size)
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1000)
        
        def __call__(self, x):
            x = self.embedding(x)
            x = nn.relu(self.linear1(x))
            return self.linear2(x)
    
    return SimpleModel()


# =============================================================================
# Metal Acceleration Tests (Section 1.1)
# =============================================================================

class TestMetalAcceleration:
    """Tests for Metal GPU acceleration."""
    
    def test_mlx_available(self):
        """Test MLX is available."""
        assert MLX_AVAILABLE, "MLX should be available on macOS"
    
    def test_verify_metal_acceleration(self):
        """Test Metal acceleration is active."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import verify_metal_acceleration
        
        result = verify_metal_acceleration()
        assert isinstance(result, bool)
        # On Apple Silicon, this should be True
        # On Intel Macs or VMs, it may be False
    
    def test_get_mlx_device_info(self):
        """Test device info retrieval."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import get_mlx_device_info
        
        info = get_mlx_device_info()
        assert isinstance(info, dict)
        # Should have some info about the device
    
    def test_mlx_array_creation(self):
        """Test basic MLX array creation on Metal."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        x = mx.array([1.0, 2.0, 3.0])
        mx.eval(x)  # Force evaluation
        
        assert x.shape == (3,)
        assert x.dtype == mx.float32
    
    def test_mlx_matmul(self):
        """Test matrix multiplication on Metal."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        a = mx.random.normal((64, 128))
        b = mx.random.normal((128, 64))
        
        c = a @ b
        mx.eval(c)
        
        assert c.shape == (64, 64)


# =============================================================================
# BF16/Float16 Support Tests (Section 1.3)
# =============================================================================

class TestMLXPrecision:
    """Tests for MLX precision/dtype support."""
    
    def test_verify_bfloat16_support(self):
        """Test BF16 support verification."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import verify_bfloat16_support
        
        result = verify_bfloat16_support()
        assert isinstance(result, dict)
        # The function returns 'bf16_available' key
        assert "bf16_available" in result
        assert isinstance(result["bf16_available"], bool)
    
    def test_get_optimal_mlx_dtype(self):
        """Test optimal dtype selection."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import get_optimal_mlx_dtype
        
        dtype = get_optimal_mlx_dtype()
        # Should return a valid MLX dtype
        assert dtype in [mx.float16, mx.bfloat16, mx.float32]
    
    def test_float16_operations(self):
        """Test Float16 tensor operations."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        x = mx.random.normal((32, 64)).astype(mx.float16)
        y = mx.random.normal((64, 32)).astype(mx.float16)
        
        z = x @ y
        mx.eval(z)
        
        assert z.dtype == mx.float16
        assert z.shape == (32, 32)
    
    def test_bfloat16_operations(self):
        """Test BFloat16 tensor operations if supported."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import verify_bfloat16_support
        
        bf16_info = verify_bfloat16_support()
        if not bf16_info.get("bf16_available", False):
            pytest.skip("BF16 not supported on this hardware")
        
        x = mx.random.normal((32, 64)).astype(mx.bfloat16)
        y = mx.random.normal((64, 32)).astype(mx.bfloat16)
        
        z = x @ y
        mx.eval(z)
        
        assert z.dtype == mx.bfloat16
        assert z.shape == (32, 32)
    
    def test_mixed_precision_conversion(self):
        """Test converting between precisions."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        x_fp32 = mx.random.normal((32, 64))
        x_fp16 = x_fp32.astype(mx.float16)
        x_back = x_fp16.astype(mx.float32)
        
        mx.eval(x_fp32, x_fp16, x_back)
        
        assert x_fp32.dtype == mx.float32
        assert x_fp16.dtype == mx.float16
        assert x_back.dtype == mx.float32


# =============================================================================
# Memory Profiling Tests (Section 1.5)
# =============================================================================

class TestMLXMemory:
    """Tests for MLX memory profiling utilities."""
    
    def test_mlx_memory_stats(self):
        """Test MLXMemoryStats dataclass."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import MLXMemoryStats
        
        # Use actual field names from implementation
        stats = MLXMemoryStats(
            peak_memory_mb=2048.0,
            cache_memory_mb=512.0,
            wired_memory_mb=1024.0,
        )
        
        assert stats.peak_memory_mb == 2048.0
        assert stats.cache_memory_mb == 512.0
        assert stats.wired_memory_mb == 1024.0
    
    def test_count_parameters(self, simple_mlx_model):
        """Test parameter counting."""
        from deepseek.mlx.optimization import count_parameters
        
        count = count_parameters(simple_mlx_model)
        
        assert isinstance(count, int)
        assert count > 0
    
    def test_get_model_memory_mb(self, simple_mlx_model):
        """Test model memory estimation."""
        from deepseek.mlx.optimization import get_model_memory_mb
        
        memory_mb = get_model_memory_mb(simple_mlx_model)
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
    
    def test_get_recommended_model_size(self):
        """Test recommended model size for different chips."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import get_recommended_model_size
        
        for chip in ["M1", "M1 Pro", "M1 Max", "M2", "M3"]:
            size_info = get_recommended_model_size(chip)
            
            assert isinstance(size_info, dict)
            # Should have some reasonable recommendations
    
    def test_memory_efficient_patterns(self):
        """Test memory-efficient computation patterns."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        # Create large tensors
        a = mx.random.normal((1000, 1000))
        b = mx.random.normal((1000, 1000))
        
        # Lazy evaluation - no memory allocated yet
        c = a @ b
        d = c + a
        
        # Only evaluated when needed
        mx.eval(d)
        
        assert d.shape == (1000, 1000)


# =============================================================================
# Lazy Evaluation Tests (Section 1.2)
# =============================================================================

class TestMLXLazyEvaluation:
    """Tests for MLX lazy evaluation optimization."""
    
    def test_force_eval(self):
        """Test force_eval utility."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import force_eval
        
        a = mx.random.normal((64, 64))
        b = mx.random.normal((64, 64))
        c = a @ b
        
        force_eval(c)
        
        # Should be evaluated now
        assert c.shape == (64, 64)
    
    def test_eval_and_sync(self):
        """Test eval_and_sync utility."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import eval_and_sync
        
        a = mx.random.normal((64, 64))
        b = a @ a.T
        
        mx.eval(b)
        eval_and_sync()  # Should complete without error
        
        assert b.shape == (64, 64)
    
    def test_lazy_chain_optimization(self):
        """Test that lazy evaluation chains operations efficiently."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        x = mx.random.normal((100, 100))
        
        # Chain of operations - should be fused
        y = x + 1
        y = y * 2
        y = mx.maximum(y, mx.zeros_like(y))  # ReLU using maximum
        y = y - 1
        
        # Only one eval needed
        mx.eval(y)
        
        assert y.shape == (100, 100)
    
    def test_mlx_training_context(self, simple_mlx_model):
        """Test MLXTrainingContext."""
        from deepseek.mlx.optimization import MLXTrainingContext
        
        # Use actual parameter names from implementation
        ctx = MLXTrainingContext(
            eval_every_n_steps=10,
            log_memory=True,
        )
        
        assert ctx.eval_every_n_steps == 10
        assert ctx.log_memory is True


# =============================================================================
# Activation Checkpointing Tests (Section 1.4)
# =============================================================================

class TestMLXCheckpointing:
    """Tests for MLX activation checkpointing."""
    
    def test_checkpoint_config(self):
        """Test MLXCheckpointConfig dataclass."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import MLXCheckpointConfig
        
        config = MLXCheckpointConfig(
            enabled=True,
            checkpoint_every_n_layers=2,
        )
        
        assert config.enabled is True
        assert config.checkpoint_every_n_layers == 2
    
    def test_activation_checkpointing_init(self):
        """Test MLXActivationCheckpointing initialization."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import MLXActivationCheckpointing, MLXCheckpointConfig
        
        config = MLXCheckpointConfig()
        checkpointer = MLXActivationCheckpointing(config)
        
        assert checkpointer.config == config
    
    def test_create_checkpointed_model_wrapper(self, simple_mlx_model):
        """Test checkpointed model wrapper creation."""
        from deepseek.mlx.optimization import create_checkpointed_model_wrapper
        
        wrapped = create_checkpointed_model_wrapper(simple_mlx_model)
        
        # Should still be callable
        x = mx.array([[1, 2, 3, 4, 5]])
        output = wrapped(x)
        mx.eval(output)
        
        assert output.shape[0] == 1
        assert output.shape[-1] == 1000  # vocab size
    
    def test_checkpointing_reduces_memory(self, simple_mlx_model):
        """Test that checkpointing reduces peak memory."""
        from deepseek.mlx.optimization import (
            MLXActivationCheckpointing,
            MLXCheckpointConfig,
        )
        
        # This is a conceptual test - actual memory reduction
        # would need to be measured in a real training loop
        
        config = MLXCheckpointConfig(enabled=True)
        checkpointer = MLXActivationCheckpointing(config)
        
        x = mx.array([[1, 2, 3, 4, 5]])
        
        # Use checkpoint_forward method with a layer function
        def forward_fn(inp):
            return simple_mlx_model(inp)
        
        result = checkpointer.checkpoint_forward(forward_fn, x, layer_idx=0)
        mx.eval(result)
        
        assert result is not None


# =============================================================================
# Inference Optimization Tests
# =============================================================================

class TestMLXInference:
    """Tests for MLX inference optimizations."""
    
    def test_optimize_for_inference(self, simple_mlx_model):
        """Test inference optimization."""
        from deepseek.mlx.optimization import optimize_for_inference
        
        optimized = optimize_for_inference(simple_mlx_model)
        
        # Should return a model
        assert optimized is not None
        
        # Should be callable
        x = mx.array([[1, 2, 3, 4, 5]])
        output = optimized(x)
        mx.eval(output)
        
        assert output.shape[-1] == 1000


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestMLXBenchmarks:
    """Tests for MLX benchmarking utilities."""
    
    def test_benchmark_forward_pass(self, simple_mlx_model):
        """Test forward pass benchmarking."""
        from deepseek.mlx.optimization import benchmark_forward_pass
        
        # Use actual parameter signature: (model, input_shape, num_iterations, warmup_iterations)
        results = benchmark_forward_pass(
            simple_mlx_model,
            input_shape=(2, 10),  # (batch_size, seq_len)
            num_iterations=2,
            warmup_iterations=1,
        )
        
        assert "avg_time_ms" in results
        assert results["avg_time_ms"] > 0
    
    def test_benchmark_training_step(self, simple_mlx_model):
        """Test training step benchmarking."""
        from deepseek.mlx.optimization import benchmark_training_step
        import mlx.nn as nn
        
        # Define a simple loss function
        def cross_entropy_loss(logits, targets):
            return mx.mean(nn.losses.cross_entropy(logits, targets))
        
        # Use actual parameter signature: (model, loss_fn, input_shape, num_iterations, warmup_iterations)
        results = benchmark_training_step(
            simple_mlx_model,
            loss_fn=cross_entropy_loss,
            input_shape=(2, 10),  # (batch_size, seq_len)
            num_iterations=2,
            warmup_iterations=1,
        )
        
        assert "avg_time_ms" in results
        assert results["avg_time_ms"] > 0
    
    def test_benchmark_vs_cpu(self):
        """Test benchmarking GPU vs CPU (conceptual)."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        # MLX always uses GPU when available
        # This test verifies operations complete in reasonable time
        
        import time
        
        size = 1000
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))
        
        start = time.perf_counter()
        c = a @ b
        mx.eval(c)
        elapsed = time.perf_counter() - start
        
        # Should complete quickly on GPU
        assert elapsed < 5.0  # 5 seconds max for 1000x1000 matmul


# =============================================================================
# Memory-Efficient Batch Processing Tests
# =============================================================================

class TestMLXBatchProcessing:
    """Tests for memory-efficient batch processing."""
    
    def test_create_batches_memory_efficient(self):
        """Test memory-efficient batch creation."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import create_batches_memory_efficient
        
        # Create some data as MLX array
        data = mx.arange(1000)
        
        # Function returns a callable that generates batches
        batch_generator = create_batches_memory_efficient(
            data,
            batch_size=4,
            seq_len=10,
            shuffle=False,
        )
        
        # Generate a batch
        input_ids, labels = batch_generator()
        mx.eval(input_ids, labels)
        
        assert input_ids.shape == (4, 10)
        assert labels.shape == (4, 10)
    
    def test_create_batches_shuffled(self):
        """Test batch creation with shuffling."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import create_batches_memory_efficient
        
        data = mx.arange(1000)
        
        # Create batch generator with shuffle
        batch_generator = create_batches_memory_efficient(
            data,
            batch_size=4,
            seq_len=10,
            shuffle=True,
        )
        
        # Generate two batches - they should likely be different with shuffle
        batch1_input, _ = batch_generator()
        batch2_input, _ = batch_generator()
        mx.eval(batch1_input, batch2_input)
        
        # Both should have correct shape
        assert batch1_input.shape == (4, 10)
        assert batch2_input.shape == (4, 10)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMLXGPUOptimizationIntegration:
    """Integration tests for MLX GPU optimization features."""
    
    def test_full_training_step(self, simple_mlx_model):
        """Test a complete training step with all optimizations."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import (
            MLXTrainingContext,
            get_optimal_mlx_dtype,
            force_eval,
        )
        
        ctx = MLXTrainingContext(eval_every_n_steps=1)
        dtype = get_optimal_mlx_dtype()
        
        # Create input
        x = mx.array([[1, 2, 3, 4, 5]])
        targets = mx.array([[2, 3, 4, 5, 6]])
        
        # Forward pass
        output = simple_mlx_model(x)
        
        # Simple loss
        loss = mx.mean((output[:, :-1, :] - mx.zeros_like(output[:, :-1, :])) ** 2)
        
        # Backward pass (compute gradients)
        grads = mx.grad(lambda m, x: mx.mean(m(x) ** 2))(simple_mlx_model, x)
        
        # Force evaluation
        force_eval(loss)
        
        assert loss.item() >= 0
    
    def test_precision_with_checkpointing(self, simple_mlx_model):
        """Test precision selection with activation checkpointing."""
        if not MLX_AVAILABLE:
            pytest.skip("MLX not available")
        
        from deepseek.mlx.optimization import (
            get_optimal_mlx_dtype,
            create_checkpointed_model_wrapper,
        )
        
        dtype = get_optimal_mlx_dtype()
        wrapped_model = create_checkpointed_model_wrapper(simple_mlx_model)
        
        x = mx.array([[1, 2, 3, 4, 5]])
        output = wrapped_model(x)
        mx.eval(output)
        
        assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
