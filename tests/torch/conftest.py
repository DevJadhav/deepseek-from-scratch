"""
Pytest configuration for DeepSeek tests.

Provides shared fixtures for:
- Device management (CPU, CUDA, MPS)
- Model configurations (tiny, small, test)
- Sample data generation
- Distributed testing utilities
- Benchmark fixtures

Usage:
    # Use fixtures in tests
    def test_something(device, tiny_model_config):
        model = create_model(tiny_model_config)
        model.to(device)
"""

import sys
from pathlib import Path
from typing import Any

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
import torch


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Provide a torch device (CPU for CI, CUDA if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Provide CUDA device, skip if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def mps_device():
    """Provide MPS device for Apple Silicon, skip if unavailable."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    return torch.device("mps")


@pytest.fixture
def cpu_device():
    """Always provide CPU device."""
    return torch.device("cpu")


# =============================================================================
# Availability Fixtures
# =============================================================================

@pytest.fixture
def mlx_available():
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


@pytest.fixture
def triton_available():
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


@pytest.fixture
def flash_attn_available():
    """Check if Flash Attention is available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


@pytest.fixture
def distributed_available():
    """Check if distributed training is available."""
    return torch.distributed.is_available()


# =============================================================================
# Model Configuration Fixtures
# =============================================================================

@pytest.fixture
def tiny_model_config() -> dict[str, Any]:
    """Tiny model config for fast unit tests."""
    return {
        "vocab_size": 1000,
        "num_layers": 2,
        "d_model": 128,
        "num_heads": 4,
        "d_latent": 32,
        "d_rope": 16,
        "d_hidden": 256,
        "use_moe": False,
        "dropout": 0.0,
    }


@pytest.fixture
def small_model_config() -> dict[str, Any]:
    """Small model config for integration tests."""
    return {
        "vocab_size": 8000,
        "num_layers": 4,
        "d_model": 256,
        "num_heads": 4,
        "d_latent": 64,
        "d_rope": 32,
        "d_hidden": 512,
        "use_moe": True,
        "num_experts": 4,
        "num_shared_experts": 1,
        "top_k": 2,
        "dropout": 0.0,
    }


@pytest.fixture
def moe_model_config() -> dict[str, Any]:
    """MoE-specific model config for MoE tests."""
    return {
        "vocab_size": 1000,
        "num_layers": 2,
        "d_model": 128,
        "num_heads": 4,
        "d_latent": 32,
        "d_rope": 16,
        "d_hidden": 256,
        "use_moe": True,
        "num_experts": 8,
        "num_shared_experts": 1,
        "top_k": 2,
        "expert_intermediate_dim": 64,
        "use_auxiliary_loss": False,
        "dropout": 0.0,
    }


@pytest.fixture
def mla_model_config() -> dict[str, Any]:
    """MLA-specific model config for MLA tests."""
    return {
        "vocab_size": 1000,
        "num_layers": 2,
        "d_model": 128,
        "num_heads": 4,
        "d_latent": 32,
        "q_latent": 64,
        "d_rope": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "d_hidden": 256,
        "decoupled_rope": True,
        "dropout": 0.0,
    }


@pytest.fixture
def mtp_model_config() -> dict[str, Any]:
    """MTP-specific model config for MTP tests."""
    return {
        "vocab_size": 1000,
        "num_layers": 2,
        "d_model": 128,
        "num_heads": 4,
        "d_latent": 32,
        "d_rope": 16,
        "d_hidden": 256,
        "mtp_enabled": True,
        "mtp_depth": 2,
        "mtp_loss_weight": 0.5,
        "dropout": 0.0,
    }


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_batch(device):
    """Generate sample batch for testing."""
    batch_size = 2
    seq_len = 64
    vocab_size = 1000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@pytest.fixture
def sample_text_data():
    """Sample text data for tokenization tests."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "DeepSeek is an AI research company.",
        "Multi-Latent Attention reduces KV cache memory usage.",
        "Mixture of Experts enables efficient scaling.",
    ]


# =============================================================================
# Training Fixtures
# =============================================================================

@pytest.fixture
def training_config() -> dict[str, Any]:
    """Basic training configuration for tests."""
    return {
        "learning_rate": 1e-4,
        "batch_size": 2,
        "max_steps": 10,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 2,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }


@pytest.fixture
def optimizer_factory():
    """Factory to create optimizers for tests."""
    def create_optimizer(model, lr=1e-4, weight_decay=0.01):
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
    return create_optimizer


# =============================================================================
# Distributed Testing Fixtures
# =============================================================================

@pytest.fixture
def world_size():
    """Default world size for distributed tests."""
    return min(2, torch.cuda.device_count() if torch.cuda.is_available() else 1)


@pytest.fixture
def distributed_config(world_size):
    """Configuration for distributed tests."""
    return {
        "world_size": world_size,
        "backend": "nccl" if torch.cuda.is_available() else "gloo",
        "init_method": "env://",
    }


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_iterations": 3,
        "benchmark_iterations": 10,
        "batch_sizes": [1, 2, 4],
        "seq_lens": [128, 256, 512],
    }


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (use -m 'not slow' to skip)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "distributed: mark test as requiring distributed setup"
    )
    config.addinivalue_line(
        "markers", "integration: mark as integration test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark as benchmark test"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on environment."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_distributed = pytest.mark.skip(reason="Distributed not available or insufficient GPUs")
    
    for item in items:
        if "gpu" in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(skip_gpu)
                
        if "distributed" in item.keywords:
            if not torch.distributed.is_available():
                item.add_marker(skip_distributed)
            elif torch.cuda.is_available() and torch.cuda.device_count() < 2:
                item.add_marker(skip_distributed)


# =============================================================================
# Utility Functions
# =============================================================================

def assert_tensor_close(actual, expected, rtol=1e-5, atol=1e-5):
    """Assert two tensors are close."""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"Tensors not close: max diff = {(actual - expected).abs().max()}"


def assert_no_nan(tensor, name="tensor"):
    """Assert tensor has no NaN values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"


def assert_no_inf(tensor, name="tensor"):
    """Assert tensor has no Inf values."""
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
