"""
GPU-specific pipeline tests for Ray orchestration.

These tests verify GPU backends (CUDA, Metal, MLX) work correctly.
They are automatically skipped on systems without the required hardware.

Run manually with:
    pytest tests/pipeline/test_gpu_pipeline.py -v
    pytest tests/pipeline/test_gpu_pipeline.py -v -k "cuda" --gpu
    pytest tests/pipeline/test_gpu_pipeline.py -v -k "metal" --gpu
    pytest tests/pipeline/test_gpu_pipeline.py -v -k "mlx" --gpu
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deepseek.pipeline.config import (
    DataConfig,
    ExportConfig,
    ModelConfig,
    PipelineConfig,
    TimeSlicedConfig,
    TrainingConfig,
    WaveBackend,
    WaveConfig,
)


# ============================================================================
# Hardware detection utilities
# ============================================================================

def has_cuda() -> bool:
    """Check if NVIDIA CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_mps() -> bool:
    """Check if Apple MPS (Metal Performance Shaders) is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def has_metal() -> bool:
    """Check if Apple Metal is available (macOS)."""
    return platform.system() == "Darwin"


def has_mlx() -> bool:
    """Check if MLX is available (Apple Silicon)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except ImportError:
        return False


def has_rust_backend() -> bool:
    """Check if Rust deepseek_rs backend is available."""
    return importlib.util.find_spec("deepseek_rs") is not None


# ============================================================================
# Pytest markers for GPU tests
# ============================================================================

requires_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason="NVIDIA CUDA not available"
)

requires_mps = pytest.mark.skipif(
    not has_mps(),
    reason="Apple MPS not available"
)

requires_metal = pytest.mark.skipif(
    not has_metal(),
    reason="Apple Metal not available (requires macOS)"
)

requires_mlx = pytest.mark.skipif(
    not has_mlx(),
    reason="MLX not available (requires Apple Silicon)"
)

requires_rust = pytest.mark.skipif(
    not has_rust_backend(),
    reason="Rust backend (deepseek_rs) not available"
)

# Combined markers
requires_any_gpu = pytest.mark.skipif(
    not (has_cuda() or has_mps() or has_mlx()),
    reason="No GPU backend available (CUDA, MPS, or MLX)"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_checkpoint_dir(temp_dir):
    """Create a mock checkpoint directory with valid structure."""
    ckpt_dir = temp_dir / "checkpoint"
    ckpt_dir.mkdir()

    (ckpt_dir / "model.pt").write_bytes(b"mock model weights")
    (ckpt_dir / "optimizer.pt").write_bytes(b"mock optimizer state")
    (ckpt_dir / "config.json").write_text(json.dumps({
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 256,
    }))
    (ckpt_dir / "training_state.json").write_text(json.dumps({
        "step": 100,
        "best_loss": 3.5,
        "last_loss": 3.6,
    }))

    return ckpt_dir


@pytest.fixture
def mock_data_dir(temp_dir):
    """Create mock training data."""
    data_dir = temp_dir / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "valid").mkdir(parents=True)

    # Create JSONL data files
    for split in ["train", "valid"]:
        data_file = data_dir / split / f"{split}.jsonl"
        with open(data_file, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"Sample text {i} for {split}. " * 5}) + "\n")

    return data_dir


@pytest.fixture
def tiny_model_config():
    """Create a tiny model config for GPU tests."""
    return ModelConfig(
        d_model=64,
        num_layers=2,
        num_heads=2,
        vocab_size=1000,
        max_seq_len=128,
    )


def make_pipeline_config(
    temp_dir: Path,
    data_dir: Path,
    model_config: ModelConfig,
    backend: WaveBackend,
    max_steps: int = 50,
) -> PipelineConfig:
    """Create a pipeline config for a specific backend."""
    steps_per_wave = max_steps // 2

    return PipelineConfig(
        run_name=f"test-{backend.value}",
        model=model_config,
        training=TrainingConfig(
            max_steps=max_steps,
            batch_size=2,
            gradient_accumulation_steps=2,
        ),
        data=DataConfig(
            data_dir=str(data_dir),
        ),
        export=ExportConfig(
            output_dir=str(temp_dir / "export"),
        ),
        time_sliced=TimeSlicedConfig(
            enabled=True,
            num_waves=2,
            gpu_ids=[0],
            pipeline_parallel_size=1,
            waves=[
                WaveConfig(
                    wave_id=1,
                    backend=backend,
                    start_step=0,
                    end_step=steps_per_wave,
                    stages=["pretrain"],
                ),
                WaveConfig(
                    wave_id=2,
                    backend=backend,
                    start_step=steps_per_wave,
                    end_step=max_steps,
                    stages=["pretrain"],
                ),
            ],
        ),
    )


# ============================================================================
# GPU Backend Tests
# ============================================================================

class TestGPUBackendConfig:
    """Test GPU backend configuration."""

    def test_wavebackend_enum_has_all_gpu_backends(self):
        """Verify all GPU backend variants exist."""
        assert hasattr(WaveBackend, "PYTORCH_CUDA")
        assert hasattr(WaveBackend, "PYTORCH_MPS")
        assert hasattr(WaveBackend, "MLX")
        assert hasattr(WaveBackend, "RUST_CUDA")
        assert hasattr(WaveBackend, "RUST_METAL")
        assert hasattr(WaveBackend, "CPU_ONLY")

    def test_is_gpu_backend_cuda(self):
        """Test CUDA is recognized as GPU backend."""
        assert WaveBackend.PYTORCH_CUDA.is_gpu_backend()
        assert WaveBackend.RUST_CUDA.is_gpu_backend()

    def test_is_gpu_backend_metal(self):
        """Test Metal backends are recognized as GPU backends."""
        assert WaveBackend.PYTORCH_MPS.is_gpu_backend()
        assert WaveBackend.RUST_METAL.is_gpu_backend()

    def test_is_gpu_backend_mlx(self):
        """Test MLX is recognized as GPU backend."""
        assert WaveBackend.MLX.is_gpu_backend()

    def test_cpu_only_is_not_gpu_backend(self):
        """Test CPU-only is not a GPU backend."""
        assert not WaveBackend.CPU_ONLY.is_gpu_backend()
        assert not WaveBackend.CPU.is_gpu_backend()


@requires_cuda
class TestCUDABackend:
    """Tests requiring NVIDIA CUDA."""

    def test_cuda_workflow_init(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test workflow can be initialized with CUDA backend."""
        from deepseek.pipeline.workflow import DeepSeekWorkflow

        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.PYTORCH_CUDA
        )

        with patch("ray.init"):
            workflow = DeepSeekWorkflow(config)
            assert workflow is not None
            assert workflow.config.time_sliced.waves[0].backend == WaveBackend.PYTORCH_CUDA

    def test_cuda_wave_config_validation(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test CUDA wave configuration is valid."""
        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.PYTORCH_CUDA
        )

        for wave in config.time_sliced.waves:
            assert wave.backend == WaveBackend.PYTORCH_CUDA
            assert wave.backend.is_gpu_backend()


@requires_rust
@requires_cuda
class TestRustCUDABackend:
    """Tests requiring Rust backend with CUDA."""

    def test_rust_cuda_workflow_init(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test workflow can be initialized with Rust+CUDA backend."""
        from deepseek.pipeline.workflow import DeepSeekWorkflow

        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.RUST_CUDA
        )

        with patch("ray.init"):
            workflow = DeepSeekWorkflow(config)
            assert workflow is not None
            assert workflow.config.time_sliced.waves[0].backend == WaveBackend.RUST_CUDA

    def test_rust_cuda_is_rust_backend(self):
        """Test RUST_CUDA is recognized as Rust backend."""
        assert WaveBackend.RUST_CUDA.is_rust_backend()
        assert WaveBackend.RUST_CUDA.is_gpu_backend()


@requires_mps
class TestMPSBackend:
    """Tests requiring Apple MPS (Metal Performance Shaders)."""

    def test_mps_workflow_init(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test workflow can be initialized with MPS backend."""
        from deepseek.pipeline.workflow import DeepSeekWorkflow

        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.PYTORCH_MPS
        )

        with patch("ray.init"):
            workflow = DeepSeekWorkflow(config)
            assert workflow is not None
            assert workflow.config.time_sliced.waves[0].backend == WaveBackend.PYTORCH_MPS


@requires_rust
@requires_metal
class TestRustMetalBackend:
    """Tests requiring Rust backend with Metal."""

    def test_rust_metal_workflow_init(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test workflow can be initialized with Rust+Metal backend."""
        from deepseek.pipeline.workflow import DeepSeekWorkflow

        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.RUST_METAL
        )

        with patch("ray.init"):
            workflow = DeepSeekWorkflow(config)
            assert workflow is not None
            assert workflow.config.time_sliced.waves[0].backend == WaveBackend.RUST_METAL

    def test_rust_metal_is_rust_backend(self):
        """Test RUST_METAL is recognized as Rust backend."""
        assert WaveBackend.RUST_METAL.is_rust_backend()
        assert WaveBackend.RUST_METAL.is_gpu_backend()


@requires_mlx
class TestMLXBackend:
    """Tests requiring Apple MLX (Apple Silicon)."""

    def test_mlx_workflow_init(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test workflow can be initialized with MLX backend."""
        from deepseek.pipeline.workflow import DeepSeekWorkflow

        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.MLX
        )

        with patch("ray.init"):
            workflow = DeepSeekWorkflow(config)
            assert workflow is not None
            assert workflow.config.time_sliced.waves[0].backend == WaveBackend.MLX

    def test_mlx_import_works(self):
        """Test MLX can be imported."""
        import mlx.core as mx
        assert mx.metal.is_available()

    def test_mlx_basic_ops(self):
        """Test basic MLX operations work."""
        import mlx.core as mx

        # Simple matmul
        a = mx.ones((4, 4))
        b = mx.ones((4, 4))
        c = mx.matmul(a, b)
        mx.eval(c)

        assert c.shape == (4, 4)
        assert float(c[0, 0]) == 4.0


class TestCPUOnlyBackend:
    """Tests for CPU-only backend (no GPU required)."""

    def test_cpu_only_workflow_init(self, temp_dir, mock_data_dir, tiny_model_config):
        """Test workflow can be initialized with CPU-only backend."""
        from deepseek.pipeline.workflow import DeepSeekWorkflow

        config = make_pipeline_config(
            temp_dir, mock_data_dir, tiny_model_config,
            backend=WaveBackend.CPU_ONLY
        )

        with patch("ray.init"):
            workflow = DeepSeekWorkflow(config)
            assert workflow is not None
            assert workflow.config.time_sliced.waves[0].backend == WaveBackend.CPU_ONLY

    def test_cpu_only_is_not_gpu(self):
        """Test CPU_ONLY is not recognized as GPU backend."""
        assert not WaveBackend.CPU_ONLY.is_gpu_backend()


# ============================================================================
# Integration Tests (require actual hardware)
# ============================================================================

@pytest.mark.slow
class TestGPUIntegration:
    """Integration tests that run actual GPU operations."""

    @requires_any_gpu
    def test_detect_available_gpu_backend(self):
        """Test that at least one GPU backend is detected."""
        available_backends = []

        if has_cuda():
            available_backends.append(WaveBackend.PYTORCH_CUDA)
        if has_mps():
            available_backends.append(WaveBackend.PYTORCH_MPS)
        if has_mlx():
            available_backends.append(WaveBackend.MLX)

        assert len(available_backends) > 0
        for backend in available_backends:
            assert backend.is_gpu_backend()

    @requires_mlx
    def test_mlx_model_creation(self):
        """Test MLX model can be created and runs forward pass."""
        import mlx.core as mx
        import mlx.nn as nn

        # Simple transformer-like layer
        class TinyBlock(nn.Module):
            def __init__(self, d_model=64):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.linear = nn.Linear(d_model, d_model)

            def __call__(self, x):
                return self.linear(self.norm(x))

        model = TinyBlock(64)
        x = mx.random.normal((2, 8, 64))
        y = model(x)
        mx.eval(y)

        assert y.shape == (2, 8, 64)

    @requires_cuda
    def test_cuda_torch_tensor_ops(self):
        """Test PyTorch CUDA tensor operations."""
        import torch

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")
        z = torch.matmul(x, y)

        assert z.device.type == "cuda"
        assert z.shape == (4, 4)

    @requires_mps
    def test_mps_torch_tensor_ops(self):
        """Test PyTorch MPS tensor operations."""
        import torch

        x = torch.randn(4, 4, device="mps")
        y = torch.randn(4, 4, device="mps")
        z = torch.matmul(x, y)

        assert z.device.type == "mps"
        assert z.shape == (4, 4)


# ============================================================================
# Benchmark helper tests
# ============================================================================

class TestBenchmarkHelpers:
    """Test benchmark helper functions."""

    def test_hardware_detection_functions_dont_raise(self):
        """Test all hardware detection functions run without errors."""
        # These should never raise, just return bool
        assert isinstance(has_cuda(), bool)
        assert isinstance(has_mps(), bool)
        assert isinstance(has_metal(), bool)
        assert isinstance(has_mlx(), bool)
        assert isinstance(has_rust_backend(), bool)

    def test_at_least_cpu_available(self):
        """Test that CPU backend is always available."""
        # CPU should always work
        assert hasattr(WaveBackend, "CPU")
        assert hasattr(WaveBackend, "CPU_ONLY")
        assert hasattr(WaveBackend, "PYTORCH_CPU")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
