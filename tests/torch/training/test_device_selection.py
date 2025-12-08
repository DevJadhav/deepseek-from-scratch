"""
Device Selection Tests - TDD Approach

Tests for unified device selection with CUDA → MPS → CPU priority chain
and binary search batch size optimization.

These tests pass on all systems regardless of hardware availability
by using mocking for unavailable devices.

Run with: uv run pytest tests/torch/training/test_device_selection.py -v
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing batch size."""
    class SimpleModel(torch.nn.Module):
        def __init__(self, hidden_size: int = 64):
            super().__init__()
            self.embedding = torch.nn.Embedding(1000, hidden_size)
            self.linear = torch.nn.Linear(hidden_size, hidden_size)
            self.output = torch.nn.Linear(hidden_size, 1000)
            self.vocab_size = 1000
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)
            x = torch.relu(self.linear(x))
            return self.output(x)
    
    return SimpleModel()


# =============================================================================
# Device Selection Tests
# =============================================================================

class TestDeviceSelection:
    """Tests for get_device() function."""
    
    def test_get_device_returns_valid_device(self):
        """Test that get_device always returns a valid torch device."""
        from deepseek.torch.utils.device import get_device
        
        device = get_device()
        assert isinstance(device, torch.device)
        # Should be able to create a tensor on the device
        tensor = torch.zeros(2, 2, device=device)
        assert tensor.device.type == device.type
    
    def test_get_device_priority_cuda_first(self):
        """Test that CUDA is prioritized when available."""
        from deepseek.torch.utils.device import get_device, DevicePriority
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device(priority=DevicePriority.CUDA_FIRST)
                assert device.type == "cuda"
    
    def test_get_device_fallback_to_mps(self):
        """Test fallback to MPS when CUDA unavailable."""
        from deepseek.torch.utils.device import get_device, DevicePriority
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device(priority=DevicePriority.CUDA_FIRST)
                assert device.type == "mps"
    
    def test_get_device_fallback_to_cpu(self):
        """Test fallback to CPU when no GPU available."""
        from deepseek.torch.utils.device import get_device, DevicePriority
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = get_device(priority=DevicePriority.CUDA_FIRST)
                assert device.type == "cpu"
    
    def test_get_device_info(self):
        """Test get_device_info returns required keys."""
        from deepseek.torch.utils.device import get_device_info
        
        info = get_device_info()
        assert "cuda_available" in info
        assert "mps_available" in info
        assert "selected_device" in info
    
    def test_device_type_string(self):
        """Test device_type_string returns correct strings."""
        from deepseek.torch.utils.device import device_type_string
        
        assert device_type_string(torch.device("cpu")) == "cpu"
        
        # Test with mock CUDA device
        with patch('torch.cuda.is_available', return_value=True):
            cuda_device = torch.device("cuda:0")
            assert device_type_string(cuda_device) == "cuda"


class TestDeviceConfig:
    """Tests for DeviceConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from deepseek.torch.utils.device import DeviceConfig, DevicePriority
        
        config = DeviceConfig()
        assert config.priority == DevicePriority.CUDA_FIRST
        assert config.cuda_device_id == 0
        assert config.min_batch_size == 1
        assert config.max_batch_size == 256
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        from deepseek.torch.utils.device import DeviceConfig
        import os
        
        with patch.dict(os.environ, {
            "DEEPSEEK_DEVICE_PRIORITY": "cpu",
            "DEEPSEEK_MIN_BATCH_SIZE": "4",
            "DEEPSEEK_MAX_BATCH_SIZE": "128",
        }):
            config = DeviceConfig.from_env()
            assert config.min_batch_size == 4
            assert config.max_batch_size == 128
    
    def test_get_priority_order(self):
        """Test priority order list."""
        from deepseek.torch.utils.device import DeviceConfig, DevicePriority
        
        config = DeviceConfig(priority=DevicePriority.CUDA_FIRST)
        assert config.get_priority_order() == ["cuda", "mps", "cpu"]
        
        config = DeviceConfig(priority=DevicePriority.CPU_ONLY)
        assert config.get_priority_order() == ["cpu"]


# =============================================================================
# Binary Search Batch Size Tests
# =============================================================================

class TestAutoBatchSizer:
    """Tests for AutoBatchSizer with binary search."""
    
    def test_binary_search_basic(self):
        """Test binary search finds valid batch size."""
        from deepseek.torch.utils.device import AutoBatchSizer, AutoBatchConfig
        
        config = AutoBatchConfig(
            max_memory_mb=1024,  # 1 GB
            memory_fraction=0.9,
            min_batch_size=1,
            max_batch_size=256,
        )
        sizer = AutoBatchSizer(config)
        
        # 10 MB per sample should fit ~90 samples in 900 MB budget
        memory_per_sample_mb = 10
        optimal = sizer.find_optimal_batch_size_binary(memory_per_sample_mb)
        
        assert optimal >= config.min_batch_size
        assert optimal <= config.max_batch_size
        assert optimal * memory_per_sample_mb <= config.max_memory_mb * config.memory_fraction
    
    def test_binary_search_small_memory(self):
        """Test binary search with small memory budget."""
        from deepseek.torch.utils.device import AutoBatchSizer, AutoBatchConfig
        
        config = AutoBatchConfig(
            max_memory_mb=100,  # 100 MB
            memory_fraction=0.9,
            min_batch_size=1,
            max_batch_size=256,
        )
        sizer = AutoBatchSizer(config)
        
        # 50 MB per sample should result in batch size 1
        memory_per_sample_mb = 50
        optimal = sizer.find_optimal_batch_size_binary(memory_per_sample_mb)
        
        assert optimal == 1
    
    def test_binary_search_large_memory(self):
        """Test binary search with large memory budget caps at max."""
        from deepseek.torch.utils.device import AutoBatchSizer, AutoBatchConfig
        
        config = AutoBatchConfig(
            max_memory_mb=10240,  # 10 GB
            memory_fraction=0.9,
            min_batch_size=1,
            max_batch_size=256,
        )
        sizer = AutoBatchSizer(config)
        
        # 1 MB per sample should result in max batch size
        memory_per_sample_mb = 1
        optimal = sizer.find_optimal_batch_size_binary(memory_per_sample_mb)
        
        assert optimal == 256
    
    def test_binary_search_respects_bounds(self):
        """Test binary search respects custom min/max bounds."""
        from deepseek.torch.utils.device import AutoBatchSizer, AutoBatchConfig
        
        config = AutoBatchConfig(
            max_memory_mb=1024,
            memory_fraction=0.9,
            min_batch_size=8,  # Custom minimum
            max_batch_size=64,  # Custom maximum
        )
        sizer = AutoBatchSizer(config)
        
        memory_per_sample_mb = 5
        optimal = sizer.find_optimal_batch_size_binary(memory_per_sample_mb)
        
        assert optimal >= 8
        assert optimal <= 64
    
    def test_binary_search_disabled(self):
        """Test behavior when auto-adjust is disabled."""
        from deepseek.torch.utils.device import AutoBatchSizer, AutoBatchConfig
        
        config = AutoBatchConfig(
            max_memory_mb=1024,
            memory_fraction=0.9,
            min_batch_size=1,
            max_batch_size=256,
            auto_adjust=False,  # Disabled
        )
        sizer = AutoBatchSizer(config)
        
        default_batch = 32
        optimal = sizer.find_optimal_batch_size_binary_with_default(10, default_batch)
        
        assert optimal == default_batch


# =============================================================================
# Integration Tests
# =============================================================================

class TestDeviceIntegration:
    """Integration tests for device selection with models."""
    
    def test_device_with_tensor_operations(self):
        """Test tensor operations work on selected device."""
        from deepseek.torch.utils.device import get_device
        
        device = get_device()
        
        a = torch.randn(4, 4, device=device)
        b = torch.randn(4, 4, device=device)
        c = torch.matmul(a, b)
        
        assert c.shape == (4, 4)
        assert c.device.type == device.type
    
    def test_model_to_device(self, simple_model):
        """Test model can be moved to selected device."""
        from deepseek.torch.utils.device import get_device
        
        device = get_device()
        model = simple_model.to(device)
        
        # Verify model parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type
    
    def test_auto_batch_with_model(self, simple_model):
        """Test auto batch sizing with actual model (CPU only for CI)."""
        from deepseek.torch.utils.device import AutoBatchSizer, AutoBatchConfig
        
        # Use conservative settings for CI
        config = AutoBatchConfig(
            max_memory_mb=512,
            memory_fraction=0.8,
            min_batch_size=1,
            max_batch_size=32,
        )
        sizer = AutoBatchSizer(config)
        
        # Estimate ~1 MB per sample (rough estimate for small model)
        memory_per_sample_mb = 1
        optimal = sizer.find_optimal_batch_size_binary(memory_per_sample_mb)
        
        # Should find a reasonable batch size
        assert optimal >= 1
        assert optimal <= 32


# =============================================================================
# Environment Variable Configuration Tests
# =============================================================================

class TestEnvironmentConfig:
    """Tests for environment variable configuration."""
    
    def test_cuda_device_id_from_env(self):
        """Test CUDA device ID from environment."""
        from deepseek.torch.utils.device import DeviceConfig
        import os
        
        with patch.dict(os.environ, {"DEEPSEEK_CUDA_DEVICE": "2"}):
            config = DeviceConfig.from_env()
            assert config.cuda_device_id == 2
    
    def test_invalid_env_uses_defaults(self):
        """Test invalid environment values use defaults."""
        from deepseek.torch.utils.device import DeviceConfig
        import os
        
        with patch.dict(os.environ, {
            "DEEPSEEK_MIN_BATCH_SIZE": "invalid",
            "DEEPSEEK_MAX_BATCH_SIZE": "also_invalid",
        }):
            config = DeviceConfig.from_env()
            assert config.min_batch_size == 1  # Default
            assert config.max_batch_size == 256  # Default
    
    def test_device_priority_from_env(self):
        """Test device priority from environment."""
        from deepseek.torch.utils.device import DeviceConfig, DevicePriority
        import os
        
        with patch.dict(os.environ, {"DEEPSEEK_DEVICE_PRIORITY": "mps"}):
            config = DeviceConfig.from_env()
            assert config.priority == DevicePriority.MPS_FIRST
        
        with patch.dict(os.environ, {"DEEPSEEK_DEVICE_PRIORITY": "cpu"}):
            config = DeviceConfig.from_env()
            assert config.priority == DevicePriority.CPU_ONLY
