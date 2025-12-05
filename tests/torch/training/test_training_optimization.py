"""
Training Optimization Tests - Python GPU Backend

Tests for:
- torch.compile integration
- Mixed precision training
- Memory profiling and optimization
- Gradient accumulation
- NaN/Inf validation
- CPU offloading
- Precision-specific weight initialization
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch


# =============================================================================
# Test Fixtures
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing optimization utilities."""
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 1000)
        self.vocab_size = 1000
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Compilation Tests (Section 1.2)
# =============================================================================

class TestCompilation:
    """Tests for torch.compile integration."""
    
    def test_compile_mode_enum(self):
        """Test CompileMode enum values."""
        from deepseek.torch.training.optimization import CompileMode
        
        assert CompileMode.REDUCE_OVERHEAD.value == "reduce-overhead"
        assert CompileMode.MAX_AUTOTUNE.value == "max-autotune"
        assert CompileMode.DEFAULT.value == "default"
        assert CompileMode.DISABLED.value == "disabled"
    
    def test_compile_config_defaults(self):
        """Test CompileConfig default values."""
        from deepseek.torch.training.optimization import CompileConfig, CompileMode
        
        config = CompileConfig()
        assert config.mode == CompileMode.REDUCE_OVERHEAD
        assert config.fullgraph is False
        assert config.dynamic is True
    
    def test_compile_config_custom(self):
        """Test CompileConfig with custom values."""
        from deepseek.torch.training.optimization import CompileConfig, CompileMode
        
        config = CompileConfig(
            mode=CompileMode.MAX_AUTOTUNE,
            fullgraph=True,
            dynamic=False,
        )
        assert config.mode == CompileMode.MAX_AUTOTUNE
        assert config.fullgraph is True
        assert config.dynamic is False
    
    def test_compile_model_disabled(self, simple_model):
        """Test that disabled compilation returns original model."""
        from deepseek.torch.training.optimization import compile_model, CompileConfig, CompileMode
        
        config = CompileConfig(mode=CompileMode.DISABLED)
        result = compile_model(simple_model, config)
        
        # Should return the same model object when disabled
        assert result is simple_model
    
    def test_compile_model_with_config(self, simple_model):
        """Test model compilation with config."""
        from deepseek.torch.training.optimization import compile_model, CompileConfig, CompileMode
        
        config = CompileConfig(mode=CompileMode.DEFAULT)
        result = compile_model(simple_model, config)
        
        # Should return a compiled model (or same model on older PyTorch)
        assert result is not None
        
        # Should still be callable
        x = torch.randint(0, 1000, (2, 10))
        output = result(x)
        assert output.shape == (2, 10, 1000)
    
    def test_compile_warmup_wrapper(self, simple_model):
        """Test compilation warmup wrapper."""
        from deepseek.torch.training.optimization import create_compile_warmup_wrapper, CompileConfig
        
        config = CompileConfig()
        model, trigger_compile = create_compile_warmup_wrapper(simple_model, config)
        
        # Should be callable
        x = torch.randint(0, 1000, (2, 10))
        
        # First call (warmup)
        output1 = model(x)
        assert output1.shape == (2, 10, 1000)
        
        # Trigger compilation
        trigger_compile()
        
        # After compilation still works
        output2 = model(x)
        assert output2.shape == (2, 10, 1000)


# =============================================================================
# Mixed Precision Tests (Section 1.3)
# =============================================================================

class TestMixedPrecision:
    """Tests for mixed precision training."""
    
    def test_precision_mode_enum(self):
        """Test PrecisionMode enum values."""
        from deepseek.torch.training.optimization import PrecisionMode
        
        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.FP16.value == "fp16"
        assert PrecisionMode.BF16.value == "bf16"
    
    def test_mixed_precision_config_defaults(self):
        """Test MixedPrecisionConfig defaults."""
        from deepseek.torch.training.optimization import MixedPrecisionConfig, PrecisionMode
        
        config = MixedPrecisionConfig()
        assert config.mode == PrecisionMode.AUTO
        assert config.compute_loss_in_fp32 is True
        assert config.optimizer_states_in_fp32 is True
    
    def test_supports_bfloat16(self):
        """Test BF16 support detection."""
        from deepseek.torch.training.optimization import supports_bfloat16
        
        result = supports_bfloat16()
        assert isinstance(result, bool)
    
    def test_supports_fp16(self):
        """Test FP16 support detection."""
        from deepseek.torch.training.optimization import supports_fp16
        
        result = supports_fp16()
        assert isinstance(result, bool)
    
    def test_get_optimal_precision(self):
        """Test automatic precision selection."""
        from deepseek.torch.training.optimization import get_optimal_precision, PrecisionMode
        
        precision = get_optimal_precision()
        assert precision in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16]
    
    def test_mixed_precision_trainer_init(self, simple_model, device):
        """Test MixedPrecisionTrainer initialization."""
        from deepseek.torch.training.optimization import MixedPrecisionTrainer, MixedPrecisionConfig
        
        config = MixedPrecisionConfig()
        trainer = MixedPrecisionTrainer(config)
        
        assert trainer.config == config
    
    def test_mixed_precision_trainer_forward(self, simple_model, device):
        """Test MixedPrecisionTrainer autocast context."""
        from deepseek.torch.training.optimization import MixedPrecisionTrainer, MixedPrecisionConfig, PrecisionMode
        
        config = MixedPrecisionConfig(mode=PrecisionMode.FP32)  # Use FP32 for CPU
        simple_model = simple_model.to(device)
        trainer = MixedPrecisionTrainer(config)
        
        x = torch.randint(0, 1000, (2, 10), device=device)
        
        with trainer.autocast_context(device_type="cpu"):
            output = simple_model(x)
        
        assert output.shape == (2, 10, 1000)


# =============================================================================
# Memory Profiling Tests (Section 1.5)
# =============================================================================

class TestMemoryProfiling:
    """Tests for memory profiling utilities."""
    
    def test_memory_stats_dataclass(self):
        """Test MemoryStats dataclass."""
        from deepseek.torch.training.optimization import MemoryStats
        
        stats = MemoryStats(
            allocated_mb=100.0,
            reserved_mb=200.0,
            peak_allocated_mb=150.0,
            peak_reserved_mb=250.0,
            active_blocks=10,
            inactive_split_blocks=5,
        )
        
        assert stats.allocated_mb == 100.0
        assert stats.reserved_mb == 200.0
        assert stats.active_blocks == 10
        assert stats.peak_allocated_mb == 150.0
        assert stats.peak_reserved_mb == 250.0
    
    def test_memory_profiler_init(self):
        """Test MemoryProfiler initialization."""
        from deepseek.torch.training.optimization import MemoryProfiler
        
        profiler = MemoryProfiler(log_interval=10)
        assert profiler.log_interval == 10
    
    def test_memory_profiler_get_stats(self):
        """Test MemoryProfiler.get_memory_stats()."""
        from deepseek.torch.training.optimization import MemoryProfiler
        
        profiler = MemoryProfiler()
        stats = profiler.get_memory_stats()
        
        # On CPU, stats will be None; on CUDA it will be MemoryStats
        if torch.cuda.is_available():
            assert stats is not None
            assert stats.allocated_mb >= 0
        else:
            assert stats is None
    
    def test_memory_profiler_context_manager(self, simple_model):
        """Test MemoryProfiler profile_region context manager."""
        from deepseek.torch.training.optimization import MemoryProfiler
        
        profiler = MemoryProfiler()
        
        # Test profile_region context manager
        with profiler.profile_region("test_region"):
            x = torch.randn(100, 100)
            y = x @ x.T
        
        # Should complete without error
        assert True
    
    def test_calculate_mbu(self, simple_model, device):
        """Test Model Bandwidth Utilization calculation."""
        from deepseek.torch.training.optimization import calculate_mbu
        
        simple_model = simple_model.to(device)
        
        # Test with known values - the function takes model, batch_size, seq_len, time_seconds, device
        mbu = calculate_mbu(
            model=simple_model,
            batch_size=4,
            seq_len=16,
            time_seconds=1.0,
            device=device,
        )
        
        assert isinstance(mbu, float)
        assert mbu >= 0  # MBU should be non-negative
    
    def test_create_pytorch_profiler(self):
        """Test PyTorch profiler creation."""
        from deepseek.torch.training.optimization import create_pytorch_profiler
        
        profiler = create_pytorch_profiler()
        
        assert profiler is not None


# =============================================================================
# NaN/Inf Validation Tests (Section 1.3)
# =============================================================================

class TestNaNInfValidation:
    """Tests for NaN/Inf validation."""
    
    def test_nan_check_result(self):
        """Test NaNCheckResult dataclass."""
        from deepseek.torch.training.optimization import NaNCheckResult
        
        result = NaNCheckResult(
            has_nan=True,
            has_inf=False,
            nan_params=["layer1.weight"],
            inf_params=[],
            nan_grads=["layer1.weight"],
            inf_grads=[],
        )
        
        assert result.has_nan is True
        assert result.has_inf is False
        assert "layer1.weight" in result.nan_params
    
    def test_check_nan_inf_clean_model(self, simple_model):
        """Test check_nan_inf with clean model."""
        from deepseek.torch.training.optimization import check_nan_inf
        
        result = check_nan_inf(simple_model, check_gradients=False)
        
        assert result.has_nan is False
        assert result.has_inf is False
        assert len(result.nan_params) == 0
        assert len(result.inf_params) == 0
    
    def test_check_nan_inf_with_nan(self, simple_model):
        """Test check_nan_inf with NaN values."""
        from deepseek.torch.training.optimization import check_nan_inf
        
        # Inject NaN
        with torch.no_grad():
            simple_model.layers[0].weight[0, 0] = float('nan')
        
        result = check_nan_inf(simple_model, check_gradients=False)
        
        assert result.has_nan is True
        assert len(result.nan_params) > 0
    
    def test_check_nan_inf_with_inf(self, simple_model):
        """Test check_nan_inf with Inf values."""
        from deepseek.torch.training.optimization import check_nan_inf
        
        # Inject Inf
        with torch.no_grad():
            simple_model.layers[0].weight[0, 0] = float('inf')
        
        result = check_nan_inf(simple_model, check_gradients=False)
        
        assert result.has_inf is True
        assert len(result.inf_params) > 0
    
    def test_nan_inf_validator_init(self):
        """Test NaNInfValidator initialization."""
        from deepseek.torch.training.optimization import NaNInfValidator
        
        validator = NaNInfValidator(
            check_interval=50,
            check_gradients=True,
            raise_on_nan=True,
        )
        
        assert validator.check_interval == 50
        assert validator.check_gradients is True
        assert validator.raise_on_nan is True
    
    def test_numerical_validator_extends_nan_inf(self):
        """Test NumericalValidator extends NaNInfValidator."""
        from deepseek.torch.training.optimization import NumericalValidator, NaNInfValidator
        
        validator = NumericalValidator()
        assert isinstance(validator, NaNInfValidator)
    
    def test_numerical_validator_loss_validation(self):
        """Test NumericalValidator loss validation."""
        from deepseek.torch.training.optimization import NumericalValidator
        
        validator = NumericalValidator(max_loss_value=100.0)
        
        # Valid loss
        valid_loss = torch.tensor(10.0)
        assert validator.validate_loss(valid_loss) is True
        
        # Loss exceeding threshold
        large_loss = torch.tensor(200.0)
        assert validator.validate_loss(large_loss) is False
    
    def test_numerical_validator_gradient_norm(self, simple_model):
        """Test NumericalValidator gradient norm checking."""
        from deepseek.torch.training.optimization import NumericalValidator
        
        validator = NumericalValidator()
        
        # Create gradients
        x = torch.randint(0, 1000, (2, 10))
        output = simple_model(x)
        loss = output.sum()
        loss.backward()
        
        norm, is_valid = validator.check_gradient_norm(simple_model, max_norm=1000.0)
        
        assert isinstance(norm, float)
        assert norm >= 0
        assert isinstance(is_valid, bool)


# =============================================================================
# Memory Budget Tests (Section 1.5)
# =============================================================================

class TestMemoryBudget:
    """Tests for memory budget management."""
    
    def test_memory_budget_config(self):
        """Test MemoryBudgetConfig dataclass."""
        from deepseek.torch.training.optimization import MemoryBudgetConfig
        
        config = MemoryBudgetConfig(
            max_memory_mb=8000.0,
            memory_fraction=0.85,
            min_batch_size=1,
            max_batch_size=128,
        )
        
        assert config.max_memory_mb == 8000.0
        assert config.memory_fraction == 0.85
        assert config.min_batch_size == 1
        assert config.max_batch_size == 128
    
    def test_memory_budget_manager_init(self):
        """Test MemoryBudgetManager initialization."""
        from deepseek.torch.training.optimization import MemoryBudgetManager, MemoryBudgetConfig
        
        config = MemoryBudgetConfig(max_memory_mb=4000.0)
        manager = MemoryBudgetManager(config)
        
        assert manager.config == config
        assert manager.memory_budget_mb == 4000.0
    
    def test_memory_budget_manager_default(self):
        """Test MemoryBudgetManager with default config."""
        from deepseek.torch.training.optimization import MemoryBudgetManager
        
        manager = MemoryBudgetManager()
        
        assert manager.config is not None
        assert manager.memory_budget_mb > 0
    
    def test_memory_budget_handle_oom(self):
        """Test MemoryBudgetManager OOM handling."""
        from deepseek.torch.training.optimization import MemoryBudgetManager, MemoryBudgetConfig
        
        config = MemoryBudgetConfig(min_batch_size=1)
        manager = MemoryBudgetManager(config)
        manager.current_batch_size = 32
        
        new_size = manager.handle_oom()
        
        assert new_size < 32  # Should reduce batch size
        assert new_size >= config.min_batch_size
        assert manager.oom_count == 1


# =============================================================================
# Gradient Accumulation Tests (Section 1.5)
# =============================================================================

class TestGradientAccumulation:
    """Tests for gradient accumulation."""
    
    def test_gradient_accumulation_config(self):
        """Test GradientAccumulationConfig dataclass."""
        from deepseek.torch.training.optimization import GradientAccumulationConfig
        
        config = GradientAccumulationConfig(
            accumulation_steps=4,
            normalize_gradients=True,
        )
        
        assert config.accumulation_steps == 4
        assert config.normalize_gradients is True
        assert config.effective_batch_multiplier == 4
    
    def test_gradient_accumulation_should_step(self):
        """Test GradientAccumulationConfig.should_step()."""
        from deepseek.torch.training.optimization import GradientAccumulationConfig
        
        config = GradientAccumulationConfig(accumulation_steps=4)
        
        # Should not step on steps 0, 1, 2
        assert config.should_step(0) is False
        assert config.should_step(1) is False
        assert config.should_step(2) is False
        
        # Should step on step 3 (4th step, 0-indexed)
        assert config.should_step(3) is True
        
        # Should step again on step 7
        assert config.should_step(7) is True
    
    def test_gradient_accumulation_loss_scale(self):
        """Test GradientAccumulationConfig.get_loss_scale()."""
        from deepseek.torch.training.optimization import GradientAccumulationConfig
        
        config = GradientAccumulationConfig(
            accumulation_steps=4,
            normalize_gradients=True,
        )
        
        assert config.get_loss_scale() == 0.25  # 1/4
        
        config_no_norm = GradientAccumulationConfig(
            accumulation_steps=4,
            normalize_gradients=False,
        )
        
        assert config_no_norm.get_loss_scale() == 1.0
    
    def test_create_gradient_accumulation_context(self, simple_model):
        """Test create_gradient_accumulation_context factory."""
        from deepseek.torch.training.optimization import create_gradient_accumulation_context
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        
        ctx = create_gradient_accumulation_context(
            accumulation_steps=4,
            optimizer=optimizer,
        )
        
        assert ctx.config.accumulation_steps == 4
        assert ctx.optimizer is optimizer
    
    def test_gradient_accumulation_context_backward(self, simple_model):
        """Test GradientAccumulationContext.backward()."""
        from deepseek.torch.training.optimization import create_gradient_accumulation_context
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        ctx = create_gradient_accumulation_context(
            accumulation_steps=2,
            optimizer=optimizer,
        )
        
        with ctx:
            x = torch.randint(0, 1000, (2, 10))
            output = simple_model(x)
            # Use absolute value to ensure positive loss
            loss = output.abs().mean()
            
            ctx.backward(loss)
            
            assert ctx.micro_step == 1
            assert ctx.accumulated_loss != 0  # Loss was accumulated
    
    def test_gradient_accumulation_context_step(self, simple_model):
        """Test GradientAccumulationContext.step()."""
        from deepseek.torch.training.optimization import create_gradient_accumulation_context
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        ctx = create_gradient_accumulation_context(
            accumulation_steps=2,
            optimizer=optimizer,
        )
        
        with ctx:
            # First micro-batch
            x1 = torch.randint(0, 1000, (2, 10))
            loss1 = simple_model(x1).abs().mean()
            ctx.backward(loss1)
            did_step1, _ = ctx.step()
            assert did_step1 is False
            
            # Second micro-batch (should step)
            x2 = torch.randint(0, 1000, (2, 10))
            loss2 = simple_model(x2).abs().mean()
            ctx.backward(loss2)
            did_step2, avg_loss = ctx.step()
            assert did_step2 is True
            assert avg_loss != 0  # Accumulated loss was returned


# =============================================================================
# CPU Offloading Tests (Section 1.5)
# =============================================================================

class TestCPUOffloading:
    """Tests for CPU offloading configuration."""
    
    def test_cpu_offload_config(self):
        """Test CPUOffloadConfig dataclass."""
        from deepseek.torch.training.optimization import CPUOffloadConfig
        
        config = CPUOffloadConfig(
            enabled=True,
            offload_optimizer=True,
            offload_gradients=False,
            pin_memory=True,
        )
        
        assert config.enabled is True
        assert config.offload_optimizer is True
        assert config.offload_gradients is False
        assert config.pin_memory is True
    
    def test_cpu_offload_create_state(self):
        """Test CPUOffloadConfig.create_cpu_optimizer_state()."""
        from deepseek.torch.training.optimization import CPUOffloadConfig
        
        config = CPUOffloadConfig(pin_memory=False)  # Disable pinning for CPU test
        param = torch.randn(10, 10)
        
        state = config.create_cpu_optimizer_state(param)
        
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert state["exp_avg"].device.type == "cpu"
        assert state["exp_avg_sq"].device.type == "cpu"
        assert state["exp_avg"].shape == param.shape
    
    def test_cpu_offload_transfer_to_cpu(self):
        """Test CPUOffloadConfig.transfer_to_cpu()."""
        from deepseek.torch.training.optimization import CPUOffloadConfig
        
        config = CPUOffloadConfig(pin_memory=False)
        state = {
            "exp_avg": torch.randn(10, 10),
            "exp_avg_sq": torch.randn(10, 10),
        }
        
        cpu_state = config.transfer_to_cpu(state)
        
        assert cpu_state["exp_avg"].device.type == "cpu"
        assert cpu_state["exp_avg_sq"].device.type == "cpu"


# =============================================================================
# Precision Weight Initialization Tests (Section 1.3)
# =============================================================================

class TestPrecisionWeightInit:
    """Tests for precision-specific weight initialization."""
    
    def test_precision_weight_initializer_init(self):
        """Test PrecisionWeightInitializer initialization."""
        from deepseek.torch.training.optimization import PrecisionWeightInitializer, PrecisionMode
        
        init = PrecisionWeightInitializer(
            precision=PrecisionMode.BF16,
            base_std=0.02,
        )
        
        assert init.precision == PrecisionMode.BF16
        assert init.base_std == 0.02
    
    def test_precision_weight_initializer_get_init_std(self):
        """Test PrecisionWeightInitializer.get_init_std()."""
        from deepseek.torch.training.optimization import PrecisionWeightInitializer, PrecisionMode
        
        init_fp32 = PrecisionWeightInitializer(precision=PrecisionMode.FP32)
        init_fp16 = PrecisionWeightInitializer(precision=PrecisionMode.FP16)
        init_bf16 = PrecisionWeightInitializer(precision=PrecisionMode.BF16)
        
        std_fp32 = init_fp32.get_init_std(64, 64)
        std_fp16 = init_fp16.get_init_std(64, 64)
        std_bf16 = init_bf16.get_init_std(64, 64)
        
        # FP16 should have smaller std than FP32/BF16
        assert std_fp16 < std_fp32
        assert std_fp16 < std_bf16
    
    def test_precision_weight_initializer_linear(self):
        """Test PrecisionWeightInitializer.initialize_linear()."""
        from deepseek.torch.training.optimization import PrecisionWeightInitializer, PrecisionMode
        
        init = PrecisionWeightInitializer(precision=PrecisionMode.BF16)
        linear = nn.Linear(64, 128)
        
        # Store original weight stats
        orig_mean = linear.weight.mean().item()
        
        init.initialize_linear(linear)
        
        # Weight should be reinitialized
        new_mean = linear.weight.mean().item()
        # Mean should be close to 0 after normal init
        assert abs(new_mean) < 0.1
        
        # Bias should be zeros
        if linear.bias is not None:
            assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))
    
    def test_precision_weight_initializer_model(self, simple_model):
        """Test PrecisionWeightInitializer.initialize_model()."""
        from deepseek.torch.training.optimization import PrecisionWeightInitializer, PrecisionMode
        
        init = PrecisionWeightInitializer(precision=PrecisionMode.BF16)
        
        # Should not raise
        init.initialize_model(simple_model)
        
        # Check that linear layers have zero bias
        for module in simple_model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))


# =============================================================================
# Compilation Overhead Measurement Tests (Section 1.2)
# =============================================================================

class TestCompilationOverhead:
    """Tests for compilation overhead measurement."""
    
    def test_measure_compilation_overhead(self, simple_model):
        """Test measure_compilation_overhead function."""
        from deepseek.torch.training.optimization import measure_compilation_overhead, CompileConfig, CompileMode
        
        sample_input = torch.randint(0, 1000, (2, 10))
        
        # Test with disabled compilation (fastest)
        config = CompileConfig(mode=CompileMode.DISABLED)
        results = measure_compilation_overhead(
            simple_model,
            sample_input,
            config,
            num_warmup=1,
            num_iterations=2,
        )
        
        assert "uncompiled_mean_ms" in results
        assert results["uncompiled_mean_ms"] > 0
    
    def test_measure_compilation_overhead_with_compile(self, simple_model):
        """Test measure_compilation_overhead with actual compilation."""
        from deepseek.torch.training.optimization import measure_compilation_overhead, CompileConfig, CompileMode
        
        sample_input = torch.randint(0, 1000, (2, 10))
        
        config = CompileConfig(mode=CompileMode.DEFAULT)
        results = measure_compilation_overhead(
            simple_model,
            sample_input,
            config,
            num_warmup=1,
            num_iterations=2,
        )
        
        assert "uncompiled_mean_ms" in results
        # If compilation worked, we should have compiled metrics
        if "compiled_mean_ms" in results:
            assert results["compiled_mean_ms"] > 0
            assert "speedup" in results


# =============================================================================
# Integration Tests
# =============================================================================

class TestTrainingOptimizationIntegration:
    """Integration tests combining multiple training optimization features."""
    
    def test_mixed_precision_with_gradient_accumulation(self, simple_model, device):
        """Test mixed precision training with gradient accumulation."""
        from deepseek.torch.training.optimization import (
            MixedPrecisionTrainer,
            MixedPrecisionConfig,
            PrecisionMode,
            create_gradient_accumulation_context,
        )
        
        simple_model = simple_model.to(device)
        config = MixedPrecisionConfig(mode=PrecisionMode.FP32)
        mp_trainer = MixedPrecisionTrainer(config)
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        ctx = create_gradient_accumulation_context(
            accumulation_steps=2,
            optimizer=optimizer,
        )
        
        with ctx:
            for i in range(4):
                x = torch.randint(0, 1000, (2, 10), device=device)
                with mp_trainer.autocast_context(device_type="cpu"):
                    output = simple_model(x)
                loss = output.abs().mean()
                ctx.backward(loss)
                did_step, avg_loss = ctx.step()
                
                # Should step every 2 iterations
                if (i + 1) % 2 == 0:
                    assert did_step is True
    
    def test_memory_profiling_with_validation(self, simple_model):
        """Test memory profiling with NaN validation."""
        from deepseek.torch.training.optimization import (
            MemoryProfiler,
            NumericalValidator,
        )
        
        profiler = MemoryProfiler()
        validator = NumericalValidator()
        
        with profiler.profile_region("training_step"):
            x = torch.randint(0, 1000, (4, 16))
            output = simple_model(x)
            loss = output.abs().mean()
            loss.backward()
        
        # Validate
        assert validator.validate_loss(loss) is True
        norm, is_valid = validator.check_gradient_norm(simple_model)
        assert is_valid is True
        
        # Get final stats
        stats = profiler.get_memory_stats()
        # On CPU, stats will be None
        if stats is not None:
            assert stats.allocated_mb >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
