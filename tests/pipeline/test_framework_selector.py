"""
Tests for Framework Selection and Fallback System

This test module verifies:
1. Framework detection and availability
2. Preference chain resolution with fallbacks
3. Configuration presets
4. Integration with pipeline stages
"""

from __future__ import annotations

from deepseek.pipeline.framework_selector import (
    Framework,
    FrameworkAvailability,
    FrameworkExecutor,
    FrameworkPreference,
    FrameworkSelector,
    PipelineFrameworkConfig,
    TaskType,
    configure_framework_selector,
    get_framework_selector,
    select_framework,
)

# =============================================================================
# Framework Detection Tests
# =============================================================================


class TestFrameworkDetection:
    """Tests for hardware/framework detection."""

    def test_framework_availability_detect(self) -> None:
        """Test framework availability detection."""
        avail = FrameworkAvailability.detect()

        assert isinstance(avail, FrameworkAvailability)
        # Some basic checks - system dependent but should not error
        assert hasattr(avail, "pytorch_available")
        assert hasattr(avail, "mlx_available")
        assert hasattr(avail, "rust_available")

    def test_availability_get_available_frameworks(self) -> None:
        """Test getting list of available frameworks."""
        avail = FrameworkAvailability.detect()
        available = avail.get_available_frameworks()

        assert isinstance(available, list)
        # At minimum should have Python CPU
        assert Framework.PYTHON_CPU in available

    def test_availability_is_framework_available(self) -> None:
        """Test checking specific framework availability."""
        avail = FrameworkAvailability.detect()

        # Python CPU should always be available
        assert avail.is_framework_available(Framework.PYTHON_CPU) is True
        # AUTO should always return True
        assert avail.is_framework_available(Framework.AUTO) is True


# =============================================================================
# Framework Preference Tests
# =============================================================================


class TestFrameworkPreference:
    """Tests for FrameworkPreference configuration."""

    def test_basic_preference_creation(self) -> None:
        """Test creating basic preference."""
        pref = FrameworkPreference(
            primary=Framework.PYTORCH_CUDA,
            fallbacks=[Framework.PYTORCH_MPS, Framework.PYTORCH_CPU],
        )

        assert pref.primary == Framework.PYTORCH_CUDA
        assert len(pref.fallbacks) == 2
        assert Framework.PYTORCH_MPS in pref.fallbacks

    def test_get_chain(self) -> None:
        """Test getting full fallback chain."""
        pref = FrameworkPreference(
            primary=Framework.MLX,
            fallbacks=[Framework.RUST_METAL, Framework.PYTORCH_MPS],
        )

        chain = pref.get_chain()
        assert chain[0] == Framework.MLX
        assert len(chain) == 3

    def test_apple_silicon_preset(self) -> None:
        """Test Apple Silicon optimized preset."""
        pref = FrameworkPreference.apple_silicon_optimized("test_task")

        assert pref.primary == Framework.MLX
        assert Framework.RUST_METAL in pref.fallbacks
        assert pref.name == "test_task"

    def test_gpu_training_preset(self) -> None:
        """Test GPU training preset."""
        pref = FrameworkPreference.gpu_training("training")

        assert pref.primary == Framework.PYTORCH_CUDA
        assert Framework.RUST_CUDA in pref.fallbacks

    def test_rust_gpu_primary_preset(self) -> None:
        """Test Rust GPU primary preset."""
        pref = FrameworkPreference.rust_gpu_primary("rust_task")

        assert pref.primary == Framework.RUST_CUDA
        assert Framework.PYTORCH_CUDA in pref.fallbacks

    def test_rust_metal_primary_preset(self) -> None:
        """Test Rust Metal primary preset."""
        pref = FrameworkPreference.rust_metal_primary("metal_task")

        assert pref.primary == Framework.RUST_METAL
        assert Framework.MLX in pref.fallbacks

    def test_data_processing_preset(self) -> None:
        """Test data processing preset."""
        pref = FrameworkPreference.data_processing("data_task")

        assert pref.primary == Framework.RUST_CPU
        assert Framework.PYTHON_CPU in pref.fallbacks


# =============================================================================
# Pipeline Framework Config Tests
# =============================================================================


class TestPipelineFrameworkConfig:
    """Tests for PipelineFrameworkConfig."""

    def test_default_config_creation(self) -> None:
        """Test creating default config."""
        config = PipelineFrameworkConfig()

        assert config.data_loading is not None
        assert config.tokenization is not None
        assert config.forward_pass is not None
        assert config.backward_pass is not None
        assert config.generation is not None
        assert config.policy_update is not None

    def test_default_generation_prefers_mlx(self) -> None:
        """Test default generation preference is MLX."""
        config = PipelineFrameworkConfig()

        # Generation should prefer MLX (Apple Silicon optimized)
        assert config.generation.primary == Framework.MLX

    def test_default_training_prefers_cuda(self) -> None:
        """Test default training preference is CUDA."""
        config = PipelineFrameworkConfig()

        # Training should prefer PyTorch CUDA
        assert config.forward_pass.primary == Framework.PYTORCH_CUDA
        assert config.backward_pass.primary == Framework.PYTORCH_CUDA

    def test_set_all_to_rust_primary(self) -> None:
        """Test switching all tasks to Rust primary."""
        config = PipelineFrameworkConfig()
        config.set_all_to_rust_primary()

        # Training tasks should now use Rust CUDA
        assert config.forward_pass.primary == Framework.RUST_CUDA
        assert config.backward_pass.primary == Framework.RUST_CUDA

        # Generation tasks should use Rust Metal
        assert config.generation.primary == Framework.RUST_METAL

    def test_set_all_to_pytorch_primary(self) -> None:
        """Test switching all tasks to PyTorch primary."""
        config = PipelineFrameworkConfig()
        config.set_all_to_pytorch_primary()

        # Training tasks should use PyTorch CUDA
        assert config.forward_pass.primary == Framework.PYTORCH_CUDA
        assert config.backward_pass.primary == Framework.PYTORCH_CUDA

    def test_get_preference_for_task_type(self) -> None:
        """Test getting preference for task type enum."""
        config = PipelineFrameworkConfig()

        pref = config.get_preference(TaskType.FORWARD_PASS)
        assert pref is not None
        assert isinstance(pref, FrameworkPreference)

    def test_from_preset_rust_primary(self) -> None:
        """Test creating config from rust_primary preset."""
        config = PipelineFrameworkConfig.from_preset("rust_primary")

        assert config.forward_pass.primary == Framework.RUST_CUDA
        assert config.generation.primary == Framework.RUST_METAL

    def test_from_preset_pytorch_only(self) -> None:
        """Test creating config from pytorch_only preset."""
        config = PipelineFrameworkConfig.from_preset("pytorch_only")

        assert config.forward_pass.primary == Framework.PYTORCH_CUDA

    def test_from_preset_apple_silicon(self) -> None:
        """Test creating config from apple_silicon preset."""
        config = PipelineFrameworkConfig.from_preset("apple_silicon")

        # Generation should use Rust Metal
        assert config.generation.primary == Framework.RUST_METAL


# =============================================================================
# Framework Selector Tests
# =============================================================================


class TestFrameworkSelector:
    """Tests for FrameworkSelector."""

    def test_selector_creation(self) -> None:
        """Test creating selector."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        assert selector.config == config
        assert selector.availability is not None

    def test_select_framework_for_task(self) -> None:
        """Test selecting framework for task type."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        framework = selector.select(TaskType.DATA_LOADING)
        assert isinstance(framework, Framework)

    def test_select_with_override_available(self) -> None:
        """Test selecting with override when available."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        # Python CPU should always be available
        framework = selector.select_with_override(
            TaskType.DATA_LOADING,
            override=Framework.PYTHON_CPU,
        )
        assert framework == Framework.PYTHON_CPU

    def test_select_with_override_unavailable_falls_back(self) -> None:
        """Test selecting with unavailable override falls back to preference."""
        # Create selector with mock availability
        avail = FrameworkAvailability()
        avail.pytorch_cuda_available = False
        avail.pytorch_mps_available = False
        avail.pytorch_available = True
        avail.mlx_available = False
        avail.rust_available = False

        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config, availability=avail)

        # Request CUDA but it's not available
        framework = selector.select_with_override(
            TaskType.FORWARD_PASS,
            override=Framework.PYTORCH_CUDA,
        )
        # Should fall back (likely to CPU)
        assert framework != Framework.PYTORCH_CUDA

    def test_get_executor(self) -> None:
        """Test getting executor for task."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        executor = selector.get_executor(TaskType.DATA_LOADING)
        assert isinstance(executor, FrameworkExecutor)

    def test_clear_cache(self) -> None:
        """Test clearing selection cache."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        # Make a selection to populate cache
        selector.select(TaskType.FORWARD_PASS)
        assert len(selector._selection_cache) > 0

        # Clear cache
        selector.clear_cache()
        assert len(selector._selection_cache) == 0

    def test_reconfigure(self) -> None:
        """Test reconfiguring selector."""
        config1 = PipelineFrameworkConfig()
        selector = FrameworkSelector(config1)

        # Make a selection
        selector.select(TaskType.FORWARD_PASS)

        # Reconfigure
        config2 = PipelineFrameworkConfig.from_preset("rust_primary")
        selector.reconfigure(config2)

        assert selector.config == config2
        assert len(selector._selection_cache) == 0


# =============================================================================
# Framework Executor Tests
# =============================================================================


class TestFrameworkExecutor:
    """Tests for FrameworkExecutor."""

    def test_executor_creation(self) -> None:
        """Test creating executor."""
        executor = FrameworkExecutor(Framework.PYTHON_CPU, TaskType.DATA_LOADING)

        assert executor.framework == Framework.PYTHON_CPU
        assert executor.task_type == TaskType.DATA_LOADING

    def test_execute_simple_function(self) -> None:
        """Test executing simple function."""
        executor = FrameworkExecutor(Framework.PYTHON_CPU, TaskType.DATA_LOADING)

        def simple_task(x: int) -> int:
            return x * 2

        result = executor.execute(simple_task, 21)
        assert result == 42

    def test_execute_with_kwargs(self) -> None:
        """Test executing with keyword arguments."""
        executor = FrameworkExecutor(Framework.PYTHON_CPU, TaskType.DATA_LOADING)

        def task_with_kwargs(a: int, b: int = 10) -> int:
            return a + b

        result = executor.execute(task_with_kwargs, 5, b=15)
        assert result == 20

    def test_get_device_string(self) -> None:
        """Test getting device string."""
        cpu_executor = FrameworkExecutor(Framework.PYTHON_CPU, TaskType.DATA_LOADING)
        assert cpu_executor.get_device_string() == "cpu"

        cuda_executor = FrameworkExecutor(Framework.PYTORCH_CUDA, TaskType.FORWARD_PASS)
        assert cuda_executor.get_device_string() == "cuda"

        mps_executor = FrameworkExecutor(Framework.PYTORCH_MPS, TaskType.FORWARD_PASS)
        assert mps_executor.get_device_string() == "mps"

        mlx_executor = FrameworkExecutor(Framework.MLX, TaskType.GENERATION)
        assert mlx_executor.get_device_string() == "gpu"


# =============================================================================
# Global Functions Tests
# =============================================================================


class TestGlobalFunctions:
    """Tests for global selector functions."""

    def test_get_framework_selector(self) -> None:
        """Test getting global selector."""
        selector = get_framework_selector()
        assert isinstance(selector, FrameworkSelector)

    def test_configure_framework_selector(self) -> None:
        """Test configuring global selector."""
        config = PipelineFrameworkConfig.from_preset("rust_primary")
        selector = configure_framework_selector(config=config)

        assert selector.config == config

    def test_configure_with_preset(self) -> None:
        """Test configuring with preset name."""
        selector = configure_framework_selector(preset="pytorch_only")

        # Should have pytorch configuration
        assert selector.config.forward_pass.primary == Framework.PYTORCH_CUDA

    def test_select_framework_global(self) -> None:
        """Test global select_framework function."""
        framework = select_framework(TaskType.DATA_LOADING)
        assert isinstance(framework, Framework)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFrameworkSelectorIntegration:
    """Integration tests for framework selection system."""

    def test_full_pipeline_config_creation(self) -> None:
        """Test creating config for full pipeline."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        # Should be able to get frameworks for all task types
        for task_type in TaskType:
            framework = selector.select(task_type)
            assert framework is not None
            assert isinstance(framework, Framework)

    def test_framework_switching_at_runtime(self) -> None:
        """Test switching framework preferences at runtime."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        # Get initial framework
        initial = selector.select(TaskType.FORWARD_PASS)

        # Switch to Rust primary
        config.set_all_to_rust_primary()
        selector.clear_cache()

        # Get updated framework (may or may not change based on availability)
        updated = selector.select(TaskType.FORWARD_PASS)

        # Both should be valid frameworks
        assert isinstance(initial, Framework)
        assert isinstance(updated, Framework)

    def test_heterogeneous_grpo_workflow(self) -> None:
        """Test typical heterogeneous GRPO workflow."""
        config = PipelineFrameworkConfig()
        selector = FrameworkSelector(config)

        # Get frameworks for GRPO workflow
        gen_fw = selector.select(TaskType.GENERATION)
        policy_fw = selector.select(TaskType.POLICY_UPDATE)
        kl_fw = selector.select(TaskType.KL_COMPUTATION)

        # All should be valid
        assert gen_fw is not None
        assert policy_fw is not None
        assert kl_fw is not None


# =============================================================================
# Framework Enum Tests
# =============================================================================


class TestFrameworkEnum:
    """Tests for Framework enum methods."""

    def test_is_gpu(self) -> None:
        """Test is_gpu method."""
        assert Framework.PYTORCH_CUDA.is_gpu() is True
        assert Framework.PYTORCH_MPS.is_gpu() is True
        assert Framework.MLX.is_gpu() is True
        assert Framework.RUST_CUDA.is_gpu() is True
        assert Framework.RUST_METAL.is_gpu() is True

        assert Framework.PYTHON_CPU.is_gpu() is False
        assert Framework.PYTORCH_CPU.is_gpu() is False
        assert Framework.RUST_CPU.is_gpu() is False

    def test_is_rust(self) -> None:
        """Test is_rust method."""
        assert Framework.RUST_CPU.is_rust() is True
        assert Framework.RUST_CUDA.is_rust() is True
        assert Framework.RUST_METAL.is_rust() is True

        assert Framework.PYTORCH_CUDA.is_rust() is False
        assert Framework.MLX.is_rust() is False

    def test_is_pytorch(self) -> None:
        """Test is_pytorch method."""
        assert Framework.PYTORCH_CPU.is_pytorch() is True
        assert Framework.PYTORCH_CUDA.is_pytorch() is True
        assert Framework.PYTORCH_MPS.is_pytorch() is True

        assert Framework.RUST_CUDA.is_pytorch() is False
        assert Framework.MLX.is_pytorch() is False

    def test_is_apple_silicon(self) -> None:
        """Test is_apple_silicon method."""
        assert Framework.MLX.is_apple_silicon() is True
        assert Framework.RUST_METAL.is_apple_silicon() is True
        assert Framework.PYTORCH_MPS.is_apple_silicon() is True

        assert Framework.PYTORCH_CUDA.is_apple_silicon() is False
        assert Framework.RUST_CUDA.is_apple_silicon() is False


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_fallback_chain(self) -> None:
        """Test preference with empty fallback chain."""
        pref = FrameworkPreference(
            primary=Framework.PYTHON_CPU,
            fallbacks=[],
        )

        chain = pref.get_chain()
        assert len(chain) == 1
        assert chain[0] == Framework.PYTHON_CPU

    def test_preference_with_task_name(self) -> None:
        """Test preference includes task name."""
        pref = FrameworkPreference(
            primary=Framework.PYTORCH_CUDA,
            fallbacks=[],
            name="critical_task",
        )

        assert pref.name == "critical_task"

    def test_duplicate_frameworks_in_fallback(self) -> None:
        """Test handling duplicate frameworks in fallback chain."""
        pref = FrameworkPreference(
            primary=Framework.PYTORCH_CUDA,
            fallbacks=[
                Framework.PYTORCH_CPU,
                Framework.PYTORCH_CPU,  # Duplicate
                Framework.PYTHON_CPU,
            ],
        )

        chain = pref.get_chain()
        # Should include duplicates as-is
        assert len(chain) == 4

    def test_all_task_types_covered(self) -> None:
        """Test all task types have preferences."""
        config = PipelineFrameworkConfig()

        for task_type in TaskType:
            pref = config.get_preference(task_type)
            assert pref is not None
            assert isinstance(pref, FrameworkPreference)
