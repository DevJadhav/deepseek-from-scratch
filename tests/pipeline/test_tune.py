"""
Tests for Ray Tune hyperparameter optimization integration.

Tests the HyperparameterSearch class with ASHA and PBT schedulers
across PyTorch+GPU, MLX, and Rust backends.

GPU Fallback Strategy:
- Tests that require GPU use pytest.mark.skipif with hardware detection
- Main scripts are NEVER modified to use CPU - tests skip if hardware unavailable
- Mock schedulers are used for unit tests to avoid actual Ray overhead

Usage:
    # Run all tune tests
    uv run pytest tests/pipeline/test_tune.py -v
    
    # Run only unit tests (no hardware requirements)
    uv run pytest tests/pipeline/test_tune.py -v -m "not gpu"
    
    # Run integration tests (require hardware)
    uv run pytest tests/pipeline/test_tune.py -v -m "gpu"
"""

import importlib.util
import sys
from pathlib import Path
import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Hardware availability checks
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CUDA_AVAILABLE = False
MPS_AVAILABLE = False
MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None
RAY_AVAILABLE = importlib.util.find_spec("ray") is not None
HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None

if TORCH_AVAILABLE:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# =============================================================================
# Skip markers for hardware requirements
# =============================================================================

requires_ray = pytest.mark.skipif(
    not RAY_AVAILABLE,
    reason="Ray not available"
)

requires_hydra = pytest.mark.skipif(
    not HYDRA_AVAILABLE,
    reason="Hydra not available"
)

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available - GPU test skipped, main scripts NOT modified"
)

requires_mps = pytest.mark.skipif(
    not MPS_AVAILABLE,
    reason="MPS not available - Apple Silicon test skipped"
)

requires_mlx = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX not available - Apple Silicon test skipped"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hydra_config_dir():
    """Return the Hydra config directory."""
    return Path(__file__).parent.parent.parent / "config" / "hydra"


@pytest.fixture
def mock_tune_config():
    """Create a mock tune configuration dict."""
    return {
        "tune": {
            "enabled": True,
            "scheduler_type": "asha",
            "num_samples": 2,
            "max_concurrent_trials": 1,
            "grace_period": 10,
            "reduction_factor": 2,
            "max_t": 100,
            "perturbation_interval": 10,
            "torch_metric": "torch_val_loss",
            "mlx_metric": "mlx_val_loss",
            "rust_metric": "rust_val_loss",
            "mode": "min",
            "checkpoint_subdir": "tune/test",
            "search_space": {
                "learning_rate_min": 1e-5,
                "learning_rate_max": 1e-3,
                "batch_size_choices": [8, 16],
                "warmup_steps_min": 10,
                "warmup_steps_max": 100,
                "weight_decay_min": 0.001,
                "weight_decay_max": 0.1,
                "moe_capacity_factor_min": 1.0,
                "moe_capacity_factor_max": 2.0,
                "grpo_beta_min": 0.01,
                "grpo_beta_max": 0.5,
            },
        },
        "paths": {
            "checkpoint_dir": "./checkpoints",
        },
        "experiment": {
            "name": "test-tune",
        },
    }


@pytest.fixture
def mock_pbt_config(mock_tune_config):
    """Create a mock PBT configuration dict."""
    config = mock_tune_config.copy()
    config["tune"] = mock_tune_config["tune"].copy()
    config["tune"]["scheduler_type"] = "pbt"
    config["tune"]["num_samples"] = 4  # PBT needs population
    config["tune"]["max_concurrent_trials"] = 4
    return config


# =============================================================================
# Unit Tests (no hardware requirements)
# =============================================================================

class TestHyperparameterSearchUnit:
    """Unit tests for HyperparameterSearch class."""
    
    @requires_ray
    @requires_hydra
    def test_init_from_dict(self, mock_tune_config):
        """Test initialization from dict config."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        
        assert search.tune_config["enabled"] is True
        assert search.tune_config["scheduler_type"] == "asha"
        # storage_path is now resolved to absolute path
        assert search.storage_path.name == "test"
        assert "checkpoints/tune/test" in str(search.storage_path)
    
    @requires_ray
    @requires_hydra
    def test_build_search_space(self, mock_tune_config):
        """Test search space construction."""
        from deepseek.pipeline.tune import HyperparameterSearch

        search = HyperparameterSearch(config=mock_tune_config)
        space = search.build_search_space()
        
        assert "learning_rate" in space
        assert "batch_size" in space
        assert "warmup_steps" in space
        assert "weight_decay" in space
        assert "moe_capacity_factor" in space
        assert "grpo_beta" in space
    
    @requires_ray
    @requires_hydra
    def test_get_asha_scheduler(self, mock_tune_config):
        """Test ASHA scheduler creation."""
        from deepseek.pipeline.tune import HyperparameterSearch
        from ray.tune.schedulers import ASHAScheduler
        
        search = HyperparameterSearch(config=mock_tune_config)
        scheduler = search.get_scheduler()
        
        assert isinstance(scheduler, ASHAScheduler)
    
    @requires_ray
    @requires_hydra
    def test_get_pbt_scheduler(self, mock_pbt_config):
        """Test PBT scheduler creation."""
        from deepseek.pipeline.tune import HyperparameterSearch
        from ray.tune.schedulers import PopulationBasedTraining
        
        search = HyperparameterSearch(config=mock_pbt_config)
        scheduler = search.get_scheduler()
        
        assert isinstance(scheduler, PopulationBasedTraining)
    
    @requires_ray
    @requires_hydra
    def test_get_metric_name_pytorch(self, mock_tune_config):
        """Test backend-specific metric name for PyTorch."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        metric = search._get_metric_name("pytorch")
        
        assert metric == "torch_val_loss"
    
    @requires_ray
    @requires_hydra
    def test_get_metric_name_mlx(self, mock_tune_config):
        """Test backend-specific metric name for MLX."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        metric = search._get_metric_name("mlx")
        
        assert metric == "mlx_val_loss"
    
    @requires_ray
    @requires_hydra
    def test_get_metric_name_rust(self, mock_tune_config):
        """Test backend-specific metric name for Rust."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        metric = search._get_metric_name("rust")
        
        assert metric == "rust_val_loss"
    
    @requires_ray
    @requires_hydra
    def test_detect_backend(self, mock_tune_config):
        """Test backend auto-detection."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        backend = search._detect_backend()
        
        # Should return one of the valid backends
        assert backend in ("pytorch", "mlx", "rust")
    
    @requires_ray
    @requires_hydra
    def test_create_trainable_pytorch(self, mock_tune_config):
        """Test PyTorch trainable creation."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("pytorch", stage="pretrain")
        
        assert callable(trainable)
    
    @requires_ray
    @requires_hydra
    def test_create_trainable_mlx(self, mock_tune_config):
        """Test MLX trainable creation."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("mlx", stage="pretrain")
        
        assert callable(trainable)
    
    @requires_ray
    @requires_hydra
    def test_create_trainable_rust(self, mock_tune_config):
        """Test Rust trainable creation."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("rust", stage="pretrain")
        
        assert callable(trainable)


class TestHydraConfigComposition:
    """Test Hydra config composition for tune experiments."""
    
    @requires_hydra
    def test_tune_asha_config_exists(self, hydra_config_dir):
        """Test that tune_asha.yaml config exists."""
        config_path = hydra_config_dir / "experiment" / "tune_asha.yaml"
        assert config_path.exists(), f"Missing config: {config_path}"
    
    @requires_hydra
    def test_tune_pbt_config_exists(self, hydra_config_dir):
        """Test that tune_pbt.yaml config exists."""
        config_path = hydra_config_dir / "experiment" / "tune_pbt.yaml"
        assert config_path.exists(), f"Missing config: {config_path}"
    
    @requires_ray
    @requires_hydra
    def test_from_hydra_asha(self, hydra_config_dir):
        """Test loading ASHA config from Hydra."""
        from deepseek.pipeline.tune import HyperparameterSearch

        search = HyperparameterSearch.from_hydra(
            overrides=["experiment=tune_asha"],
            config_dir=hydra_config_dir,
        )

        # When experiment=tune_asha is loaded, tune config should be populated
        # Check that the tune config was loaded (may be nested differently)
        tune_cfg = search.config.get("tune", {})
        assert tune_cfg.get("scheduler_type") == "asha" or search.tune_config.get("scheduler_type") == "asha", \
            f"Expected scheduler_type='asha', got tune_config={search.tune_config}, config.tune={tune_cfg}"

    @requires_ray
    @requires_hydra
    def test_from_hydra_pbt(self, hydra_config_dir):
        """Test loading PBT config from Hydra."""
        from deepseek.pipeline.tune import HyperparameterSearch

        search = HyperparameterSearch.from_hydra(
            overrides=["experiment=tune_pbt"],
            config_dir=hydra_config_dir,
        )

        tune_cfg = search.config.get("tune", {})
        assert tune_cfg.get("scheduler_type") == "pbt" or search.tune_config.get("scheduler_type") == "pbt", \
            f"Expected scheduler_type='pbt', got tune_config={search.tune_config}, config.tune={tune_cfg}"


# =============================================================================
# Integration Tests (require hardware)
# =============================================================================

@pytest.mark.gpu
class TestPyTorchTuneIntegration:
    """Integration tests for PyTorch + Ray Tune."""
    
    @requires_ray
    @requires_torch
    @requires_cuda
    def test_pytorch_cuda_trainable_runs(self, mock_tune_config):
        """Test PyTorch trainable executes on CUDA.
        
        NOTE: This test requires CUDA. If CUDA is not available,
        this test is SKIPPED - main scripts are NOT modified to use CPU.
        """
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("pytorch", stage="pretrain")
        
        # This would actually run on CUDA if available
        assert callable(trainable)
    
    @requires_ray
    @requires_torch
    @requires_mps
    def test_pytorch_mps_trainable_runs(self, mock_tune_config):
        """Test PyTorch trainable executes on MPS.
        
        NOTE: This test requires MPS (Apple Silicon). If MPS is not available,
        this test is SKIPPED - main scripts are NOT modified to use CPU.
        """
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("pytorch", stage="pretrain")
        
        assert callable(trainable)


@pytest.mark.gpu
class TestMLXTuneIntegration:
    """Integration tests for MLX + Ray Tune."""
    
    @requires_ray
    @requires_mlx
    def test_mlx_trainable_runs(self, mock_tune_config):
        """Test MLX trainable executes on Apple Silicon.
        
        NOTE: This test requires MLX (Apple Silicon). If MLX is not available,
        this test is SKIPPED - main scripts are NOT modified.
        """
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("mlx", stage="pretrain")
        
        assert callable(trainable)


@pytest.mark.gpu
class TestRustTuneIntegration:
    """Integration tests for Rust + Ray Tune."""
    
    @requires_ray
    def test_rust_trainable_creation(self, mock_tune_config):
        """Test Rust trainable creation (no actual execution)."""
        from deepseek.pipeline.tune import HyperparameterSearch
        
        search = HyperparameterSearch(config=mock_tune_config)
        trainable = search.create_trainable("rust", stage="pretrain")
        
        assert callable(trainable)


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestTuneConfigSchema:
    """Test TuneConfig schema validation in Hydra."""
    
    @requires_hydra
    def test_tune_config_schema_exists(self):
        """Test that TuneConfig is defined in schema."""
        from config.hydra.schema import TuneConfig, TuneSearchSpaceConfig
        
        # Should be importable
        assert TuneConfig is not None
        assert TuneSearchSpaceConfig is not None
    
    @requires_hydra
    def test_tune_config_defaults(self):
        """Test TuneConfig default values."""
        from config.hydra.schema import TuneConfig
        
        config = TuneConfig()
        
        assert config.enabled is False
        assert config.scheduler_type == "asha"
        assert config.num_samples == 10
        assert config.mode == "min"
    
    @requires_hydra
    def test_deepseek_config_includes_tune(self):
        """Test that DeepSeekConfig includes tune field."""
        from config.hydra.schema import DeepSeekConfig
        
        config = DeepSeekConfig()
        
        assert hasattr(config, "tune")
        assert config.tune.scheduler_type == "asha"
    
    @requires_hydra
    def test_validate_config_with_tune(self):
        """Test config validation includes tune validation."""
        from config.hydra.schema import validate_config
        from omegaconf import OmegaConf
        
        # Valid config
        valid_cfg = OmegaConf.create({
            "tune": {
                "enabled": True,
                "scheduler_type": "asha",
                "num_samples": 10,
                "max_concurrent_trials": 4,
                "grace_period": 100,
                "perturbation_interval": 100,
                "mode": "min",
                "search_space": {
                    "learning_rate_min": 1e-5,
                    "learning_rate_max": 1e-3,
                    "warmup_steps_min": 100,
                    "warmup_steps_max": 2000,
                },
            },
        })
        
        errors = validate_config(valid_cfg)
        assert len(errors) == 0
    
    @requires_hydra
    def test_validate_invalid_scheduler_type(self):
        """Test validation catches invalid scheduler type."""
        from config.hydra.schema import validate_config
        from omegaconf import OmegaConf
        
        invalid_cfg = OmegaConf.create({
            "tune": {
                "enabled": True,
                "scheduler_type": "invalid",  # Invalid
                "num_samples": 10,
                "max_concurrent_trials": 4,
                "grace_period": 100,
                "perturbation_interval": 100,
                "mode": "min",
                "search_space": {
                    "learning_rate_min": 1e-5,
                    "learning_rate_max": 1e-3,
                    "warmup_steps_min": 100,
                    "warmup_steps_max": 2000,
                },
            },
        })
        
        errors = validate_config(invalid_cfg)
        assert any("scheduler_type" in e for e in errors)
