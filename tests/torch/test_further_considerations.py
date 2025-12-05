"""
Tests for Further Considerations Implementations.

Tests the following Further Considerations tasks:
1. 256-Expert Hierarchical Routing Configuration
2. FP8/Precision Auto-Detection
3. MLX Distributed Placeholder
4. Model Scale Configurations

Run with:
    uv run pytest deepseek-from-scratch-python/tests/test_further_considerations.py -v
"""

import sys
from pathlib import Path

import pytest

# Add paths
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

mlx_path = Path(__file__).parent.parent / "mlx_impl"
if str(mlx_path) not in sys.path:
    sys.path.insert(0, str(mlx_path))

# Get project root (parent of deepseek-from-scratch-python)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Test: Hierarchical Routing Configuration
# =============================================================================

class TestHierarchicalRoutingConfig:
    """Tests for 256-Expert Hierarchical Routing Configuration."""
    
    def test_hierarchical_config_exists(self):
        """Test that hierarchical routing config file exists."""
        config_path = PROJECT_ROOT / "config/hydra/model/moe/hierarchical.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_default_moe_config_exists(self):
        """Test that default MoE config exists."""
        config_path = PROJECT_ROOT / "config/hydra/model/moe/default.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_hierarchical_config_parseable(self):
        """Test that hierarchical config is valid YAML."""
        import yaml
        config_path = PROJECT_ROOT / "config/hydra/model/moe/hierarchical.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key fields
        assert "moe" in config
        assert config["moe"]["use_hierarchical_routing"] is True
        assert "hierarchical_routing" in config["moe"]
        assert config["moe"]["hierarchical_routing"]["enabled"] is True
        assert config["moe"]["hierarchical_routing"]["num_groups"] == 8
    
    def test_schema_has_hierarchical_routing(self):
        """Test that schema defines hierarchical routing config."""
        # Import the schema
        sys.path.insert(0, str(PROJECT_ROOT / "config/hydra"))
        from schema import HierarchicalRoutingConfig, MoEConfig
        
        # Test defaults
        moe_config = MoEConfig()
        assert hasattr(moe_config, 'use_hierarchical_routing')
        assert hasattr(moe_config, 'hierarchical_routing')
        
        hr_config = HierarchicalRoutingConfig()
        assert hr_config.num_groups == 8
        assert hr_config.group_top_k == 2


# =============================================================================
# Test: Precision Auto-Detection
# =============================================================================

class TestPrecisionAutoDetection:
    """Tests for FP8/BF16/FP16 Precision Auto-Detection."""
    
    def test_precision_module_importable(self):
        """Test that precision module can be imported."""
        from deepseek.torch.utils.precision import (
            HardwareInfo,
            PrecisionConfig,
            PrecisionManager,
            PrecisionMode,
            detect_hardware_info,
            detect_optimal_precision,
        )
    
    def test_precision_modes_defined(self):
        """Test that all precision modes are defined."""
        from deepseek.torch.utils.precision import PrecisionMode
        
        assert PrecisionMode.AUTO.value == "auto"
        assert PrecisionMode.FP8.value == "fp8"
        assert PrecisionMode.BF16.value == "bf16"
        assert PrecisionMode.FP16.value == "fp16"
        assert PrecisionMode.FP32.value == "fp32"
    
    def test_hardware_detection(self):
        """Test hardware info detection."""
        from deepseek.torch.utils.precision import detect_hardware_info
        
        info = detect_hardware_info()
        assert info.device_name is not None
        assert isinstance(info.supports_fp8, bool)
        assert isinstance(info.supports_bf16, bool)
        assert isinstance(info.supports_fp16, bool)
    
    def test_optimal_precision_detection(self):
        """Test optimal precision detection."""
        from deepseek.torch.utils.precision import detect_optimal_precision, PrecisionMode
        
        precision = detect_optimal_precision()
        assert isinstance(precision, PrecisionMode)
    
    def test_precision_manager_creation(self):
        """Test PrecisionManager can be created."""
        from deepseek.torch.utils.precision import PrecisionManager
        
        # Test with different modes
        for mode in ["auto", "bf16", "fp16", "fp32"]:
            manager = PrecisionManager(mode=mode)
            assert manager.effective_mode is not None
    
    def test_precision_config_yaml_exists(self):
        """Test that precision config YAML exists."""
        config_path = PROJECT_ROOT / "config/hydra/training/precision/auto.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_precision_config_parseable(self):
        """Test that precision config is valid YAML."""
        import yaml
        config_path = PROJECT_ROOT / "config/hydra/training/precision/auto.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "precision" in config
        assert config["precision"]["mode"] == "auto"
        assert "fp8" in config["precision"]
        assert "bf16" in config["precision"]
        assert "fp16" in config["precision"]


# =============================================================================
# Test: MLX Distributed Placeholder
# =============================================================================

class TestMLXDistributedPlaceholder:
    """Tests for MLX Distributed Placeholder."""
    
    @pytest.fixture
    def mlx_available(self):
        """Check if MLX is available."""
        try:
            import mlx.core as mx
            return True
        except ImportError:
            return False
    
    def test_mlx_distributed_module_exists(self):
        """Test that MLX distributed module exists."""
        module_path = Path(__file__).parent.parent.parent / "src" / "deepseek" / "mlx" / "mlx_distributed.py"
        assert module_path.exists(), f"Module not found at {module_path}"
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "src" / "deepseek" / "mlx" / "mlx_distributed.py").exists(),
        reason="MLX distributed module not found"
    )
    def test_mlx_distributed_importable(self):
        """Test that MLX distributed can be imported."""
        try:
            from deepseek.mlx.mlx_distributed import (
                MLXDistributedConfig,
                MLXDistributedPlaceholder,
                detect_apple_silicon,
                get_available_memory_gb,
            )
        except ImportError as e:
            if "mlx" in str(e).lower():
                pytest.skip("MLX not installed")
            raise
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "mlx_impl" / "mlx_distributed.py").exists(),
        reason="MLX distributed module not found"
    )
    def test_mlx_distributed_error_message(self):
        """Test that distributed initialization raises helpful error."""
        try:
            from mlx_impl.mlx_distributed import MLXDistributedPlaceholder
            
            placeholder = MLXDistributedPlaceholder()
            
            # This should raise NotImplementedError with helpful message
            with pytest.raises(NotImplementedError) as exc_info:
                placeholder.init_process_group()
            
            error_msg = str(exc_info.value)
            assert "MLX DISTRIBUTED TRAINING NOT SUPPORTED" in error_msg
            assert "single-device" in error_msg.lower() or "single Apple Silicon" in error_msg
        except ImportError:
            pytest.skip("MLX not installed")
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "mlx_impl" / "mlx_distributed.py").exists(),
        reason="MLX distributed module not found"  
    )
    def test_mlx_distributed_is_available_returns_false(self):
        """Test that is_available returns False."""
        try:
            from mlx_impl.mlx_distributed import MLXDistributedPlaceholder
            
            placeholder = MLXDistributedPlaceholder()
            assert placeholder.is_available() is False
        except ImportError:
            pytest.skip("MLX not installed")


# =============================================================================
# Test: Model Scale Configurations
# =============================================================================

class TestModelScaleConfigs:
    """Tests for Model Scale Configurations (109M, 1B, 7B+)."""
    
    def test_109m_config_exists(self):
        """Test that 109M config exists."""
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_109m.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_1b_config_exists(self):
        """Test that 1B config exists."""
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_1b.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_7b_config_exists(self):
        """Test that 7B config exists."""
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_7b.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_v3_config_exists(self):
        """Test that V3 config exists."""
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_v3.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_109m_config_valid(self):
        """Test that 109M config is valid."""
        import yaml
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_109m.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key fields
        assert config["name"] == "deepseek_109m"
        assert config["num_layers"] == 6
        assert config["d_model"] == 512
    
    def test_1b_config_valid(self):
        """Test that 1B config is valid."""
        import yaml
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_1b.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key fields
        assert config["name"] == "deepseek_1b"
        assert config["num_layers"] == 24
        assert config["d_model"] == 2048
        assert config["moe"]["num_experts"] == 16
    
    def test_7b_config_valid(self):
        """Test that 7B config is valid."""
        import yaml
        config_path = PROJECT_ROOT / "config/hydra/model/deepseek_7b.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key fields
        assert config["name"] == "deepseek_7b"
        assert config["num_layers"] == 32
        assert config["d_model"] == 4096
        assert config["moe"]["num_experts"] == 64
        assert config["moe"]["use_hierarchical_routing"] is True
    
    def test_configs_have_training_sections(self):
        """Test that all scale configs have training recommendations."""
        import yaml
        
        configs = ["deepseek_109m.yaml", "deepseek_1b.yaml", "deepseek_7b.yaml"]
        for config_name in configs:
            config_path = PROJECT_ROOT / f"config/hydra/model/{config_name}"
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            assert "training" in config, f"{config_name} missing training section"


# =============================================================================
# Test: Success Criteria Infrastructure
# =============================================================================

class TestSuccessCriteriaInfrastructure:
    """Tests for Success Criteria infrastructure."""
    
    def test_ablation_scripts_exist(self):
        """Test that ablation study scripts exist."""
        ablation_dir = PROJECT_ROOT / "scripts/ablation"
        
        expected_scripts = [
            "run_all_ablations.py",
            "run_attention_ablation.py",
            "run_expert_ablation.py",
            "run_balancing_ablation.py",
            "run_mtp_ablation.py",
            "run_precision_ablation.py",
        ]
        
        for script in expected_scripts:
            script_path = ablation_dir / script
            assert script_path.exists(), f"Ablation script not found: {script}"
    
    def test_blog_posts_exist(self):
        """Test that technical blog posts exist."""
        blog_dir = PROJECT_ROOT / "docs/blog"
        
        expected_posts = [
            "01_mla_deep_dive.md",
            "02_auxiliary_loss_free.md",
            "03_dualpipe_explained.md",
            "04_expert_specialization.md",
            "05_production_lessons.md",
        ]
        
        for post in expected_posts:
            post_path = blog_dir / post
            assert post_path.exists(), f"Blog post not found: {post}"
    
    def test_reproducibility_doc_exists(self):
        """Test that reproducibility documentation exists."""
        repro_path = PROJECT_ROOT / "REPRODUCIBILITY.md"
        assert repro_path.exists(), "REPRODUCIBILITY.md not found"
    
    def test_ci_workflow_exists(self):
        """Test that CI workflow exists."""
        ci_path = PROJECT_ROOT / ".github/workflows/ci.yml"
        assert ci_path.exists(), "CI workflow not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
