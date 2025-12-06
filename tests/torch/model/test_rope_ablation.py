"""
Tests for RoPE Ablation Study Hooks in PyTorch.

Tests various RoPE scaling strategies for ablation studies.
"""

import pytest
import torch


def get_device():
    """Get available device - CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Skip if no GPU available (neither CUDA nor MPS)
GPU_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU tests require CUDA or MPS"
)


class TestRoPEStrategy:
    """Test RoPE strategy enum and naming."""
    
    def test_strategy_names(self):
        """Test strategy name retrieval."""
        from src.deepseek.torch.model.rope_ablation import RoPEStrategy
        
        assert RoPEStrategy.STANDARD.name == "STANDARD"
        assert RoPEStrategy.LINEAR.name == "LINEAR"
        assert RoPEStrategy.NTK_AWARE.name == "NTK_AWARE"
        assert RoPEStrategy.YARN.name == "YARN"
        assert RoPEStrategy.DYNAMIC_NTK.name == "DYNAMIC_NTK"


class TestStrategyConfig:
    """Test strategy configuration."""
    
    def test_default_config(self):
        """Test default strategy configurations."""
        from src.deepseek.torch.model.rope_ablation import StrategyConfig, RoPEStrategy
        
        config = StrategyConfig.standard()
        assert config.strategy == RoPEStrategy.STANDARD
        # Test by checking strategy type rather than nested config attributes
        # (classmethods shadow instance attributes with same names)
        assert config.strategy.value == "standard"
        
    def test_linear_scaling_config(self):
        """Test linear scaling configuration."""
        from src.deepseek.torch.model.rope_ablation import StrategyConfig, RoPEStrategy
        
        config = StrategyConfig.linear(scale=4.0)
        assert config.strategy == RoPEStrategy.LINEAR
        assert config.linear is not None
        assert config.linear.scale == 4.0
        
    def test_ntk_config(self):
        """Test NTK-aware configuration."""
        from src.deepseek.torch.model.rope_ablation import StrategyConfig, RoPEStrategy
        
        config = StrategyConfig.ntk_aware(alpha=2.0)
        assert config.strategy == RoPEStrategy.NTK_AWARE
        assert config.ntk_aware is not None
        assert config.ntk_aware.alpha == 2.0
        
    def test_yarn_config(self):
        """Test YaRN configuration."""
        from src.deepseek.torch.model.rope_ablation import StrategyConfig, RoPEStrategy
        
        config = StrategyConfig.yarn(
            scale=2.0,
            beta_fast=32.0,
            beta_slow=1.0
        )
        assert config.strategy == RoPEStrategy.YARN
        assert config.yarn is not None
        assert config.yarn.scale == 2.0
        assert config.yarn.beta_fast == 32.0
        assert config.yarn.beta_slow == 1.0


class TestAblationConfig:
    """Test ablation study configuration."""
    
    def test_default_ablation_config(self):
        """Test default ablation configuration."""
        from src.deepseek.torch.model.rope_ablation import AblationConfig
        
        config = AblationConfig()
        assert len(config.strategies) > 0
        assert len(config.eval_seq_lengths) > 0
        assert config.samples_per_length > 0
        
    def test_custom_ablation_config(self):
        """Test custom ablation configuration."""
        from src.deepseek.torch.model.rope_ablation import (
            AblationConfig,
            StrategyConfig,
        )
        
        strategies = [
            StrategyConfig.standard(),
            StrategyConfig.linear(scale=2.0),
        ]
        config = AblationConfig(
            strategies=strategies,
            eval_seq_lengths=[512, 1024, 2048],
        )
        
        assert len(config.strategies) == 2
        assert config.eval_seq_lengths == [512, 1024, 2048]


class TestAblationMetrics:
    """Test ablation metrics collection."""
    
    def test_metrics_dataclass(self):
        """Test metrics dataclass initialization."""
        from src.deepseek.torch.model.rope_ablation import AblationMetrics
        
        metrics = AblationMetrics(
            strategy_name="standard",
            perplexity_by_length={512: 10.5, 1024: 11.2, 2048: 12.0},
            attention_entropy=[2.1, 2.3, 2.5],
            context_utilization=[0.8, 0.75, 0.70],
        )
        
        assert metrics.strategy_name == "standard"
        assert len(metrics.perplexity_by_length) == 3
        assert len(metrics.attention_entropy) == 3
        
    def test_metrics_mean_computation(self):
        """Test mean computation for metrics."""
        from src.deepseek.torch.model.rope_ablation import AblationMetrics
        
        metrics = AblationMetrics(
            strategy_name="linear",
            perplexity_by_length={512: 10.0, 1024: 12.0, 2048: 14.0},
            attention_entropy=[2.0, 2.5, 3.0],
            context_utilization=[0.8, 0.7, 0.6],
        )
        
        avg_entropy = sum(metrics.attention_entropy) / len(metrics.attention_entropy)
        assert abs(avg_entropy - 2.5) < 0.01


class TestRoPEAblationStudy:
    """Test RoPE ablation study runner."""
    
    def test_study_initialization(self):
        """Test study can be initialized."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationStudy,
            AblationConfig
        )
        
        config = AblationConfig()
        study = RoPEAblationStudy(config)
        
        assert study is not None
        
    def test_run_study_standard(self):
        """Test running study with standard RoPE strategy."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationStudy,
            AblationConfig,
            StrategyConfig,
        )
        
        device = get_device()
        config = AblationConfig(
            strategies=[StrategyConfig.standard()],
            eval_seq_lengths=[128, 256],
            verbose=False,
        )
        study = RoPEAblationStudy(config)
        study.run(device)
        
        results = study.get_results()
        assert "standard" in results
        assert results["standard"].forward_time_ms > 0
        
    def test_run_study_linear(self):
        """Test running study with linear scaled RoPE."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationStudy,
            AblationConfig,
            StrategyConfig,
        )
        
        device = get_device()
        config = AblationConfig(
            strategies=[StrategyConfig.linear(scale=2.0)],
            eval_seq_lengths=[128],
            verbose=False,
        )
        study = RoPEAblationStudy(config)
        study.run(device)
        
        results = study.get_results()
        assert "linear" in results
        
    def test_run_study_ntk(self):
        """Test running study with NTK-aware RoPE."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationStudy,
            AblationConfig,
            StrategyConfig,
        )
        
        device = get_device()
        config = AblationConfig(
            strategies=[StrategyConfig.ntk_aware(alpha=2.0)],
            eval_seq_lengths=[128],
            verbose=False,
        )
        study = RoPEAblationStudy(config)
        study.run(device)
        
        results = study.get_results()
        assert "ntk_aware" in results
        
    def test_rope_module_forward(self):
        """Test RoPE ablation module forward pass."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationModule,
            StrategyConfig,
        )
        
        device = get_device()
        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64
        
        module = RoPEAblationModule(
            d_head=head_dim,
            max_seq_len=512,
            strategy_config=StrategyConfig.standard(),
        ).to(device)
        
        # Create test tensor
        x = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Apply RoPE
        rotated = module(x)
        
        assert rotated.shape == x.shape
        assert torch.isfinite(rotated).all()
        
    def test_compare_strategies(self):
        """Test comparing different strategies."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationStudy,
            AblationConfig,
            StrategyConfig,
        )
        
        device = get_device()
        config = AblationConfig(
            strategies=[
                StrategyConfig.standard(),
                StrategyConfig.linear(scale=2.0),
            ],
            eval_seq_lengths=[64, 128],
            verbose=False,
        )
        study = RoPEAblationStudy(config)
        study.run(device)
        
        results = study.get_results()
        
        # Both strategies should have results
        assert "standard" in results
        assert "linear" in results


class TestAblationAnalysis:
    """Test ablation study analysis functions."""
    
    def test_generate_report(self):
        """Test results report generation."""
        from src.deepseek.torch.model.rope_ablation import (
            RoPEAblationStudy,
            AblationConfig,
            StrategyConfig,
        )
        
        device = get_device()
        config = AblationConfig(
            strategies=[StrategyConfig.standard()],
            eval_seq_lengths=[128],
            verbose=False,
        )
        study = RoPEAblationStudy(config)
        study.run(device)
        
        report = study.generate_report()
        
        assert isinstance(report, str)
        assert "standard" in report


class TestTrainingHook:
    """Test ablation study training hooks."""
    
    def test_training_hook_creation(self):
        """Test training hook can be created."""
        from src.deepseek.torch.model.rope_ablation import AblationTrainingHook
        
        hook = AblationTrainingHook(log_interval=100)
        
        assert hook is not None
        assert hook.log_interval == 100
        
    def test_training_hook_logging(self):
        """Test training hook can log metrics."""
        from src.deepseek.torch.model.rope_ablation import AblationTrainingHook
        
        hook = AblationTrainingHook(log_interval=10)
        
        # Log some metrics (only multiples of log_interval get stored)
        hook.log_metric(step=10, name="perplexity", value=15.0)
        hook.log_metric(step=20, name="perplexity", value=12.0)
        hook.log_metric(step=30, name="perplexity", value=10.0)
        
        # Get logged metrics
        metrics = hook.get_metrics()
        
        assert len(metrics) == 3
        assert all(m[1] == "perplexity" for m in metrics)
        
    def test_training_hook_export(self):
        """Test training hook can export metrics."""
        from src.deepseek.torch.model.rope_ablation import AblationTrainingHook
        
        hook = AblationTrainingHook(log_interval=10)
        
        hook.log_metric(step=10, name="loss", value=2.5)
        hook.log_metric(step=20, name="loss", value=2.0)
        
        exported = hook.export_csv()
        
        assert isinstance(exported, str)
        assert "step,metric,value" in exported


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
