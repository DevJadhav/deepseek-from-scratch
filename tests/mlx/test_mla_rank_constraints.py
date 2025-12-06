"""
Tests for MLA Rank Constraints in MLX.

Tests SVD-based initialization, rank regularization loss,
numerical stability checks, and gradient clipping for MLA.
"""

import pytest

# Check if MLX is available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX is not installed"
)


class TestRankConstraintConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.deepseek.mlx.mla_rank_constraints import RankConstraintConfig
        
        config = RankConstraintConfig()
        assert config.use_svd_init
        assert config.svd_rank_ratio == 1.0
        assert config.rank_regularization_weight == 0.01
        assert config.target_rank is None  # None by default
        assert config.condition_number_threshold == 1e4
        assert config.min_singular_value == 1e-6
        assert config.latent_grad_clip_norm == 1.0
        assert config.use_adaptive_grad_clip
        
    def test_production_config(self):
        """Test production configuration values."""
        from src.deepseek.mlx.mla_rank_constraints import RankConstraintConfig
        
        config = RankConstraintConfig.production()
        assert config.use_svd_init
        assert config.svd_rank_ratio == 0.95
        assert config.rank_regularization_weight == 0.001
        assert config.condition_number_threshold == 1e4
        
    def test_research_config(self):
        """Test research configuration values."""
        from src.deepseek.mlx.mla_rank_constraints import RankConstraintConfig
        
        config = RankConstraintConfig.research()
        assert config.svd_rank_ratio == 0.9
        assert config.rank_regularization_weight == 0.01
        assert config.condition_number_threshold == 1e3


class TestSVDInitializer:
    """Test SVD-based initialization for low-rank matrices."""
    
    def test_svd_low_rank_init(self):
        """Test that SVD initializer creates properly shaped matrices."""
        from src.deepseek.mlx.mla_rank_constraints import SVDInitializer
        
        # Initialize a projection matrix
        matrix = SVDInitializer.initialize_low_rank(
            shape=(512, 768),
            target_rank=32
        )
        
        assert matrix.shape == (512, 768)
        assert mx.all(mx.isfinite(matrix))
        
    def test_svd_paired_projections(self):
        """Test SVD initialization for paired down/up projections."""
        from src.deepseek.mlx.mla_rank_constraints import SVDInitializer
        
        d_model = 768
        d_latent = 64
        d_out = 768
        
        down_proj, up_proj = SVDInitializer.initialize_paired_projections(
            d_model=d_model,
            d_latent=d_latent,
            d_out=d_out
        )
        
        assert down_proj.shape == (d_latent, d_model)
        assert up_proj.shape == (d_out, d_latent)
        assert mx.all(mx.isfinite(down_proj))
        assert mx.all(mx.isfinite(up_proj))


class TestRankRegularizationLoss:
    """Test rank regularization loss computation."""
    
    def test_regularization_loss_computation(self):
        """Test that regularization loss can be computed."""
        from src.deepseek.mlx.mla_rank_constraints import RankRegularizationLoss
        
        regularizer = RankRegularizationLoss(target_rank=32, weight=0.01)
        
        # Create a test projection matrix
        matrix = mx.random.normal(shape=(512, 768))
        
        loss = regularizer(matrix)
        
        assert loss.shape == ()
        assert float(loss) >= 0
        assert mx.isfinite(loss)
        
    def test_low_rank_matrix_has_low_loss(self):
        """Test that low-rank matrices have lower loss than full-rank."""
        from src.deepseek.mlx.mla_rank_constraints import RankRegularizationLoss
        
        regularizer = RankRegularizationLoss(target_rank=32, weight=0.01)
        
        # Create a low-rank matrix (rank 32) with normalized scale
        U = mx.random.normal(shape=(512, 32)) / 32
        V = mx.random.normal(shape=(32, 768)) / 32
        low_rank_matrix = U @ V
        
        # Create a full-rank matrix with similar scale
        full_rank_matrix = mx.random.normal(shape=(512, 768)) / 32
        
        low_rank_loss = float(regularizer(low_rank_matrix))
        full_rank_loss = float(regularizer(full_rank_matrix))
        
        # Low-rank matrix should have lower regularization loss
        assert low_rank_loss < full_rank_loss


class TestNumericalStabilityChecker:
    """Test numerical stability checks for projection matrices."""
    
    def test_stability_check_identity(self):
        """Test stability check on identity matrix (well-conditioned)."""
        from src.deepseek.mlx.mla_rank_constraints import (
            NumericalStabilityChecker,
            StabilityStatus
        )
        
        checker = NumericalStabilityChecker(
            condition_threshold=1000.0,
            min_singular_value=1e-6
        )
        
        # Identity matrix has condition number 1
        identity = mx.eye(64)
        report = checker.check(identity, "identity")
        
        assert report.status == StabilityStatus.OK
        assert report.condition_number < 10.0  # Should be close to 1
        
    def test_stability_check_history(self):
        """Test that stability checker maintains history."""
        from src.deepseek.mlx.mla_rank_constraints import NumericalStabilityChecker
        
        checker = NumericalStabilityChecker()
        
        # Check multiple matrices
        for _ in range(5):
            matrix = mx.random.normal(shape=(64, 64))
            checker.check(matrix, "test_matrix")
        
        history = checker.history.get("test_matrix")
        assert history is not None
        assert len(history) == 5


class TestLatentProjectionGradientClipper:
    """Test gradient clipping for latent projection weights."""
    
    def test_gradient_clip_coefficient(self):
        """Test gradient clipping coefficient computation."""
        from src.deepseek.mlx.mla_rank_constraints import LatentProjectionGradientClipper
        
        clipper = LatentProjectionGradientClipper(max_norm=1.0, adaptive=False)
        
        # Large gradient norm should trigger clipping
        stats = clipper.compute_clip_coefficient(grad_norm=10.0)
        
        assert stats.was_clipped
        assert stats.clip_coefficient < 1.0
        
        # Small gradient norm should not trigger clipping
        stats = clipper.compute_clip_coefficient(grad_norm=0.5)
        assert not stats.was_clipped
        assert stats.clip_coefficient == 1.0
        
    def test_adaptive_clipping(self):
        """Test adaptive gradient clipping."""
        from src.deepseek.mlx.mla_rank_constraints import LatentProjectionGradientClipper
        
        clipper = LatentProjectionGradientClipper(
            max_norm=1.0, 
            adaptive=True, 
            warmup_steps=5
        )
        
        # Simulate warmup
        for _ in range(10):
            clipper.compute_clip_coefficient(grad_norm=5.0)
        
        # After warmup, clip threshold should adapt
        assert clipper.grad_norm_ema is not None
        assert clipper.grad_norm_ema > 0


class TestMLARankConstraintManager:
    """Test the unified MLA rank constraint manager."""
    
    def test_manager_initialization(self):
        """Test manager can be initialized."""
        from src.deepseek.mlx.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        config = RankConstraintConfig()
        manager = MLARankConstraintManager(config)
        
        assert manager is not None
        assert manager.config == config
        
    def test_manager_initialize_weights(self):
        """Test manager can initialize MLA weights."""
        from src.deepseek.mlx.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        config = RankConstraintConfig(target_rank=64)
        manager = MLARankConstraintManager(config)
        
        weights = manager.initialize_weights(
            d_model=768,
            d_latent=64
        )
        
        assert weights.kv_down.shape == (64, 768)
        assert weights.k_up.shape == (768, 64)
        assert weights.v_up.shape == (768, 64)
        
    def test_manager_compute_regularization(self):
        """Test manager can compute regularization loss."""
        from src.deepseek.mlx.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        config = RankConstraintConfig(
            target_rank=64,
            rank_regularization_weight=0.01
        )
        manager = MLARankConstraintManager(config)
        
        # Initialize weights first
        weights = manager.initialize_weights(
            d_model=768,
            d_latent=64
        )
        
        loss = manager.compute_rank_regularization_loss(
            kv_down=weights.kv_down,
            k_up=weights.k_up,
            v_up=weights.v_up
        )
        
        assert float(loss) >= 0
        
    def test_manager_stability_report(self):
        """Test manager can generate stability report."""
        from src.deepseek.mlx.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        config = RankConstraintConfig()
        manager = MLARankConstraintManager(config)
        
        # Create test weights
        kv_down = mx.eye(64, 768)[:64, :]
        k_up = mx.eye(768, 64)[:, :64]
        v_up = mx.eye(768, 64)[:, :64]
        
        report = manager.check_stability(kv_down, k_up, v_up)
        
        assert report.kv_down is not None
        assert report.k_up is not None
        assert report.v_up is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
