"""
Tests for MLA Rank Constraints in PyTorch.

Tests SVD-based initialization, rank regularization loss,
numerical stability checks, and gradient clipping for MLA.
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple


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


class TestRankConstraintConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.deepseek.torch.model.mla_rank_constraints import RankConstraintConfig
        
        config = RankConstraintConfig()
        assert config.use_svd_init == True
        assert config.svd_rank_ratio == 1.0
        assert config.rank_regularization_weight == 0.01
        assert config.target_rank is None  # None by default, uses d_latent
        assert config.condition_number_threshold == 1e4
        assert config.min_singular_value == 1e-6
        assert config.latent_grad_clip_norm == 1.0
        assert config.use_adaptive_grad_clip == True
        
    def test_production_config(self):
        """Test production configuration values."""
        from src.deepseek.torch.model.mla_rank_constraints import RankConstraintConfig
        
        config = RankConstraintConfig.production()
        assert config.use_svd_init == True
        assert config.svd_rank_ratio == 0.95
        assert config.rank_regularization_weight == 0.001
        assert config.condition_number_threshold == 1e4
        
    def test_research_config(self):
        """Test research configuration values."""
        from src.deepseek.torch.model.mla_rank_constraints import RankConstraintConfig
        
        config = RankConstraintConfig.research()
        assert config.svd_rank_ratio == 0.9
        assert config.rank_regularization_weight == 0.01
        assert config.condition_number_threshold == 1e3


class TestSVDInitializer:
    """Test SVD-based initialization for low-rank matrices."""
    
    def test_svd_low_rank_init(self):
        """Test that SVD initializer creates properly shaped matrices."""
        from src.deepseek.torch.model.mla_rank_constraints import SVDInitializer
        
        device = get_device()
        
        # Initialize a projection matrix
        matrix = SVDInitializer.initialize_low_rank(
            shape=(512, 768),
            target_rank=32,
            device=device
        )
        
        assert matrix.shape == (512, 768)
        assert matrix.device.type == device.type
        assert torch.isfinite(matrix).all()
        
    def test_svd_rank_constraint(self):
        """Test that SVD initializer respects rank constraint."""
        from src.deepseek.torch.model.mla_rank_constraints import SVDInitializer
        
        device = get_device()
        target_rank = 32
        
        matrix = SVDInitializer.initialize_low_rank(
            shape=(512, 768),
            target_rank=target_rank,
            device=device
        )
        
        # Check effective rank via SVD
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # Most singular values after target_rank should be small
        assert S[:target_rank].sum() > S[target_rank:].sum() * 10
        
    def test_svd_paired_projections(self):
        """Test SVD initialization for paired down/up projections."""
        from src.deepseek.torch.model.mla_rank_constraints import SVDInitializer
        
        device = get_device()
        
        d_model = 768
        d_latent = 64
        d_out = 768
        
        down_proj, up_proj = SVDInitializer.initialize_paired_projections(
            d_model=d_model,
            d_latent=d_latent,
            d_out=d_out,
            device=device
        )
        
        assert down_proj.shape == (d_latent, d_model)
        assert up_proj.shape == (d_out, d_latent)
        assert torch.isfinite(down_proj).all()
        assert torch.isfinite(up_proj).all()
        
        # Verify composition preserves approximate rank
        composed = up_proj @ down_proj
        U, S, Vh = torch.linalg.svd(composed, full_matrices=False)
        # Most energy should be in top singular values
        total_energy = (S ** 2).sum()
        top_energy = (S[:d_latent] ** 2).sum()
        assert top_energy / total_energy > 0.9


class TestRankRegularizationLoss:
    """Test rank regularization loss computation."""
    
    def test_regularization_loss_computation(self):
        """Test that regularization loss can be computed."""
        from src.deepseek.torch.model.mla_rank_constraints import RankRegularizationLoss
        
        device = get_device()
        regularizer = RankRegularizationLoss(target_rank=32, weight=0.01)
        
        # Create a test projection matrix
        matrix = torch.randn(512, 768, device=device)
        
        loss = regularizer(matrix)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_low_rank_matrix_has_low_loss(self):
        """Test that low-rank matrices have lower loss than full-rank."""
        from src.deepseek.torch.model.mla_rank_constraints import RankRegularizationLoss
        
        device = get_device()
        regularizer = RankRegularizationLoss(target_rank=32, weight=0.01)
        
        # Create a low-rank matrix (rank 32) with normalized scale
        U = torch.randn(512, 32, device=device) / 32
        V = torch.randn(32, 768, device=device) / 32
        low_rank_matrix = U @ V
        
        # Create a full-rank matrix with similar scale
        full_rank_matrix = torch.randn(512, 768, device=device) / 32
        
        low_rank_loss = regularizer(low_rank_matrix)
        full_rank_loss = regularizer(full_rank_matrix)
        
        # Low-rank matrix should have lower regularization loss
        assert low_rank_loss < full_rank_loss


class TestNumericalStabilityChecker:
    """Test numerical stability checks for projection matrices."""
    
    def test_stability_check_identity(self):
        """Test stability check on identity matrix (well-conditioned)."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            NumericalStabilityChecker,
            StabilityStatus
        )
        
        device = get_device()
        checker = NumericalStabilityChecker(
            condition_threshold=1000.0,
            min_singular_value=1e-6
        )
        
        # Identity matrix has condition number 1
        identity = torch.eye(64, device=device)
        report = checker.check(identity, "identity")
        
        assert report.status == StabilityStatus.OK
        assert report.condition_number < 10.0  # Should be close to 1
        
    def test_stability_check_ill_conditioned(self):
        """Test stability check on ill-conditioned matrix."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            NumericalStabilityChecker,
            StabilityStatus
        )
        
        device = get_device()
        checker = NumericalStabilityChecker(
            condition_threshold=100.0,
            min_singular_value=1e-6
        )
        
        # Create an ill-conditioned matrix
        matrix = torch.diag(torch.tensor([1.0, 1e-6], device=device))
        report = checker.check(matrix, "ill_conditioned")
        
        assert report.status in [StabilityStatus.WARNING, StabilityStatus.CORRECTED]
        assert report.condition_number > 100.0
        
    def test_stability_check_history(self):
        """Test that stability checker maintains history."""
        from src.deepseek.torch.model.mla_rank_constraints import NumericalStabilityChecker
        
        device = get_device()
        checker = NumericalStabilityChecker()
        
        # Check multiple matrices
        for i in range(5):
            matrix = torch.randn(64, 64, device=device)
            checker.check(matrix, "test_matrix")
        
        history = checker.get_history("test_matrix")
        assert history is not None
        assert len(history) == 5


class TestLatentProjectionGradientClipper:
    """Test gradient clipping for latent projection weights."""
    
    def test_gradient_clip_coefficient(self):
        """Test gradient clipping coefficient computation."""
        from src.deepseek.torch.model.mla_rank_constraints import LatentProjectionGradientClipper
        
        clipper = LatentProjectionGradientClipper(max_norm=1.0, adaptive=False)
        
        # Large gradient norm should trigger clipping
        stats = clipper.compute_clip_coefficient(grad_norm=10.0)
        
        assert stats.was_clipped
        assert stats.clip_coefficient < 1.0
        
        # Small gradient norm should not trigger clipping
        stats = clipper.compute_clip_coefficient(grad_norm=0.5)
        assert not stats.was_clipped
        assert stats.clip_coefficient == 1.0
        
    def test_gradient_clip_parameters(self):
        """Test gradient clipping on parameters."""
        from src.deepseek.torch.model.mla_rank_constraints import LatentProjectionGradientClipper
        
        device = get_device()
        clipper = LatentProjectionGradientClipper(max_norm=1.0, adaptive=False)
        
        # Create layers with large gradients
        layer = nn.Linear(768, 512).to(device)
        layer.weight.grad = torch.randn(512, 768, device=device) * 100
        
        original_norm = layer.weight.grad.norm().item()
        
        # Clip gradients
        stats = clipper.clip_gradients([layer.weight])
        
        clipped_norm = layer.weight.grad.norm().item()
        
        assert stats.was_clipped
        assert clipped_norm < original_norm
        
    def test_adaptive_clipping(self):
        """Test adaptive gradient clipping."""
        from src.deepseek.torch.model.mla_rank_constraints import LatentProjectionGradientClipper
        
        clipper = LatentProjectionGradientClipper(
            max_norm=1.0, 
            adaptive=True, 
            warmup_steps=5
        )
        
        # Simulate warmup
        for i in range(10):
            stats = clipper.compute_clip_coefficient(grad_norm=5.0)
        
        # After warmup, clip threshold should adapt
        assert clipper.grad_norm_ema is not None
        assert clipper.grad_norm_ema > 0


class TestMLARankConstraintManager:
    """Test the unified MLA rank constraint manager."""
    
    def test_manager_initialization(self):
        """Test manager can be initialized."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        config = RankConstraintConfig()
        manager = MLARankConstraintManager(config)
        
        assert manager is not None
        assert manager.config == config
        
    def test_manager_initialize_weights(self):
        """Test manager can initialize MLA weights."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        device = get_device()
        config = RankConstraintConfig(target_rank=64)
        manager = MLARankConstraintManager(config)
        
        weights = manager.initialize_weights(
            d_model=768,
            d_latent=64,
            device=device
        )
        
        assert weights.kv_down.shape == (64, 768)
        assert weights.k_up.shape == (768, 64)
        assert weights.v_up.shape == (768, 64)
        
    def test_manager_compute_regularization(self):
        """Test manager can compute regularization loss."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        device = get_device()
        config = RankConstraintConfig(
            target_rank=64,
            rank_regularization_weight=0.01
        )
        manager = MLARankConstraintManager(config)
        
        # Initialize weights first to set up rank regularization
        weights = manager.initialize_weights(
            d_model=768,
            d_latent=64,
            device=device
        )
        
        loss = manager.compute_rank_regularization_loss(
            kv_down=weights.kv_down,
            k_up=weights.k_up,
            v_up=weights.v_up
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
    def test_manager_stability_report(self):
        """Test manager can generate stability report."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager,
            StabilityStatus
        )
        
        device = get_device()
        config = RankConstraintConfig()
        manager = MLARankConstraintManager(config)
        
        # Create test weights
        kv_down = torch.eye(64, 768, device=device)[:64, :]
        k_up = torch.eye(768, 64, device=device)[:, :64]
        v_up = torch.eye(768, 64, device=device)[:, :64]
        
        report = manager.check_stability(kv_down, k_up, v_up)
        
        assert report.kv_down is not None
        assert report.k_up is not None
        assert report.v_up is not None


class TestIntegration:
    """Integration tests for rank constraints with MLA."""
    
    def test_end_to_end_training_step(self):
        """Test complete training step with rank constraints."""
        from src.deepseek.torch.model.mla_rank_constraints import (
            RankConstraintConfig,
            MLARankConstraintManager
        )
        
        device = get_device()
        config = RankConstraintConfig(
            target_rank=32,
            rank_regularization_weight=0.01
        )
        manager = MLARankConstraintManager(config)
        
        # Initialize projections
        d_model = 512
        d_latent = 32
        
        down_proj = nn.Linear(d_model, d_latent, bias=False).to(device)
        up_proj = nn.Linear(d_latent, d_model, bias=False).to(device)
        
        # Initialize with manager
        weights = manager.initialize_weights(
            d_model=d_model,
            d_latent=d_latent,
            device=device
        )
        
        with torch.no_grad():
            down_proj.weight.copy_(weights.kv_down)
            # k_up is (d_model, d_latent), up_proj.weight is (d_model, d_latent)
            up_proj.weight.copy_(weights.k_up)
        
        # Forward pass
        x = torch.randn(2, 16, d_model, device=device)
        h = down_proj(x)
        y = up_proj(h)
        
        # Compute losses
        reconstruction_loss = (y - x).pow(2).mean()
        
        reg_loss = manager.compute_rank_regularization_loss(
            kv_down=down_proj.weight,
            k_up=up_proj.weight.t(),  # Back to original shape
            v_up=weights.v_up
        )
        
        total_loss = reconstruction_loss + reg_loss
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        stats = manager.clip_latent_gradients([down_proj.weight, up_proj.weight])
        
        # Check stability
        report = manager.check_stability(
            down_proj.weight, 
            up_proj.weight.t(), 
            weights.v_up
        )
        
        assert report.kv_down is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
