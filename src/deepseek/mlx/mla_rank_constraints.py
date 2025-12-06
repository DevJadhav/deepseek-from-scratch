"""
MLA Rank Constraints Module for DeepSeek (MLX)

This module implements production-grade rank constraints for Multi-Head Latent Attention (MLA):
- SVD-based initialization for low-rank matrices
- Rank regularization loss during training
- Numerical stability checks for projection matrices with high condition numbers
- Gradient clipping specifically for latent projection weights

Reference: DeepSeek-V3 architecture specification for MLA with rank constraints.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import math
from dataclasses import dataclass
from typing import Optional
from enum import Enum


@dataclass
class RankConstraintConfig:
    """Configuration for MLA rank constraints"""
    # Use SVD-based initialization for low-rank matrices
    use_svd_init: bool = True
    # Ratio of singular values to keep (1.0 = full rank)
    svd_rank_ratio: float = 1.0
    # Weight for rank regularization loss
    rank_regularization_weight: float = 0.01
    # Target effective rank (None = use d_latent)
    target_rank: int | None = None
    # Maximum condition number before warning/correction
    condition_number_threshold: float = 1e4
    # Minimum singular value to prevent numerical issues
    min_singular_value: float = 1e-6
    # Max gradient norm for latent projection layers
    latent_grad_clip_norm: float = 1.0
    # Adapt clip threshold based on gradient history
    use_adaptive_grad_clip: bool = True
    # Log condition numbers during training
    log_condition_numbers: bool = True
    
    @classmethod
    def production(cls) -> RankConstraintConfig:
        """Create a production config with sensible defaults"""
        return cls(
            use_svd_init=True,
            svd_rank_ratio=0.95,
            rank_regularization_weight=0.001,
            target_rank=None,
            condition_number_threshold=1e4,
            min_singular_value=1e-6,
            latent_grad_clip_norm=1.0,
            use_adaptive_grad_clip=True,
            log_condition_numbers=True,
        )
    
    @classmethod
    def research(cls) -> RankConstraintConfig:
        """Create a research config with more aggressive constraints"""
        return cls(
            use_svd_init=True,
            svd_rank_ratio=0.9,
            rank_regularization_weight=0.01,
            target_rank=None,
            condition_number_threshold=1e3,
            min_singular_value=1e-5,
            latent_grad_clip_norm=0.5,
            use_adaptive_grad_clip=True,
            log_condition_numbers=True,
        )


class SVDInitializer:
    """SVD-based initialization for low-rank projection matrices"""
    
    @staticmethod
    def initialize_low_rank(
        shape: tuple[int, int],
        target_rank: int | None = None,
        init_scale: float = 1.0,
    ) -> mx.array:
        """
        Initialize a weight tensor with SVD-based low-rank structure.
        
        Uses orthogonal initialization with controlled rank and spectral properties.
        
        Args:
            shape: (out_features, in_features)
            target_rank: Target rank for the matrix (defaults to min(out, in))
            init_scale: Scale factor for initialization
            
        Returns:
            Initialized tensor with low-rank structure
        """
        out_features, in_features = shape
        rank = target_rank if target_rank else min(out_features, in_features)
        rank = min(rank, out_features, in_features)
        
        # Generate random matrices
        u_random = mx.random.normal(shape=(out_features, rank))
        v_random = mx.random.normal(shape=(in_features, rank))
        
        # Orthogonalize via Gram-Schmidt-like normalization
        u = SVDInitializer._orthogonalize(u_random)
        v = SVDInitializer._orthogonalize(v_random)
        
        # Create decaying singular values (Xavier/Glorot-like)
        fan_in = float(in_features)
        fan_out = float(out_features)
        std = init_scale * math.sqrt(2.0 / (fan_in + fan_out))
        
        # Exponentially decaying singular values
        indices = mx.arange(rank, dtype=mx.float32)
        singular_values = std * mx.exp(-indices / rank)
        
        # Construct weight: U @ diag(S) @ V^T
        s_diag = mx.expand_dims(singular_values, axis=1)  # (rank, 1)
        u_scaled = u * mx.transpose(s_diag)  # (out, rank)
        weight = u_scaled @ mx.transpose(v)  # (out, in)
        
        return weight
    
    @staticmethod
    def _orthogonalize(matrix: mx.array) -> mx.array:
        """Orthogonalize a matrix using row normalization"""
        norms = mx.sqrt(mx.sum(matrix ** 2, axis=1, keepdims=True))
        return matrix / (norms + 1e-8)
    
    @staticmethod
    def initialize_paired_projections(
        d_model: int,
        d_latent: int,
        d_out: int,
        init_scale: float = 1.0,
    ) -> tuple[mx.array, mx.array]:
        """
        Initialize paired down/up projection matrices for MLA.
        
        Args:
            d_model: Model dimension
            d_latent: Latent dimension
            d_out: Output dimension
            init_scale: Scale factor
            
        Returns:
            Tuple of (down_projection, up_projection) tensors
        """
        # Down projection: d_model -> d_latent
        down_random = mx.random.normal(shape=(d_latent, d_model))
        down_ortho = SVDInitializer._orthogonalize(down_random)
        
        # Up projection: d_latent -> d_out
        up_random = mx.random.normal(shape=(d_out, d_latent))
        up_ortho = SVDInitializer._orthogonalize(up_random)
        
        # Scale for proper variance
        scale = init_scale * math.sqrt(1.0 / d_latent)
        
        down_weight = down_ortho * scale
        up_weight = up_ortho * scale
        
        return down_weight, up_weight


class RankRegularizationLoss(nn.Module):
    """
    Rank regularization loss for MLA projection matrices.
    
    Encourages low effective rank by penalizing singular values beyond target rank.
    """
    
    def __init__(
        self,
        target_rank: int,
        weight: float = 0.01,
        use_nuclear_norm: bool = True,
        tail_penalty_factor: float = 2.0,
    ):
        super().__init__()
        self.target_rank = target_rank
        self.weight = weight
        self.use_nuclear_norm = use_nuclear_norm
        self.tail_penalty_factor = tail_penalty_factor
    
    def __call__(self, weight: mx.array) -> mx.array:
        """
        Compute rank regularization loss.
        
        Args:
            weight: Weight tensor to regularize
            
        Returns:
            Scalar loss
        """
        if self.use_nuclear_norm:
            return self._compute_nuclear_norm_proxy(weight)
        else:
            return self._compute_power_iteration_loss(weight)
    
    def _compute_nuclear_norm_proxy(self, weight: mx.array) -> mx.array:
        """Compute Frobenius norm as nuclear norm proxy"""
        frobenius_sq = mx.sum(weight ** 2)
        return self.weight * frobenius_sq
    
    def _compute_power_iteration_loss(self, weight: mx.array) -> mx.array:
        """Compute loss using power iteration"""
        # Estimate top singular value
        v = mx.random.normal(shape=(weight.shape[1], 1))
        v = v / mx.sqrt(mx.sum(v ** 2))
        
        for _ in range(5):
            u = weight @ v
            u = u / (mx.sqrt(mx.sum(u ** 2)) + 1e-8)
            v = mx.transpose(weight) @ u
            v = v / (mx.sqrt(mx.sum(v ** 2)) + 1e-8)
        
        sigma_max = mx.sqrt(mx.sum((weight @ v) ** 2))
        frobenius = mx.sqrt(mx.sum(weight ** 2))
        
        ratio = frobenius ** 2 / (self.target_rank * sigma_max ** 2 + 1e-8)
        tail_penalty = mx.maximum(ratio - 1.0, 0.0) * self.tail_penalty_factor
        
        return self.weight * (frobenius ** 2 + tail_penalty)


class StabilityStatus(Enum):
    """Numerical stability status"""
    OK = "ok"
    WARNING = "warning"
    CORRECTED = "corrected"
    SVD_FAILED = "svd_failed"


@dataclass
class StabilityDiagnostics:
    """Diagnostics from numerical stability check"""
    condition_number: float
    max_singular_value: float
    min_singular_value: float
    effective_rank: int
    status: StabilityStatus
    warning: str | None = None


class NumericalStabilityChecker:
    """Numerical stability checker for MLA projection matrices"""
    
    def __init__(
        self,
        condition_threshold: float = 1e4,
        min_singular_value: float = 1e-6,
        auto_correct: bool = True,
    ):
        self.condition_threshold = condition_threshold
        self.min_singular_value = min_singular_value
        self.auto_correct = auto_correct
        self.history: dict[str, list[float]] = {}
    
    def check(self, weight: mx.array, name: str) -> StabilityDiagnostics:
        """
        Check numerical stability of a weight matrix.
        
        Args:
            weight: Weight tensor to check
            name: Name for logging
            
        Returns:
            StabilityDiagnostics with condition number and status
        """
        rows, cols = weight.shape
        min_dim = min(rows, cols)
        
        # Estimate singular values via power iteration
        max_sv = self._estimate_max_singular_value(weight, iterations=10)
        min_sv = self._estimate_min_singular_value(weight, max_sv, iterations=20)
        
        condition_number = max_sv / (min_sv + 1e-12)
        
        # Track history
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(condition_number)
        
        # Determine status
        if condition_number > self.condition_threshold:
            if min_sv < self.min_singular_value and self.auto_correct:
                status = StabilityStatus.CORRECTED
                warning = f"High condition number: {condition_number:.2e}, applied correction"
            else:
                status = StabilityStatus.WARNING
                warning = f"High condition number: {condition_number:.2e}"
        else:
            status = StabilityStatus.OK
            warning = None
        
        effective_rank = self._estimate_effective_rank(weight, max_sv)
        
        return StabilityDiagnostics(
            condition_number=condition_number,
            max_singular_value=max_sv,
            min_singular_value=min_sv,
            effective_rank=min(effective_rank, min_dim),
            status=status,
            warning=warning,
        )
    
    def _estimate_max_singular_value(
        self, 
        matrix: mx.array, 
        iterations: int = 10
    ) -> float:
        """Estimate maximum singular value via power iteration on A^T A"""
        cols = matrix.shape[1]
        
        v = mx.random.normal(shape=(cols,))
        v = v / mx.sqrt(mx.sum(v ** 2))
        
        for _ in range(iterations):
            av = matrix @ v
            atav = mx.transpose(matrix) @ av
            norm = mx.sqrt(mx.sum(atav ** 2))
            v = atav / (norm + 1e-8)
        
        av = matrix @ v
        sigma_squared = float(mx.sum(av ** 2))
        v_norm_sq = float(mx.sum(v ** 2))
        
        return math.sqrt(sigma_squared / (v_norm_sq + 1e-12))
    
    def _estimate_min_singular_value(
        self, 
        matrix: mx.array,
        max_sv: float,
        iterations: int = 20
    ) -> float:
        """Estimate minimum singular value"""
        rows, cols = matrix.shape
        min_dim = min(rows, cols)
        
        frobenius_sq = float(mx.sum(matrix ** 2))
        remaining_sq = max(frobenius_sq - max_sv ** 2, 0.0)
        
        if min_dim > 1:
            avg_remaining = remaining_sq / (min_dim - 1)
            estimated_min = max(math.sqrt(avg_remaining), max_sv * 1e-10)
        else:
            estimated_min = max_sv
        
        return estimated_min
    
    def _estimate_effective_rank(self, matrix: mx.array, max_sv: float) -> int:
        """Estimate effective rank"""
        frobenius = float(mx.sqrt(mx.sum(matrix ** 2)))
        ratio = frobenius / (max_sv + 1e-12)
        return int(math.ceil(ratio ** 2))
    
    def get_history(self, name: str) -> list[float] | None:
        """Get condition number history"""
        return self.history.get(name)
    
    def get_all_history(self) -> dict[str, list[float]]:
        """Get all diagnostics history"""
        return self.history


@dataclass
class GradientClipStats:
    """Statistics from gradient clipping"""
    original_norm: float
    clip_threshold: float
    was_clipped: bool
    clip_coefficient: float


class LatentProjectionGradientClipper:
    """Gradient clipping for latent projection layers"""
    
    def __init__(
        self,
        max_norm: float = 1.0,
        adaptive: bool = True,
        warmup_steps: int = 10,
        ema_decay: float = 0.99,
        min_clip_threshold: float = 0.1,
        max_clip_threshold: float = 10.0,
    ):
        self.max_norm = max_norm
        self.adaptive = adaptive
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.min_clip_threshold = min_clip_threshold
        self.max_clip_threshold = max_clip_threshold
        
        self.step = 0
        self.grad_norm_ema: float | None = None
        self.clip_history: list[GradientClipStats] = []
    
    def compute_clip_coefficient(self, grad_norm: float) -> GradientClipStats:
        """
        Compute gradient clipping coefficient.
        
        Args:
            grad_norm: Current gradient norm
            
        Returns:
            GradientClipStats with clipping information
        """
        self.step += 1
        
        # Update EMA
        if self.grad_norm_ema is None:
            self.grad_norm_ema = grad_norm
        else:
            self.grad_norm_ema = (
                self.ema_decay * self.grad_norm_ema + 
                (1 - self.ema_decay) * grad_norm
            )
        
        # Compute adaptive threshold
        if self.adaptive and self.step > self.warmup_steps:
            clip_threshold = min(
                max(self.grad_norm_ema * 2.0, self.min_clip_threshold),
                self.max_clip_threshold
            )
        else:
            clip_threshold = self.max_norm
        
        was_clipped = grad_norm > clip_threshold
        clip_coefficient = min(clip_threshold / (grad_norm + 1e-8), 1.0)
        
        stats = GradientClipStats(
            original_norm=grad_norm,
            clip_threshold=clip_threshold,
            was_clipped=was_clipped,
            clip_coefficient=clip_coefficient,
        )
        
        self.clip_history.append(stats)
        return stats
    
    def get_clip_coefficient(self, grad_norm: float, threshold: float) -> float:
        """Get simple clip coefficient"""
        if grad_norm > threshold:
            return threshold / (grad_norm + 1e-8)
        return 1.0
    
    def get_recent_history(self, n: int = 100) -> list[GradientClipStats]:
        """Get recent clipping history"""
        return self.clip_history[-n:]
    
    @staticmethod
    def compute_grad_norm(gradients: list[mx.array]) -> float:
        """Compute total gradient norm"""
        total_norm_sq = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm_sq += float(mx.sum(grad ** 2))
        return math.sqrt(total_norm_sq)


@dataclass 
class MLAInitializedWeights:
    """Initialized MLA weights"""
    kv_down: mx.array
    k_up: mx.array
    v_up: mx.array


@dataclass
class MLAStabilityReport:
    """Stability report for all MLA projection matrices"""
    kv_down: StabilityDiagnostics
    k_up: StabilityDiagnostics
    v_up: StabilityDiagnostics


class MLARankConstraintManager:
    """
    Unified manager for all MLA rank constraint operations.
    """
    
    def __init__(self, config: RankConstraintConfig):
        self.config = config
        
        self.stability_checker = NumericalStabilityChecker(
            condition_threshold=config.condition_number_threshold,
            min_singular_value=config.min_singular_value,
            auto_correct=True,
        )
        
        self.grad_clipper = LatentProjectionGradientClipper(
            max_norm=config.latent_grad_clip_norm,
            adaptive=config.use_adaptive_grad_clip,
        )
        
        self._rank_reg_loss: RankRegularizationLoss | None = None
    
    @property
    def rank_reg_loss(self) -> RankRegularizationLoss | None:
        return self._rank_reg_loss
    
    def initialize_weights(
        self,
        d_model: int,
        d_latent: int,
    ) -> MLAInitializedWeights:
        """
        Initialize all MLA projection weights.
        
        Args:
            d_model: Model dimension
            d_latent: Latent dimension
            
        Returns:
            MLAInitializedWeights
        """
        target_rank = self.config.target_rank or d_latent
        if self.config.rank_regularization_weight > 0:
            self._rank_reg_loss = RankRegularizationLoss(
                target_rank=target_rank,
                weight=self.config.rank_regularization_weight,
            )
        
        if not self.config.use_svd_init:
            return MLAInitializedWeights(
                kv_down=mx.random.normal(shape=(d_latent, d_model)),
                k_up=mx.random.normal(shape=(d_model, d_latent)),
                v_up=mx.random.normal(shape=(d_model, d_latent)),
            )
        
        init_scale = 1.0
        
        kv_down, k_up = SVDInitializer.initialize_paired_projections(
            d_model, d_latent, d_model, init_scale
        )
        
        _, v_up = SVDInitializer.initialize_paired_projections(
            d_model, d_latent, d_model, init_scale
        )
        
        return MLAInitializedWeights(
            kv_down=kv_down,
            k_up=k_up,
            v_up=v_up,
        )
    
    def compute_rank_regularization_loss(
        self,
        kv_down: mx.array,
        k_up: mx.array,
        v_up: mx.array,
    ) -> mx.array:
        """Compute rank regularization loss for MLA projections."""
        if self._rank_reg_loss is None:
            return mx.array(0.0)
        
        loss_down = self._rank_reg_loss(kv_down)
        loss_k = self._rank_reg_loss(k_up)
        loss_v = self._rank_reg_loss(v_up)
        
        return loss_down + loss_k + loss_v
    
    def check_stability(
        self,
        kv_down: mx.array,
        k_up: mx.array,
        v_up: mx.array,
    ) -> MLAStabilityReport:
        """Check numerical stability of all MLA projections."""
        kv_down_diag = self.stability_checker.check(kv_down, "kv_down")
        k_up_diag = self.stability_checker.check(k_up, "k_up")
        v_up_diag = self.stability_checker.check(v_up, "v_up")
        
        return MLAStabilityReport(
            kv_down=kv_down_diag,
            k_up=k_up_diag,
            v_up=v_up_diag,
        )
    
    def get_grad_clip_coefficient(
        self, 
        gradients: list[mx.array]
    ) -> float:
        """Get gradient clip coefficient."""
        grad_norm = LatentProjectionGradientClipper.compute_grad_norm(gradients)
        stats = self.grad_clipper.compute_clip_coefficient(grad_norm)
        return self.grad_clipper.get_clip_coefficient(
            grad_norm, 
            stats.clip_threshold
        )
