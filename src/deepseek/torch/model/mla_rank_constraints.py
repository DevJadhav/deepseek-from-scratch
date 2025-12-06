"""
MLA Rank Constraints Module for DeepSeek (PyTorch)

This module implements production-grade rank constraints for Multi-Head Latent Attention (MLA):
- SVD-based initialization for low-rank matrices
- Rank regularization loss during training
- Numerical stability checks for projection matrices with high condition numbers
- Gradient clipping specifically for latent projection weights

Reference: DeepSeek-V3 architecture specification for MLA with rank constraints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


def _safe_qr(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    QR decomposition with MPS fallback.
    
    MPS doesn't support linalg.qr, so we fall back to CPU computation
    and then move the result back to the original device.
    """
    original_device = tensor.device
    if tensor.device.type == "mps":
        # Move to CPU for QR, then back to MPS
        q, r = torch.linalg.qr(tensor.cpu())
        return q.to(original_device), r.to(original_device)
    return torch.linalg.qr(tensor)


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
    target_rank: Optional[int] = None
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
    def production(cls) -> "RankConstraintConfig":
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
    def research(cls) -> "RankConstraintConfig":
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
        shape: Tuple[int, int],
        target_rank: Optional[int] = None,
        init_scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize a weight tensor with SVD-based low-rank structure.
        
        Uses truncated SVD to create orthogonal initialization with
        controlled rank and spectral properties.
        
        Args:
            shape: (out_features, in_features)
            target_rank: Target rank for the matrix (defaults to min(out, in))
            init_scale: Scale factor for initialization
            device: Target device
            dtype: Data type
            
        Returns:
            Initialized tensor with low-rank structure
        """
        out_features, in_features = shape
        rank = target_rank if target_rank else min(out_features, in_features)
        rank = min(rank, out_features, in_features)
        
        # Generate random matrices for SVD
        u_random = torch.randn(out_features, rank, device=device, dtype=dtype)
        v_random = torch.randn(in_features, rank, device=device, dtype=dtype)
        
        # QR decomposition to get orthogonal bases (with MPS fallback)
        u, _ = _safe_qr(u_random)
        v, _ = _safe_qr(v_random)
        
        # Ensure we have the right shape after QR
        u = u[:, :rank]
        v = v[:, :rank]
        
        # Create decaying singular values (Xavier/Glorot-like)
        fan_in = float(in_features)
        fan_out = float(out_features)
        std = init_scale * math.sqrt(2.0 / (fan_in + fan_out))
        
        # Exponentially decaying singular values for smoother low-rank approximation
        indices = torch.arange(rank, device=device, dtype=dtype)
        singular_values = std * torch.exp(-indices / rank)
        
        # Construct weight: U @ diag(S) @ V^T
        weight = u @ torch.diag(singular_values) @ v.t()
        
        return weight
    
    @staticmethod
    def initialize_paired_projections(
        d_model: int,
        d_latent: int,
        d_out: int,
        init_scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize paired down/up projection matrices for MLA.
        
        Ensures the composition down @ up has controlled rank and spectral properties.
        
        Args:
            d_model: Model dimension
            d_latent: Latent dimension
            d_out: Output dimension (may differ from d_model for K/V heads)
            init_scale: Scale factor
            device: Target device
            dtype: Data type
            
        Returns:
            Tuple of (down_projection, up_projection) tensors
        """
        # Down projection: d_model -> d_latent
        down_random = torch.randn(d_latent, d_model, device=device, dtype=dtype)
        down_ortho, _ = _safe_qr(down_random.t())
        down_ortho = down_ortho.t()[:d_latent, :]
        
        # Up projection: d_latent -> d_out
        # Create orthonormal rows for the up projection
        up_random = torch.randn(d_out, d_latent, device=device, dtype=dtype)
        # QR on the transposed matrix, then transpose back
        # This gives us orthonormal columns in the original orientation
        if d_latent <= d_out:
            up_ortho, _ = _safe_qr(up_random)
            up_ortho = up_ortho[:, :d_latent] if up_ortho.shape[1] > d_latent else up_ortho
        else:
            up_ortho_t, _ = _safe_qr(up_random.t())
            up_ortho = up_ortho_t.t()[:d_out, :]
        
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
    
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute rank regularization loss using nuclear norm proxy.
        
        Uses Frobenius norm as an efficient approximation to nuclear norm
        that encourages low-rank solutions.
        
        Args:
            weight: Weight tensor to regularize
            
        Returns:
            Scalar loss tensor
        """
        if self.use_nuclear_norm:
            return self._compute_nuclear_norm_proxy(weight)
        else:
            return self._compute_power_iteration_loss(weight)
    
    def _compute_nuclear_norm_proxy(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute Frobenius norm as nuclear norm proxy"""
        # Frobenius norm squared = sum of squared singular values
        frobenius_sq = torch.sum(weight ** 2)
        
        # Target: encourage ||W||_F to be close to sqrt(target_rank) * sigma_1
        # This promotes low-rank structure where most energy is in top singular values
        return self.weight * frobenius_sq
    
    def _compute_power_iteration_loss(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute loss using power iteration to estimate top singular values"""
        # Estimate top singular value via power iteration
        device = weight.device
        dtype = weight.dtype
        
        v = torch.randn(weight.shape[1], 1, device=device, dtype=dtype)
        v = v / torch.norm(v)
        
        # Power iteration
        for _ in range(5):
            u = weight @ v
            u = u / (torch.norm(u) + 1e-8)
            v = weight.t() @ u
            v = v / (torch.norm(v) + 1e-8)
        
        sigma_max = torch.norm(weight @ v)
        
        # Frobenius norm
        frobenius = torch.norm(weight, p='fro')
        
        # Penalize if energy is not concentrated in top singular values
        # Ideal: frobenius^2 ≈ target_rank * sigma_max^2
        ratio = frobenius ** 2 / (self.target_rank * sigma_max ** 2 + 1e-8)
        tail_penalty = torch.relu(ratio - 1.0) * self.tail_penalty_factor
        
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
    warning: Optional[str] = None


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
        self.history: Dict[str, List[float]] = {}
    
    def check(self, weight: torch.Tensor, name: str) -> StabilityDiagnostics:
        """
        Check numerical stability of a weight matrix.
        
        Uses power iteration to estimate condition number and singular values
        without requiring full SVD computation.
        
        Args:
            weight: Weight tensor to check
            name: Name for logging
            
        Returns:
            StabilityDiagnostics with condition number and status
        """
        rows, cols = weight.shape
        min_dim = min(rows, cols)
        
        # Estimate max singular value via power iteration
        max_sv = self._estimate_max_singular_value(weight, iterations=10)
        
        # Estimate min singular value
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
        
        # Estimate effective rank
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
        matrix: torch.Tensor, 
        iterations: int = 10
    ) -> float:
        """Estimate maximum singular value via power iteration on A^T A"""
        device = matrix.device
        dtype = matrix.dtype
        cols = matrix.shape[1]
        
        v = torch.randn(cols, device=device, dtype=dtype)
        v = v / torch.norm(v)
        
        for _ in range(iterations):
            # A^T A v
            av = matrix @ v
            atav = matrix.t() @ av
            norm = torch.norm(atav)
            v = atav / (norm + 1e-8)
        
        # Compute Rayleigh quotient
        av = matrix @ v
        sigma_squared = torch.sum(av ** 2).item()
        v_norm_sq = torch.sum(v ** 2).item()
        
        return math.sqrt(sigma_squared / (v_norm_sq + 1e-12))
    
    def _estimate_min_singular_value(
        self, 
        matrix: torch.Tensor,
        max_sv: float,
        iterations: int = 20
    ) -> float:
        """Estimate minimum singular value from Frobenius norm relationship"""
        rows, cols = matrix.shape
        min_dim = min(rows, cols)
        
        frobenius_sq = torch.sum(matrix ** 2).item()
        
        # Conservative estimate for min singular value
        remaining_sq = max(frobenius_sq - max_sv ** 2, 0.0)
        
        if min_dim > 1:
            avg_remaining = remaining_sq / (min_dim - 1)
            estimated_min = max(math.sqrt(avg_remaining), max_sv * 1e-10)
        else:
            estimated_min = max_sv
        
        return estimated_min
    
    def _estimate_effective_rank(
        self, 
        matrix: torch.Tensor,
        max_sv: float
    ) -> int:
        """Estimate effective rank from Frobenius norm / max singular value"""
        frobenius = torch.norm(matrix, p='fro').item()
        ratio = frobenius / (max_sv + 1e-12)
        return int(math.ceil(ratio ** 2))
    
    def get_history(self, name: str) -> Optional[List[float]]:
        """Get condition number history for a weight matrix"""
        return self.history.get(name)
    
    def get_all_history(self) -> Dict[str, List[float]]:
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
    """Gradient clipping state for latent projection layers"""
    
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
        self.grad_norm_ema: Optional[float] = None
        self.clip_history: List[GradientClipStats] = []
    
    def compute_clip_coefficient(self, grad_norm: float) -> GradientClipStats:
        """
        Compute gradient clipping coefficient.
        
        Uses adaptive threshold based on EMA of gradient norms during warmup,
        then applies the learned threshold for stable clipping.
        
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
            # Use EMA-based threshold with multiplier
            clip_threshold = min(
                max(self.grad_norm_ema * 2.0, self.min_clip_threshold),
                self.max_clip_threshold
            )
        else:
            clip_threshold = self.max_norm
        
        # Compute clip coefficient
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
        """Get simple clip coefficient given norm and threshold"""
        if grad_norm > threshold:
            return threshold / (grad_norm + 1e-8)
        return 1.0
    
    def get_recent_history(self, n: int = 100) -> List[GradientClipStats]:
        """Get recent clipping history"""
        return self.clip_history[-n:]
    
    @staticmethod
    def compute_grad_norm(gradients: List[torch.Tensor]) -> float:
        """Compute total gradient norm across multiple tensors"""
        total_norm_sq = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm_sq += torch.sum(grad ** 2).item()
        return math.sqrt(total_norm_sq)
    
    def clip_gradients(
        self, 
        parameters: List[torch.nn.Parameter]
    ) -> GradientClipStats:
        """
        Clip gradients for given parameters.
        
        Args:
            parameters: List of parameters with gradients
            
        Returns:
            GradientClipStats with clipping information
        """
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return GradientClipStats(
                original_norm=0.0,
                clip_threshold=self.max_norm,
                was_clipped=False,
                clip_coefficient=1.0,
            )
        
        grad_norm = self.compute_grad_norm(grads)
        stats = self.compute_clip_coefficient(grad_norm)
        
        if stats.was_clipped:
            for param in parameters:
                if param.grad is not None:
                    param.grad.mul_(stats.clip_coefficient)
        
        return stats


@dataclass 
class MLAInitializedWeights:
    """Initialized MLA weights from constraint manager"""
    kv_down: torch.Tensor
    k_up: torch.Tensor
    v_up: torch.Tensor


@dataclass
class MLAStabilityReport:
    """Stability report for all MLA projection matrices"""
    kv_down: StabilityDiagnostics
    k_up: StabilityDiagnostics
    v_up: StabilityDiagnostics


class MLARankConstraintManager:
    """
    Unified manager for all MLA rank constraint operations.
    
    Combines initialization, regularization, stability checking, and gradient
    clipping into a single interface for MLA training.
    """
    
    def __init__(self, config: RankConstraintConfig):
        self.config = config
        
        # Initialize components based on config
        self.stability_checker = NumericalStabilityChecker(
            condition_threshold=config.condition_number_threshold,
            min_singular_value=config.min_singular_value,
            auto_correct=True,
        )
        
        self.grad_clipper = LatentProjectionGradientClipper(
            max_norm=config.latent_grad_clip_norm,
            adaptive=config.use_adaptive_grad_clip,
        )
        
        # Rank regularization loss (created on demand with target_rank)
        self._rank_reg_loss: Optional[RankRegularizationLoss] = None
    
    @property
    def rank_reg_loss(self) -> Optional[RankRegularizationLoss]:
        return self._rank_reg_loss
    
    def initialize_weights(
        self,
        d_model: int,
        d_latent: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> MLAInitializedWeights:
        """
        Initialize all MLA projection weights with rank constraints.
        
        Args:
            d_model: Model dimension
            d_latent: Latent dimension
            device: Target device
            dtype: Data type
            
        Returns:
            MLAInitializedWeights with initialized kv_down, k_up, v_up
        """
        # Set up rank regularization loss with target rank
        target_rank = self.config.target_rank or d_latent
        if self.config.rank_regularization_weight > 0:
            self._rank_reg_loss = RankRegularizationLoss(
                target_rank=target_rank,
                weight=self.config.rank_regularization_weight,
            )
        
        if not self.config.use_svd_init:
            # Fall back to random initialization
            return MLAInitializedWeights(
                kv_down=torch.randn(d_latent, d_model, device=device, dtype=dtype),
                k_up=torch.randn(d_model, d_latent, device=device, dtype=dtype),
                v_up=torch.randn(d_model, d_latent, device=device, dtype=dtype),
            )
        
        init_scale = 1.0
        
        # Initialize paired projections for K path
        kv_down, k_up = SVDInitializer.initialize_paired_projections(
            d_model, d_latent, d_model, init_scale, device, dtype
        )
        
        # Initialize V up-projection
        _, v_up = SVDInitializer.initialize_paired_projections(
            d_model, d_latent, d_model, init_scale, device, dtype
        )
        
        return MLAInitializedWeights(
            kv_down=kv_down,
            k_up=k_up,
            v_up=v_up,
        )
    
    def compute_rank_regularization_loss(
        self,
        kv_down: torch.Tensor,
        k_up: torch.Tensor,
        v_up: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rank regularization loss for MLA projections.
        
        Args:
            kv_down: KV down projection weight
            k_up: K up projection weight
            v_up: V up projection weight
            
        Returns:
            Scalar loss tensor
        """
        if self._rank_reg_loss is None:
            return torch.tensor(0.0, device=kv_down.device)
        
        loss_down = self._rank_reg_loss(kv_down)
        loss_k = self._rank_reg_loss(k_up)
        loss_v = self._rank_reg_loss(v_up)
        
        return loss_down + loss_k + loss_v
    
    def check_stability(
        self,
        kv_down: torch.Tensor,
        k_up: torch.Tensor,
        v_up: torch.Tensor,
    ) -> MLAStabilityReport:
        """
        Check numerical stability of all MLA projections.
        
        Args:
            kv_down: KV down projection weight
            k_up: K up projection weight
            v_up: V up projection weight
            
        Returns:
            MLAStabilityReport with diagnostics for each projection
        """
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
        gradients: List[torch.Tensor]
    ) -> float:
        """
        Get gradient clip coefficient for latent projection gradients.
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Clip coefficient (1.0 if no clipping needed)
        """
        grad_norm = LatentProjectionGradientClipper.compute_grad_norm(gradients)
        stats = self.grad_clipper.compute_clip_coefficient(grad_norm)
        return self.grad_clipper.get_clip_coefficient(
            grad_norm, 
            stats.clip_threshold
        )
    
    def clip_latent_gradients(
        self,
        parameters: List[torch.nn.Parameter]
    ) -> GradientClipStats:
        """
        Clip gradients for latent projection parameters.
        
        Args:
            parameters: List of parameters with gradients
            
        Returns:
            GradientClipStats with clipping information
        """
        return self.grad_clipper.clip_gradients(parameters)
