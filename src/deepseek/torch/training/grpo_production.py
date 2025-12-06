"""
Production GRPO (Group Relative Policy Optimization) - PyTorch Implementation

This module implements production-ready GRPO with:
1. PPO-style clipping for policy ratio (prevents too large updates)
2. Dynamic beta adjustment based on KL divergence (adaptive regularization)
3. Reference model update strategies (soft update vs. hard copy)
4. Heterogeneous generation/training support (Apple Silicon rollout + CUDA training)

Reference: production_hardening.md Section 3.3 Phase 3: Post-Training (RLHF/GRPO)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ReferenceUpdateStrategy(Enum):
    """Strategy for updating the reference model."""

    HARD_COPY = "hard_copy"  # Periodic full copy
    SOFT_UPDATE = "soft_update"  # EMA-style polyak averaging
    NO_UPDATE = "no_update"  # Keep frozen


class BetaSchedule(Enum):
    """KL penalty beta scheduling strategy."""

    CONSTANT = "constant"  # Fixed beta
    ADAPTIVE = "adaptive"  # Adjust based on KL divergence
    LINEAR_DECAY = "linear_decay"  # Linear decay over training


@dataclass
class GRPOConfig:
    """Configuration for production GRPO training.

    Attributes:
        beta: Initial KL penalty coefficient
        clip_ratio: PPO-style clipping ratio (e.g., 0.2 means clip to [0.8, 1.2])
        clip_value_loss: Whether to clip value function loss
        value_clip_range: Range for value function clipping

        beta_schedule: How to adjust beta over training
        target_kl: Target KL divergence for adaptive beta
        beta_min: Minimum beta value
        beta_max: Maximum beta value
        beta_adjustment_factor: How much to adjust beta per step

        ref_update_strategy: How to update reference model
        ref_update_interval: Steps between hard copy updates
        soft_update_tau: Polyak averaging coefficient (0.001 typical)

        group_size: Number of samples per prompt (G in GRPO)
        advantage_normalization: Whether to normalize advantages
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Gradient clipping threshold
    """

    # Core GRPO parameters
    beta: float = 0.01
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    value_clip_range: float = 0.2

    # Dynamic beta adjustment
    beta_schedule: BetaSchedule = BetaSchedule.ADAPTIVE
    target_kl: float = 0.01  # Target KL for adaptive adjustment
    beta_min: float = 0.001
    beta_max: float = 0.1
    beta_adjustment_factor: float = 1.5

    # Reference model update
    ref_update_strategy: ReferenceUpdateStrategy = ReferenceUpdateStrategy.SOFT_UPDATE
    ref_update_interval: int = 100  # For hard copy
    soft_update_tau: float = 0.001  # For soft update

    # Training parameters
    group_size: int = 4
    advantage_normalization: bool = True
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 10


@dataclass
class GRPOState:
    """Tracks GRPO training state."""

    step: int = 0
    current_beta: float = 0.01
    mean_kl: float = 0.0
    mean_entropy: float = 0.0
    mean_ratio: float = 1.0
    clip_fraction: float = 0.0
    mean_advantage: float = 0.0
    total_updates: int = 0
    last_ref_update_step: int = 0

    # History for logging
    kl_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)


@dataclass
class RolloutBatch:
    """Batch of rollout data for GRPO training.

    Contains generated outputs from rollout actors along with
    their log probabilities and rewards.
    """

    # Generated sequences (G, Seq)
    input_ids: Any  # torch.Tensor
    attention_mask: Any | None = None  # torch.Tensor

    # Log probabilities from behavior policy (G, Seq)
    behavior_log_probs: Any | None = None

    # Rewards (G,)
    rewards: Any | None = None

    # Optional: precomputed advantages (G,)
    advantages: Any | None = None

    # Metadata
    prompt_ids: list[str] | None = None
    generation_time: float = 0.0


# =============================================================================
# Production GRPO Trainer
# =============================================================================


class ProductionGRPOTrainer:
    """Production-ready GRPO trainer with PPO-style clipping and adaptive beta.

    Key features:
    - PPO-style probability ratio clipping for stable updates
    - Dynamic beta adjustment based on KL divergence
    - Multiple reference model update strategies
    - Entropy bonus for exploration
    - Comprehensive logging and monitoring

    Example:
        ```python
        config = GRPOConfig(
            beta=0.01,
            clip_ratio=0.2,
            beta_schedule=BetaSchedule.ADAPTIVE,
            ref_update_strategy=ReferenceUpdateStrategy.SOFT_UPDATE,
        )

        trainer = ProductionGRPOTrainer(
            policy_model=model,
            ref_model=ref_model,
            config=config,
        )

        # Training loop
        for batch in dataloader:
            rollouts = generate_rollouts(batch)
            metrics = trainer.train_step(rollouts)
        ```
    """

    def __init__(
        self,
        policy_model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        optimizer: Any | None = None,
        config: GRPOConfig | None = None,
        device: str | None = None,
    ):
        """Initialize GRPO trainer.

        Args:
            policy_model: The policy model to train
            ref_model: Reference model for KL penalty (frozen copy)
            optimizer: Optimizer for policy model
            config: GRPO configuration
            device: Device to use (cuda, mps, cpu)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ProductionGRPOTrainer")

        self.config = config or GRPOConfig()
        self.state = GRPOState(current_beta=self.config.beta)

        # Device selection with fallback
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Models
        self.policy_model = policy_model
        self.ref_model = ref_model

        # Freeze reference model
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        # Optimizer
        self.optimizer = optimizer

        LOGGER.info(
            "Initialized ProductionGRPOTrainer: device=%s, clip_ratio=%.2f, beta=%.4f",
            self.device,
            self.config.clip_ratio,
            self.config.beta,
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Compute normalized advantages from rewards.

        GRPO uses group-relative advantages:
            A_i = (r_i - mean(r)) / std(r)

        Args:
            rewards: Reward tensor (G,)
            normalize: Whether to normalize (default: use config)

        Returns:
            Advantages tensor (G,)
        """
        if normalize is None:
            normalize = self.config.advantage_normalization

        if normalize:
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            advantages = (rewards - mean_r) / std_r
        else:
            advantages = rewards - rewards.mean()

        return advantages

    def compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute log probabilities of tokens under the policy.

        Args:
            logits: Model logits (B, Seq, Vocab)
            input_ids: Token IDs (B, Seq)
            attention_mask: Mask for valid tokens (B, Seq)

        Returns:
            Log probabilities (B, Seq)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        # Mask invalid positions
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask

        return token_log_probs

    def compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference.

        KL(policy || ref) = sum(policy * (log_policy - log_ref))

        Args:
            policy_logits: Policy model logits (B, Seq, Vocab)
            ref_logits: Reference model logits (B, Seq, Vocab)
            attention_mask: Mask for valid tokens (B, Seq)

        Returns:
            KL divergence per sequence (B,)
        """
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_probs = F.softmax(policy_logits, dim=-1)

        # KL per token
        kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)  # (B, Seq)

        # Mask and average
        if attention_mask is not None:
            kl = kl * attention_mask
            seq_kl = kl.sum(dim=1) / (attention_mask.sum(dim=1) + 1e-8)
        else:
            seq_kl = kl.mean(dim=1)

        return seq_kl

    def compute_entropy(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute entropy of the policy distribution.

        H(p) = -sum(p * log(p))

        Args:
            logits: Policy logits (B, Seq, Vocab)
            attention_mask: Mask for valid tokens (B, Seq)

        Returns:
            Entropy per sequence (B,)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # Entropy per token
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, Seq)

        # Mask and average
        if attention_mask is not None:
            entropy = entropy * attention_mask
            seq_entropy = entropy.sum(dim=1) / (attention_mask.sum(dim=1) + 1e-8)
        else:
            seq_entropy = entropy.mean(dim=1)

        return seq_entropy

    def compute_ppo_loss(
        self,
        policy_log_probs: torch.Tensor,
        behavior_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO-style clipped policy loss.

        L_CLIP = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

        Args:
            policy_log_probs: Log probs from current policy (B, Seq)
            behavior_log_probs: Log probs from behavior policy (B, Seq)
            advantages: Advantage estimates (B,)
            attention_mask: Mask for valid tokens (B, Seq)

        Returns:
            Tuple of (loss, mean_ratio, clip_fraction)
        """
        # Compute probability ratio
        # ratio = exp(log_pi - log_pi_old) = pi / pi_old
        log_ratio = policy_log_probs - behavior_log_probs

        # Mask invalid positions
        if attention_mask is not None:
            log_ratio = log_ratio * attention_mask
            seq_log_ratio = log_ratio.sum(dim=1) / (attention_mask.sum(dim=1) + 1e-8)
        else:
            seq_log_ratio = log_ratio.mean(dim=1)

        ratio = torch.exp(seq_log_ratio)  # (B,)

        # Clipping
        eps = self.config.clip_ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

        # Advantages are (B,) - broadcast for element-wise
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages

        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Compute clip fraction for monitoring
        clip_fraction = ((ratio - 1.0).abs() > eps).float().mean()

        return policy_loss, ratio.mean(), clip_fraction

    def update_beta(self, mean_kl: float) -> None:
        """Update beta based on KL divergence (adaptive schedule).

        If KL > target: increase beta (more conservative)
        If KL < target: decrease beta (more aggressive)

        Args:
            mean_kl: Mean KL divergence from current batch
        """
        if self.config.beta_schedule != BetaSchedule.ADAPTIVE:
            return

        target = self.config.target_kl
        factor = self.config.beta_adjustment_factor

        if mean_kl > target * 1.5:
            # KL too high, increase penalty
            new_beta = self.state.current_beta * factor
        elif mean_kl < target * 0.5:
            # KL too low, decrease penalty
            new_beta = self.state.current_beta / factor
        else:
            # In acceptable range
            return

        # Clamp to bounds
        self.state.current_beta = max(self.config.beta_min, min(self.config.beta_max, new_beta))

        LOGGER.debug(
            "Updated beta: %.4f (KL=%.4f, target=%.4f)",
            self.state.current_beta,
            mean_kl,
            target,
        )

    def update_reference_model(self) -> bool:
        """Update reference model based on configured strategy.

        Returns:
            True if update was performed
        """
        if self.ref_model is None or self.policy_model is None:
            return False

        strategy = self.config.ref_update_strategy

        if strategy == ReferenceUpdateStrategy.NO_UPDATE:
            return False

        if strategy == ReferenceUpdateStrategy.HARD_COPY:
            # Periodic full copy
            steps_since_update = self.state.step - self.state.last_ref_update_step
            if steps_since_update >= self.config.ref_update_interval:
                self.ref_model.load_state_dict(self.policy_model.state_dict())
                self.state.last_ref_update_step = self.state.step
                LOGGER.info("Hard copied policy to reference at step %d", self.state.step)
                return True

        elif strategy == ReferenceUpdateStrategy.SOFT_UPDATE:
            # Polyak averaging: ref = tau * policy + (1 - tau) * ref
            tau = self.config.soft_update_tau
            with torch.no_grad():
                for ref_param, policy_param in zip(
                    self.ref_model.parameters(),
                    self.policy_model.parameters(),
                    strict=True,
                ):
                    ref_param.data.mul_(1 - tau).add_(policy_param.data, alpha=tau)
            return True

        return False

    def compute_grpo_loss(
        self,
        rollouts: RolloutBatch,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute full GRPO loss with PPO clipping.

        Loss = L_CLIP + beta * KL - entropy_coef * H

        Args:
            rollouts: Batch of rollout data
            policy_logits: Current policy logits
            ref_logits: Reference model logits

        Returns:
            Dictionary with loss components
        """
        input_ids = rollouts.input_ids
        attention_mask = rollouts.attention_mask
        rewards = rollouts.rewards
        behavior_log_probs = rollouts.behavior_log_probs

        # Compute advantages
        if rollouts.advantages is not None:
            advantages = rollouts.advantages
        else:
            advantages = self.compute_advantages(rewards)

        # Compute policy log probs
        policy_log_probs = self.compute_log_probs(policy_logits, input_ids, attention_mask)

        # Compute KL divergence
        kl = self.compute_kl_divergence(policy_logits, ref_logits, attention_mask)
        mean_kl = kl.mean()

        # Compute entropy bonus
        entropy = self.compute_entropy(policy_logits, attention_mask)
        mean_entropy = entropy.mean()

        # Compute PPO-style clipped loss
        if behavior_log_probs is not None:
            policy_loss, mean_ratio, clip_fraction = self.compute_ppo_loss(
                policy_log_probs, behavior_log_probs, advantages, attention_mask
            )
        else:
            # Fallback to simple REINFORCE-style loss
            if attention_mask is not None:
                seq_log_probs = (policy_log_probs * attention_mask).sum(dim=1)
            else:
                seq_log_probs = policy_log_probs.sum(dim=1)

            policy_loss = -(advantages * seq_log_probs).mean()
            mean_ratio = torch.tensor(1.0)
            clip_fraction = torch.tensor(0.0)

        # KL penalty
        kl_loss = self.state.current_beta * mean_kl

        # Entropy bonus (negative because we want to maximize entropy)
        entropy_loss = -self.config.entropy_coef * mean_entropy

        # Total loss
        total_loss = policy_loss + kl_loss + entropy_loss

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "mean_kl": mean_kl,
            "mean_entropy": mean_entropy,
            "mean_ratio": mean_ratio,
            "clip_fraction": clip_fraction,
            "mean_advantage": advantages.mean(),
        }

    def train_step(
        self,
        rollouts: RolloutBatch,
    ) -> dict[str, float]:
        """Perform a single GRPO training step.

        Args:
            rollouts: Batch of rollout data with rewards

        Returns:
            Dictionary of training metrics
        """
        if self.policy_model is None or self.optimizer is None:
            raise RuntimeError("Policy model and optimizer required for training")

        self.policy_model.train()

        # Move data to device
        input_ids = rollouts.input_ids.to(self.device)
        attention_mask = (
            rollouts.attention_mask.to(self.device) if rollouts.attention_mask is not None else None
        )
        rewards = rollouts.rewards.to(self.device)

        # Update rollouts with device tensors
        rollouts = RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            rewards=rewards,
            behavior_log_probs=(
                rollouts.behavior_log_probs.to(self.device)
                if rollouts.behavior_log_probs is not None
                else None
            ),
            advantages=(
                rollouts.advantages.to(self.device) if rollouts.advantages is not None else None
            ),
        )

        # Forward pass
        policy_outputs = self.policy_model(input_ids, attention_mask=attention_mask)
        policy_logits = (
            policy_outputs.logits if hasattr(policy_outputs, "logits") else policy_outputs
        )

        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs

        # Compute loss
        loss_dict = self.compute_grpo_loss(rollouts, policy_logits, ref_logits)

        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()

        # Update state
        self.state.step += 1
        self.state.total_updates += 1
        mean_kl = loss_dict["mean_kl"].item()
        self.state.mean_kl = mean_kl
        self.state.mean_entropy = loss_dict["mean_entropy"].item()
        self.state.mean_ratio = loss_dict["mean_ratio"].item()
        self.state.clip_fraction = loss_dict["clip_fraction"].item()
        self.state.mean_advantage = loss_dict["mean_advantage"].item()

        # Update beta (adaptive schedule)
        self.update_beta(mean_kl)

        # Update reference model
        self.update_reference_model()

        # Track history
        self.state.kl_history.append(mean_kl)
        self.state.loss_history.append(loss_dict["total_loss"].item())
        self.state.reward_history.append(rewards.mean().item())

        # Return metrics
        metrics = {
            "loss": loss_dict["total_loss"].item(),
            "policy_loss": loss_dict["policy_loss"].item(),
            "kl_loss": loss_dict["kl_loss"].item(),
            "entropy_loss": loss_dict["entropy_loss"].item(),
            "mean_kl": mean_kl,
            "mean_entropy": self.state.mean_entropy,
            "mean_ratio": self.state.mean_ratio,
            "clip_fraction": self.state.clip_fraction,
            "beta": self.state.current_beta,
            "step": self.state.step,
        }

        # Log periodically
        if self.state.step % self.config.log_interval == 0:
            LOGGER.info(
                "Step %d: loss=%.4f, KL=%.4f, clip_frac=%.2f, beta=%.4f",
                self.state.step,
                metrics["loss"],
                metrics["mean_kl"],
                metrics["clip_fraction"],
                metrics["beta"],
            )

        return metrics

    def get_state_dict(self) -> dict[str, Any]:
        """Get trainer state for checkpointing."""
        return {
            "config": {
                "beta": self.config.beta,
                "clip_ratio": self.config.clip_ratio,
                "beta_schedule": self.config.beta_schedule.value,
                "ref_update_strategy": self.config.ref_update_strategy.value,
            },
            "state": {
                "step": self.state.step,
                "current_beta": self.state.current_beta,
                "total_updates": self.state.total_updates,
                "last_ref_update_step": self.state.last_ref_update_step,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load trainer state from checkpoint."""
        if "state" in state_dict:
            self.state.step = state_dict["state"]["step"]
            self.state.current_beta = state_dict["state"]["current_beta"]
            self.state.total_updates = state_dict["state"]["total_updates"]
            self.state.last_ref_update_step = state_dict["state"]["last_ref_update_step"]


# =============================================================================
# Legacy GRPO Trainer (backward compatibility)
# =============================================================================


class GRPOTrainer:
    """Original GRPO trainer (backward compatible).

    This is the original simple implementation. For production use,
    prefer ProductionGRPOTrainer.
    """

    def __init__(self, beta: float = 0.01):
        self.beta = beta

    def compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        rewards: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute basic GRPO loss."""
        G, Seq, _ = logits.shape

        # 1. Advantages
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # 2. Policy Log Probs
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = token_log_probs.sum(dim=1)

        # 3. KL Divergence
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        mean_kl = kl.mean(dim=1)

        # 4. Loss
        loss = -(advantages * seq_log_probs) + self.beta * mean_kl
        return loss.mean()


class GroupSampler:
    """Group sampler for GRPO (placeholder)."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    def sample(self, prompt: str) -> list[str]:
        return [f"Sampled output {i} for prompt..." for i in range(self.group_size)]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Config
    "GRPOConfig",
    "GRPOState",
    "RolloutBatch",
    "ReferenceUpdateStrategy",
    "BetaSchedule",
    # Trainers
    "ProductionGRPOTrainer",
    "GRPOTrainer",
    "GroupSampler",
]
