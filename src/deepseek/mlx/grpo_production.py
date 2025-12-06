"""
Production GRPO (Group Relative Policy Optimization) - MLX Implementation

This module implements production-ready GRPO for Apple Silicon with:
1. PPO-style clipping for policy ratio
2. Dynamic beta adjustment based on KL divergence
3. Reference model update strategies
4. Optimized for Apple Silicon (M1/M2/M3/M4)

Reference: production_hardening.md Section 3.3 Phase 3: Post-Training (RLHF/GRPO)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# MLX imports with fallback
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore
    nn = None  # type: ignore

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration (shared with PyTorch)
# =============================================================================


class ReferenceUpdateStrategy(Enum):
    """Strategy for updating the reference model."""

    HARD_COPY = "hard_copy"
    SOFT_UPDATE = "soft_update"
    NO_UPDATE = "no_update"


class BetaSchedule(Enum):
    """KL penalty beta scheduling strategy."""

    CONSTANT = "constant"
    ADAPTIVE = "adaptive"
    LINEAR_DECAY = "linear_decay"


@dataclass
class GRPOConfig:
    """Configuration for production GRPO training."""

    # Core GRPO parameters
    beta: float = 0.01
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    value_clip_range: float = 0.2

    # Dynamic beta adjustment
    beta_schedule: BetaSchedule = BetaSchedule.ADAPTIVE
    target_kl: float = 0.01
    beta_min: float = 0.001
    beta_max: float = 0.1
    beta_adjustment_factor: float = 1.5

    # Reference model update
    ref_update_strategy: ReferenceUpdateStrategy = ReferenceUpdateStrategy.SOFT_UPDATE
    ref_update_interval: int = 100
    soft_update_tau: float = 0.001

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
    """Batch of rollout data for GRPO training."""

    input_ids: Any  # mx.array
    attention_mask: Any | None = None
    behavior_log_probs: Any | None = None
    rewards: Any | None = None
    advantages: Any | None = None
    prompt_ids: list[str] | None = None
    generation_time: float = 0.0


# =============================================================================
# Production GRPO Trainer - MLX
# =============================================================================


class ProductionGRPOTrainerMLX:
    """Production-ready GRPO trainer for MLX (Apple Silicon).

    Key features:
    - PPO-style probability ratio clipping
    - Dynamic beta adjustment based on KL divergence
    - Multiple reference model update strategies
    - Entropy bonus for exploration
    - Optimized for Apple Silicon

    Example:
        ```python
        config = GRPOConfig(
            beta=0.01,
            clip_ratio=0.2,
            beta_schedule=BetaSchedule.ADAPTIVE,
        )

        trainer = ProductionGRPOTrainerMLX(
            policy_model=model,
            ref_model=ref_model,
            config=config,
        )

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
    ):
        """Initialize MLX GRPO trainer."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX required for ProductionGRPOTrainerMLX")

        self.config = config or GRPOConfig()
        self.state = GRPOState(current_beta=self.config.beta)

        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer

        LOGGER.info(
            "Initialized ProductionGRPOTrainerMLX: clip_ratio=%.2f, beta=%.4f",
            self.config.clip_ratio,
            self.config.beta,
        )

    def compute_advantages(
        self,
        rewards: mx.array,
        normalize: bool | None = None,
    ) -> mx.array:
        """Compute normalized advantages from rewards."""
        if normalize is None:
            normalize = self.config.advantage_normalization

        if normalize:
            mean_r = mx.mean(rewards)
            std_r = mx.std(rewards) + 1e-8
            advantages = (rewards - mean_r) / std_r
        else:
            advantages = rewards - mx.mean(rewards)

        return advantages

    def compute_log_probs(
        self,
        logits: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Compute log probabilities of tokens under the policy."""
        # Log softmax
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

        # Gather log probs for actual tokens using advanced indexing
        G, Seq, _ = logits.shape
        batch_idx = mx.arange(G)[:, None]
        seq_idx = mx.arange(Seq)[None, :]
        token_log_probs = log_probs[batch_idx, seq_idx, input_ids]

        # Mask invalid positions
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask

        return token_log_probs

    def compute_kl_divergence(
        self,
        policy_logits: mx.array,
        ref_logits: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Compute KL divergence between policy and reference."""
        policy_probs = mx.softmax(policy_logits, axis=-1)
        policy_log_probs = mx.log(policy_probs + 1e-10)
        ref_log_probs = mx.log(mx.softmax(ref_logits, axis=-1) + 1e-10)

        # KL per token
        kl = mx.sum(policy_probs * (policy_log_probs - ref_log_probs), axis=-1)

        # Mask and average
        if attention_mask is not None:
            kl = kl * attention_mask
            seq_kl = mx.sum(kl, axis=1) / (mx.sum(attention_mask, axis=1) + 1e-8)
        else:
            seq_kl = mx.mean(kl, axis=1)

        return seq_kl

    def compute_entropy(
        self,
        logits: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Compute entropy of the policy distribution."""
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(probs + 1e-10)

        # Entropy per token
        entropy = -mx.sum(probs * log_probs, axis=-1)

        # Mask and average
        if attention_mask is not None:
            entropy = entropy * attention_mask
            seq_entropy = mx.sum(entropy, axis=1) / (mx.sum(attention_mask, axis=1) + 1e-8)
        else:
            seq_entropy = mx.mean(entropy, axis=1)

        return seq_entropy

    def compute_ppo_loss(
        self,
        policy_log_probs: mx.array,
        behavior_log_probs: mx.array,
        advantages: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute PPO-style clipped policy loss."""
        # Compute probability ratio
        log_ratio = policy_log_probs - behavior_log_probs

        # Mask invalid positions
        if attention_mask is not None:
            log_ratio = log_ratio * attention_mask
            seq_log_ratio = mx.sum(log_ratio, axis=1) / (mx.sum(attention_mask, axis=1) + 1e-8)
        else:
            seq_log_ratio = mx.mean(log_ratio, axis=1)

        ratio = mx.exp(seq_log_ratio)

        # Clipping
        eps = self.config.clip_ratio
        clipped_ratio = mx.clip(ratio, 1.0 - eps, 1.0 + eps)

        # Surrogate objectives
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages

        # Take minimum (pessimistic bound)
        policy_loss = -mx.mean(mx.minimum(surrogate1, surrogate2))

        # Compute clip fraction
        clip_fraction = mx.mean((mx.abs(ratio - 1.0) > eps).astype(mx.float32))

        return policy_loss, mx.mean(ratio), clip_fraction

    def update_beta(self, mean_kl: float) -> None:
        """Update beta based on KL divergence."""
        if self.config.beta_schedule != BetaSchedule.ADAPTIVE:
            return

        target = self.config.target_kl
        factor = self.config.beta_adjustment_factor

        if mean_kl > target * 1.5:
            new_beta = self.state.current_beta * factor
        elif mean_kl < target * 0.5:
            new_beta = self.state.current_beta / factor
        else:
            return

        self.state.current_beta = max(self.config.beta_min, min(self.config.beta_max, new_beta))

    def update_reference_model(self) -> bool:
        """Update reference model based on configured strategy."""
        if self.ref_model is None or self.policy_model is None:
            return False

        strategy = self.config.ref_update_strategy

        if strategy == ReferenceUpdateStrategy.NO_UPDATE:
            return False

        if strategy == ReferenceUpdateStrategy.HARD_COPY:
            steps_since_update = self.state.step - self.state.last_ref_update_step
            if steps_since_update >= self.config.ref_update_interval:
                # Copy weights
                policy_weights = self.policy_model.parameters()
                self.ref_model.update(policy_weights)
                self.state.last_ref_update_step = self.state.step
                LOGGER.info("Hard copied policy to reference at step %d", self.state.step)
                return True

        elif strategy == ReferenceUpdateStrategy.SOFT_UPDATE:
            # Polyak averaging
            tau = self.config.soft_update_tau
            policy_params = self.policy_model.parameters()
            ref_params = self.ref_model.parameters()

            # Update each parameter
            def soft_update(ref_tree: dict, policy_tree: dict) -> dict:
                updated = {}
                for key in ref_tree:
                    if isinstance(ref_tree[key], dict):
                        updated[key] = soft_update(ref_tree[key], policy_tree[key])
                    elif isinstance(ref_tree[key], mx.array):
                        updated[key] = (1 - tau) * ref_tree[key] + tau * policy_tree[key]
                    else:
                        updated[key] = ref_tree[key]
                return updated

            new_ref_params = soft_update(ref_params, policy_params)
            self.ref_model.update(new_ref_params)
            return True

        return False

    def compute_grpo_loss(
        self,
        rollouts: RolloutBatch,
        policy_logits: mx.array,
        ref_logits: mx.array,
    ) -> dict[str, mx.array]:
        """Compute full GRPO loss with PPO clipping."""
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
        mean_kl = mx.mean(kl)

        # Compute entropy bonus
        entropy = self.compute_entropy(policy_logits, attention_mask)
        mean_entropy = mx.mean(entropy)

        # Compute PPO-style clipped loss
        if behavior_log_probs is not None:
            policy_loss, mean_ratio, clip_fraction = self.compute_ppo_loss(
                policy_log_probs, behavior_log_probs, advantages, attention_mask
            )
        else:
            # Fallback to simple REINFORCE-style loss
            if attention_mask is not None:
                seq_log_probs = mx.sum(policy_log_probs * attention_mask, axis=1)
            else:
                seq_log_probs = mx.sum(policy_log_probs, axis=1)

            policy_loss = -mx.mean(advantages * seq_log_probs)
            mean_ratio = mx.array(1.0)
            clip_fraction = mx.array(0.0)

        # KL penalty
        kl_loss = self.state.current_beta * mean_kl

        # Entropy bonus
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
            "mean_advantage": mx.mean(advantages),
        }

    def train_step(self, rollouts: RolloutBatch) -> dict[str, float]:
        """Perform a single GRPO training step."""
        if self.policy_model is None or self.optimizer is None:
            raise RuntimeError("Policy model and optimizer required for training")

        input_ids = rollouts.input_ids
        # attention_mask accessed via rollouts in loss_fn

        def loss_fn(model: nn.Module) -> tuple[mx.array, dict[str, mx.array]]:
            """Loss function for value_and_grad."""
            policy_logits = model(input_ids)
            ref_logits = self.ref_model(input_ids)
            loss_dict = self.compute_grpo_loss(rollouts, policy_logits, ref_logits)
            return loss_dict["total_loss"], loss_dict

        # Compute gradients
        (loss, loss_dict), grads = mx.value_and_grad(loss_fn, has_aux=True)(self.policy_model)

        # Update with optimizer
        self.optimizer.update(self.policy_model, grads)

        # Evaluate to get values
        mx.eval(self.policy_model.parameters())

        # Update state
        self.state.step += 1
        self.state.total_updates += 1
        mean_kl = float(loss_dict["mean_kl"])
        self.state.mean_kl = mean_kl
        self.state.mean_entropy = float(loss_dict["mean_entropy"])
        self.state.mean_ratio = float(loss_dict["mean_ratio"])
        self.state.clip_fraction = float(loss_dict["clip_fraction"])
        self.state.mean_advantage = float(loss_dict["mean_advantage"])

        # Update beta
        self.update_beta(mean_kl)

        # Update reference model
        self.update_reference_model()

        # Track history
        self.state.kl_history.append(mean_kl)
        self.state.loss_history.append(float(loss_dict["total_loss"]))
        self.state.reward_history.append(float(mx.mean(rollouts.rewards)))

        # Return metrics
        metrics = {
            "loss": float(loss_dict["total_loss"]),
            "policy_loss": float(loss_dict["policy_loss"]),
            "kl_loss": float(loss_dict["kl_loss"]),
            "entropy_loss": float(loss_dict["entropy_loss"]),
            "mean_kl": mean_kl,
            "mean_entropy": self.state.mean_entropy,
            "mean_ratio": self.state.mean_ratio,
            "clip_fraction": self.state.clip_fraction,
            "beta": self.state.current_beta,
            "step": self.state.step,
        }

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


# =============================================================================
# Legacy GRPO Trainer (backward compatibility)
# =============================================================================


class GRPOTrainer:
    """Original GRPO trainer for MLX (backward compatible)."""

    def __init__(self, beta: float = 0.01):
        self.beta = beta

    def compute_loss(
        self,
        logits: mx.array,
        input_ids: mx.array,
        rewards: mx.array,
        ref_logits: mx.array,
    ) -> mx.array:
        """Compute basic GRPO loss."""
        G, Seq, Vocab = logits.shape

        # 1. Compute advantages
        mean_r = mx.mean(rewards)
        std_r = mx.std(rewards) + 1e-8
        advantages = (rewards - mean_r) / std_r

        # 2. Compute policy log probs
        log_probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(log_probs + 1e-10)

        batch_idx = mx.arange(G)[:, None]
        seq_idx = mx.arange(Seq)[None, :]
        token_log_probs = log_probs[batch_idx, seq_idx, input_ids]
        seq_log_probs = mx.sum(token_log_probs, axis=1)

        # 3. Compute KL divergence
        ref_log_probs = mx.softmax(ref_logits, axis=-1)
        ref_log_probs = mx.log(ref_log_probs + 1e-10)

        probs = mx.softmax(logits, axis=-1)
        kl = mx.sum(probs * (log_probs - ref_log_probs), axis=-1)
        mean_kl = mx.mean(kl, axis=1)

        # 4. Compute loss
        loss = -(advantages * seq_log_probs) + self.beta * mean_kl
        return mx.mean(loss)


class GroupSampler:
    """Group sampler for GRPO."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    def sample(self, prompt: str) -> list:
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
    "ProductionGRPOTrainerMLX",
    "GRPOTrainer",
    "GroupSampler",
]
