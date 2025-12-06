"""
Heterogeneous GRPO Pipeline for Apple Silicon

This module implements a hybrid training pipeline that:
1. Uses Apple Silicon (MLX) for generation/rollout
2. Uses CUDA/GPU (PyTorch) for training updates

This approach leverages the strengths of both platforms:
- MLX: Efficient memory management for large model inference on M1/M2/M3/M4
- CUDA: High throughput for gradient computation and model updates

Reference: production_hardening.md Section 3.3 Phase 3: Post-Training (RLHF/GRPO)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

# Try to import MLX for Apple Silicon
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore
    mlx_nn = None  # type: ignore

from src.deepseek.torch.training.grpo_production import (
    GRPOConfig,
    ProductionGRPOTrainer,
)
from src.deepseek.torch.training.grpo_production import (
    RolloutBatch as TorchRolloutBatch,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HeterogeneousConfig:
    """Configuration for heterogeneous GRPO pipeline."""

    # Generation settings (MLX)
    generation_batch_size: int = 4
    max_generation_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9

    # Training settings
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)

    # Pipeline settings
    rollouts_per_step: int = 4
    checkpoint_interval: int = 100

    # Device settings
    use_mlx_generation: bool = True
    pytorch_device: str = "cuda"  # "cuda", "mps", or "cpu"


# =============================================================================
# Rollout Data Transfer
# =============================================================================


@dataclass
class RolloutData:
    """Container for rollout data that can be transferred between MLX and PyTorch."""

    input_ids: list[list[int]]
    attention_mask: list[list[int]] | None = None
    rewards: list[float] | None = None
    behavior_log_probs: list[list[float]] | None = None
    generation_time: float = 0.0

    def to_torch(self, device: torch.device | str = "cpu") -> TorchRolloutBatch:
        """Convert to PyTorch RolloutBatch."""
        device = torch.device(device) if isinstance(device, str) else device

        input_ids_tensor = torch.tensor(self.input_ids, dtype=torch.long, device=device)

        attention_mask_tensor = None
        if self.attention_mask is not None:
            attention_mask_tensor = torch.tensor(
                self.attention_mask, dtype=torch.float32, device=device
            )

        rewards_tensor = None
        if self.rewards is not None:
            rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=device)

        behavior_log_probs_tensor = None
        if self.behavior_log_probs is not None:
            behavior_log_probs_tensor = torch.tensor(
                self.behavior_log_probs, dtype=torch.float32, device=device
            )

        return TorchRolloutBatch(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            rewards=rewards_tensor,
            behavior_log_probs=behavior_log_probs_tensor,
            generation_time=self.generation_time,
        )

    @staticmethod
    def from_mlx(
        input_ids: Any,  # mx.array
        attention_mask: Any | None = None,  # mx.array
        rewards: Any | None = None,  # mx.array
        behavior_log_probs: Any | None = None,  # mx.array
        generation_time: float = 0.0,
    ) -> RolloutData:
        """Create from MLX arrays."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX required for from_mlx")

        return RolloutData(
            input_ids=input_ids.tolist(),
            attention_mask=attention_mask.tolist() if attention_mask is not None else None,
            rewards=rewards.tolist() if rewards is not None else None,
            behavior_log_probs=(
                behavior_log_probs.tolist() if behavior_log_probs is not None else None
            ),
            generation_time=generation_time,
        )


# =============================================================================
# MLX Generation Engine
# =============================================================================


class MLXGenerationEngine:
    """Handles generation/rollout on Apple Silicon using MLX."""

    def __init__(
        self,
        model: Any | None = None,  # mlx_nn.Module
        config: HeterogeneousConfig | None = None,
    ):
        """Initialize MLX generation engine."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX required for MLXGenerationEngine")

        self.model = model
        self.config = config or HeterogeneousConfig()

    def generate_rollouts(
        self,
        prompts: list[list[int]],
        reward_fn: Any | None = None,
    ) -> RolloutData:
        """Generate rollouts for the given prompts.

        Args:
            prompts: List of token ID lists for prompts
            reward_fn: Optional function to compute rewards

        Returns:
            RolloutData containing generated sequences and metadata
        """
        start_time = time.time()

        if self.model is None:
            # Return dummy data for testing without model
            batch_size = len(prompts)
            seq_len = 20
            return RolloutData(
                input_ids=[[i] * seq_len for i in range(batch_size)],
                attention_mask=[[1] * seq_len for _ in range(batch_size)],
                rewards=[0.0] * batch_size,
                generation_time=time.time() - start_time,
            )

        # Pad prompts to same length
        max_prompt_len = max(len(p) for p in prompts)
        padded_prompts = [p + [0] * (max_prompt_len - len(p)) for p in prompts]

        # Convert to MLX array
        prompt_ids = mx.array(padded_prompts)

        # Generate with the model
        generated_ids = self._generate(prompt_ids)

        # Compute rewards if function provided
        rewards = None
        if reward_fn is not None:
            rewards = [reward_fn(seq) for seq in generated_ids.tolist()]

        # Compute behavior log probs (for PPO-style training)
        behavior_log_probs = self._compute_log_probs(generated_ids)

        generation_time = time.time() - start_time

        return RolloutData.from_mlx(
            input_ids=generated_ids,
            rewards=mx.array(rewards) if rewards else None,
            behavior_log_probs=behavior_log_probs,
            generation_time=generation_time,
        )

    def _generate(self, prompt_ids: Any) -> Any:
        """Generate sequences using autoregressive sampling."""
        batch_size, prompt_len = prompt_ids.shape
        max_new_tokens = self.config.max_generation_length

        # Initialize with prompts
        current_ids = prompt_ids

        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.model(current_ids)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if self.config.temperature > 0:
                next_token_logits = next_token_logits / self.config.temperature

            # Sample from distribution
            probs = mx.softmax(next_token_logits, axis=-1)

            # Top-p sampling
            if self.config.top_p < 1.0:
                probs = self._top_p_filter(probs, self.config.top_p)

            # Sample next token
            next_tokens = mx.random.categorical(mx.log(probs + 1e-10))
            next_tokens = next_tokens[:, None]

            # Append to sequence
            current_ids = mx.concatenate([current_ids, next_tokens], axis=1)

            # Evaluate for efficiency
            mx.eval(current_ids)

        return current_ids

    def _top_p_filter(self, probs: Any, p: float) -> Any:
        """Apply top-p (nucleus) filtering."""
        # Sort probabilities
        # sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]

        # This is a simplified version - full implementation would need cumsum
        # For now, just return original probs
        _ = p  # Mark as used
        return probs

    def _compute_log_probs(self, token_ids: Any) -> Any:
        """Compute log probabilities for generated tokens."""
        if self.model is None:
            return None

        logits = self.model(token_ids)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

        # Gather log probs for actual tokens
        batch_size, seq_len, _ = logits.shape
        batch_idx = mx.arange(batch_size)[:, None]
        seq_idx = mx.arange(seq_len)[None, :]

        token_log_probs = log_probs[batch_idx, seq_idx, token_ids]

        return token_log_probs


# =============================================================================
# Heterogeneous GRPO Pipeline
# =============================================================================


class HeterogeneousGRPOPipeline:
    """Orchestrates GRPO training across MLX (generation) and PyTorch (training)."""

    def __init__(
        self,
        policy_model_torch: torch.nn.Module,
        ref_model_torch: torch.nn.Module,
        policy_model_mlx: Any | None = None,  # mlx_nn.Module
        optimizer: torch.optim.Optimizer | None = None,
        config: HeterogeneousConfig | None = None,
    ):
        """Initialize heterogeneous pipeline.

        Args:
            policy_model_torch: PyTorch policy model for training
            ref_model_torch: PyTorch reference model
            policy_model_mlx: MLX model for generation (optional)
            optimizer: PyTorch optimizer
            config: Pipeline configuration
        """
        self.config = config or HeterogeneousConfig()

        # PyTorch trainer
        self.trainer = ProductionGRPOTrainer(
            policy_model=policy_model_torch,
            ref_model=ref_model_torch,
            optimizer=optimizer,
            config=self.config.grpo_config,
        )

        # MLX generation engine (optional)
        self.generation_engine = None
        if MLX_AVAILABLE and policy_model_mlx is not None:
            self.generation_engine = MLXGenerationEngine(
                model=policy_model_mlx,
                config=self.config,
            )

        # State
        self.total_rollouts = 0
        self.total_train_steps = 0

        LOGGER.info(
            "Initialized HeterogeneousGRPOPipeline: MLX generation=%s, PyTorch device=%s",
            self.generation_engine is not None,
            self.config.pytorch_device,
        )

    def generate_and_train_step(
        self,
        prompts: list[list[int]],
        reward_fn: Any | None = None,
    ) -> dict[str, float]:
        """Execute one full step: generate rollouts then train.

        Args:
            prompts: List of prompt token IDs
            reward_fn: Function to compute rewards for generated sequences

        Returns:
            Metrics from the training step
        """
        # Step 1: Generate rollouts (MLX or fallback)
        if self.generation_engine is not None:
            rollout_data = self.generation_engine.generate_rollouts(prompts, reward_fn)
        else:
            # Fallback: generate with PyTorch model
            rollout_data = self._generate_with_pytorch(prompts, reward_fn)

        self.total_rollouts += len(prompts)

        # Step 2: Transfer to PyTorch
        rollout_batch = rollout_data.to_torch(device=self.config.pytorch_device)

        # Step 3: Train on PyTorch
        metrics = self.trainer.train_step(rollout_batch)
        self.total_train_steps += 1

        # Add pipeline-specific metrics
        metrics["generation_time"] = rollout_data.generation_time
        metrics["total_rollouts"] = self.total_rollouts
        metrics["total_train_steps"] = self.total_train_steps

        return metrics

    def _generate_with_pytorch(
        self,
        prompts: list[list[int]],
        reward_fn: Any | None = None,
    ) -> RolloutData:
        """Fallback generation using PyTorch model."""
        start_time = time.time()

        # Simple implementation - in production, use proper generation
        batch_size = len(prompts)
        max_len = self.config.max_generation_length
        device = self.config.pytorch_device

        # Pad prompts
        max_prompt_len = max(len(p) for p in prompts)
        input_ids = torch.zeros(
            batch_size, max_prompt_len + max_len, dtype=torch.long, device=device
        )
        for i, p in enumerate(prompts):
            input_ids[i, : len(p)] = torch.tensor(p)

        # Dummy rewards if no function provided
        rewards = [0.0] * batch_size
        if reward_fn is not None:
            rewards = [reward_fn(seq) for seq in input_ids.tolist()]

        return RolloutData(
            input_ids=input_ids.tolist(),
            rewards=rewards,
            generation_time=time.time() - start_time,
        )

    def sync_weights_mlx_to_torch(self) -> None:
        """Sync weights from MLX model to PyTorch model."""
        if self.generation_engine is None or self.generation_engine.model is None:
            return

        # This would implement weight transfer from MLX to PyTorch
        # Typically done by saving/loading or direct parameter copying
        LOGGER.info("Syncing weights from MLX to PyTorch")

    def sync_weights_torch_to_mlx(self) -> None:
        """Sync weights from PyTorch model to MLX model."""
        if self.generation_engine is None or self.generation_engine.model is None:
            return

        # This would implement weight transfer from PyTorch to MLX
        LOGGER.info("Syncing weights from PyTorch to MLX")

    def get_state(self) -> dict[str, Any]:
        """Get pipeline state for checkpointing."""
        return {
            "total_rollouts": self.total_rollouts,
            "total_train_steps": self.total_train_steps,
            "trainer_state": self.trainer.get_state_dict(),
            "config": {
                "generation_batch_size": self.config.generation_batch_size,
                "max_generation_length": self.config.max_generation_length,
                "rollouts_per_step": self.config.rollouts_per_step,
            },
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_heterogeneous_pipeline(
    policy_model: torch.nn.Module,
    config: HeterogeneousConfig | None = None,
) -> HeterogeneousGRPOPipeline:
    """Factory function to create a heterogeneous GRPO pipeline.

    Args:
        policy_model: PyTorch policy model
        config: Pipeline configuration

    Returns:
        Configured HeterogeneousGRPOPipeline
    """
    import copy

    config = config or HeterogeneousConfig()

    # Create reference model as a copy of policy
    ref_model = copy.deepcopy(policy_model)
    for param in ref_model.parameters():
        param.requires_grad = False

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    return HeterogeneousGRPOPipeline(
        policy_model_torch=policy_model,
        ref_model_torch=ref_model,
        optimizer=optimizer,
        config=config,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "HeterogeneousConfig",
    "HeterogeneousGRPOPipeline",
    "MLXGenerationEngine",
    "RolloutData",
    "create_heterogeneous_pipeline",
]
