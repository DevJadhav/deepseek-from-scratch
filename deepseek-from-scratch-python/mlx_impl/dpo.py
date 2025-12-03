"""
DPO (Direct Preference Optimization) - MLX Implementation
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # "sigmoid", "ipo", "kto"


class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, model=None, ref_model=None, config: DPOConfig = None):
        self.model = model
        self.ref_model = ref_model
        self.config = config or DPOConfig()
        
    def compute_dpo_loss(
        self,
        policy_chosen_logps: mx.array,
        policy_rejected_logps: mx.array,
        ref_chosen_logps: mx.array,
        ref_rejected_logps: mx.array,
    ) -> Tuple[mx.array, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Standard DPO:
        L = -log σ(β * (log π/π_ref(chosen) - log π/π_ref(rejected)))
        
        Args:
            policy_chosen_logps: Log probs of chosen responses under policy
            policy_rejected_logps: Log probs of rejected responses under policy
            ref_chosen_logps: Log probs of chosen responses under reference
            ref_rejected_logps: Log probs of rejected responses under reference
            
        Returns:
            Tuple of (loss, metrics dict)
        """
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO logits
        logits = self.config.beta * (chosen_logratios - rejected_logratios)
        
        # Compute loss based on type
        if self.config.loss_type == "sigmoid":
            # Standard DPO: -log σ(logits)
            losses = -mx.log(mx.sigmoid(logits) + 1e-10)
            
        elif self.config.loss_type == "ipo":
            # Identity Preference Optimization
            losses = (logits - 1 / (2 * self.config.beta)) ** 2
            
        elif self.config.loss_type == "kto":
            # Kahneman-Tversky Optimization
            chosen_losses = 1 - mx.sigmoid(self.config.beta * chosen_logratios)
            rejected_losses = 1 - mx.sigmoid(-self.config.beta * rejected_logratios)
            losses = chosen_losses + rejected_losses
            
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        loss = mx.mean(losses)
        
        # Label smoothing
        if self.config.label_smoothing > 0 and self.config.loss_type == "sigmoid":
            smooth_loss = -mx.mean(mx.log(mx.sigmoid(-logits) + 1e-10))
            loss = (1 - self.config.label_smoothing) * loss + \
                   self.config.label_smoothing * smooth_loss
        
        # Compute metrics
        chosen_rewards = self.config.beta * chosen_logratios
        rejected_rewards = self.config.beta * rejected_logratios
        
        metrics = {
            "loss": float(loss),
            "chosen_reward": float(mx.mean(chosen_rewards)),
            "rejected_reward": float(mx.mean(rejected_rewards)),
            "reward_margin": float(mx.mean(chosen_rewards - rejected_rewards)),
            "accuracy": float(mx.mean((chosen_rewards > rejected_rewards).astype(mx.float32))),
        }
        
        return loss, metrics
