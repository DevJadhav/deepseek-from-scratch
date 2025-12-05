"""
Direct Preference Optimization (DPO) for DeepSeek

This module implements:
- DPO training algorithm
- DPO variants (IPO, KTO, ORPO)
- Reference model handling
- Preference dataset processing

Based on: "Direct Preference Optimization" (https://arxiv.org/abs/2305.18290)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Literal
import os
import copy


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # DPO parameters
    beta: float = 0.1  # KL penalty coefficient
    label_smoothing: float = 0.0  # Label smoothing for stability
    loss_type: Literal["sigmoid", "ipo", "kto"] = "sigmoid"
    reference_free: bool = False  # Use reference-free variant
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_length: int = 2048
    max_prompt_length: int = 512
    
    # Regularization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Checkpointing
    save_steps: int = 500
    save_dir: str = "./dpo_checkpoints"
    
    # Logging
    log_steps: int = 10


# =============================================================================
# Dataset
# =============================================================================

class DPODataset(Dataset):
    """
    Dataset for DPO training.
    
    Expects data in format:
    [
        {
            "prompt": "...",
            "chosen": "...",
            "rejected": "..."
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _tokenize_pair(
        self,
        prompt: str,
        response: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a prompt-response pair and create labels."""
        # Tokenize prompt
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=True,
        )
        prompt_ids = prompt_enc["input_ids"]
        
        # Tokenize response
        response_enc = self.tokenizer(
            response,
            max_length=self.max_length - len(prompt_ids),
            truncation=True,
            add_special_tokens=False,
        )
        response_ids = response_enc["input_ids"]
        
        # Combine
        input_ids = prompt_ids + response_ids
        
        # Pad
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
        else:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * min(len(prompt_ids) + len(response_ids), self.max_length)
        attention_mask += [0] * (self.max_length - len(attention_mask))
        
        # Create labels (mask prompt tokens with -100)
        labels = [-100] * len(prompt_ids)
        labels += response_ids[:self.max_length - len(prompt_ids)]
        labels += [-100] * (self.max_length - len(labels))
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize chosen and rejected
        chosen_ids, chosen_mask, chosen_labels = self._tokenize_pair(prompt, chosen)
        rejected_ids, rejected_mask, rejected_labels = self._tokenize_pair(prompt, rejected)
        
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels,
        }


# =============================================================================
# DPO Trainer
# =============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    DPO loss:
    L = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    
    Where:
    - π = policy model
    - π_ref = reference model (frozen)
    - y_w = chosen response
    - y_l = rejected response
    - β = KL penalty coefficient
    """
    
    def __init__(
        self,
        policy: nn.Module,
        reference: Optional[nn.Module],
        tokenizer,
        config: DPOConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        
        # Policy model
        self.policy = policy.to(device)
        
        # Reference model (frozen copy)
        if reference is None and not config.reference_free:
            reference = copy.deepcopy(policy)
        
        if reference is not None:
            self.reference = reference.to(device)
            self.reference.eval()
            for param in self.reference.parameters():
                param.requires_grad = False
        else:
            self.reference = None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # State
        self.global_step = 0
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        print(f"DPOTrainer initialized on {device}")
        print(f"Loss type: {config.loss_type}, Beta: {config.beta}")
        if config.reference_free:
            print("Using reference-free DPO")
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of responses.
        
        Args:
            model: The model to compute log probs for
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Labels with -100 for masked positions (batch, seq_len)
        
        Returns:
            Sequence log probabilities (batch,)
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs of actual tokens
        # shift_labels shape: (batch, seq-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1).clamp(min=0),
        ).squeeze(-1)
        
        # Mask for valid labels (not -100)
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask
        
        # Sum over sequence to get sequence log prob
        seq_log_probs = token_log_probs.sum(dim=-1)
        
        # Optionally normalize by length
        # seq_lengths = mask.sum(dim=-1)
        # seq_log_probs = seq_log_probs / seq_lengths.clamp(min=1)
        
        return seq_log_probs
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Standard DPO:
        L = -log σ(β * (log π/π_ref(chosen) - log π/π_ref(rejected)))
        """
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO logits
        logits = self.config.beta * (chosen_logratios - rejected_logratios)
        
        # Compute loss based on type
        if self.config.loss_type == "sigmoid":
            # Standard DPO
            losses = -F.logsigmoid(logits)
        
        elif self.config.loss_type == "ipo":
            # Identity Preference Optimization
            # L = (logits - 1/(2β))²
            losses = (logits - 1 / (2 * self.config.beta)) ** 2
        
        elif self.config.loss_type == "kto":
            # Kahneman-Tversky Optimization (asymmetric)
            # Different treatment for chosen vs rejected
            chosen_losses = 1 - F.sigmoid(self.config.beta * chosen_logratios)
            rejected_losses = 1 - F.sigmoid(-self.config.beta * rejected_logratios)
            losses = chosen_losses + rejected_losses
        
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        loss = losses.mean()
        
        # Label smoothing
        if self.config.label_smoothing > 0 and self.config.loss_type == "sigmoid":
            smooth_loss = -F.logsigmoid(-logits).mean()
            loss = (1 - self.config.label_smoothing) * loss + \
                   self.config.label_smoothing * smooth_loss
        
        # Compute metrics
        chosen_rewards = self.config.beta * chosen_logratios.detach()
        rejected_rewards = self.config.beta * rejected_logratios.detach()
        
        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
            "chosen_logratio": chosen_logratios.mean().item(),
            "rejected_logratio": rejected_logratios.mean().item(),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single DPO training step."""
        self.policy.train()
        
        # Move to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        
        # Compute policy log probs
        policy_chosen_logps = self.compute_log_probs(
            self.policy, chosen_ids, chosen_mask, chosen_labels
        )
        policy_rejected_logps = self.compute_log_probs(
            self.policy, rejected_ids, rejected_mask, rejected_labels
        )
        
        # Compute reference log probs
        with torch.no_grad():
            if self.reference is not None:
                ref_chosen_logps = self.compute_log_probs(
                    self.reference, chosen_ids, chosen_mask, chosen_labels
                )
                ref_rejected_logps = self.compute_log_probs(
                    self.reference, rejected_ids, rejected_mask, rejected_labels
                )
            else:
                # Reference-free: use zeros
                ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        
        # Compute DPO loss
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        
        return metrics
    
    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ):
        """Full DPO training loop."""
        dataset = DPODataset(
            train_data,
            self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                
                if self.global_step % self.config.log_steps == 0:
                    print(
                        f"  Step {self.global_step:5d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc: {metrics['accuracy']:.2%} | "
                        f"Margin: {metrics['reward_margin']:.3f}"
                    )
                
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # Evaluate
            if eval_data:
                eval_metrics = self.evaluate(eval_data)
                print(f"  Eval - Acc: {eval_metrics['accuracy']:.2%}")
        
        self.save_checkpoint(final=True)
        print("\nDPO training complete!")
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_data: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate on held-out data."""
        self.policy.eval()
        
        dataset = DPODataset(
            eval_data,
            self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        total_correct = 0
        total_samples = 0
        total_margin = 0.0
        
        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_mask = batch["chosen_attention_mask"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)
            
            # Policy log probs
            policy_chosen = self.compute_log_probs(
                self.policy, chosen_ids, chosen_mask, chosen_labels
            )
            policy_rejected = self.compute_log_probs(
                self.policy, rejected_ids, rejected_mask, rejected_labels
            )
            
            # Reference log probs
            if self.reference is not None:
                ref_chosen = self.compute_log_probs(
                    self.reference, chosen_ids, chosen_mask, chosen_labels
                )
                ref_rejected = self.compute_log_probs(
                    self.reference, rejected_ids, rejected_mask, rejected_labels
                )
            else:
                ref_chosen = torch.zeros_like(policy_chosen)
                ref_rejected = torch.zeros_like(policy_rejected)
            
            # Compute implicit rewards
            chosen_rewards = self.config.beta * (policy_chosen - ref_chosen)
            rejected_rewards = self.config.beta * (policy_rejected - ref_rejected)
            
            correct = (chosen_rewards > rejected_rewards).sum().item()
            margin = (chosen_rewards - rejected_rewards).sum().item()
            
            total_correct += correct
            total_margin += margin
            total_samples += chosen_ids.size(0)
        
        return {
            "accuracy": total_correct / max(total_samples, 1),
            "margin": total_margin / max(total_samples, 1),
        }
    
    def save_checkpoint(self, final: bool = False):
        """Save checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.save_dir, f"dpo_checkpoint_{suffix}.pt")
        
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config.__dict__,
        }, path)
        
        print(f"Checkpoint saved: {path}")


# =============================================================================
# DPO Variants
# =============================================================================

class IPOTrainer(DPOTrainer):
    """
    Identity Preference Optimization.
    
    Uses L = (logits - 1/(2β))² instead of log-sigmoid.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.loss_type = "ipo"


class KTOTrainer(DPOTrainer):
    """
    Kahneman-Tversky Optimization.
    
    Asymmetric treatment of preferred vs rejected.
    Works with unpaired data.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.loss_type = "kto"


class ORPOTrainer:
    """
    Odds Ratio Preference Optimization.
    
    Reference-free variant using odds ratio.
    
    L = -log σ(log(odds(chosen)/odds(rejected)))
    
    Where odds(y|x) = P(y|x) / (1 - P(y|x))
    """
    
    def __init__(
        self,
        policy: nn.Module,
        tokenizer,
        config: DPOConfig,
        device: Optional[torch.device] = None,
        lambda_orpo: float = 0.1,
    ):
        self.policy = policy
        self.tokenizer = tokenizer
        self.config = config
        self.lambda_orpo = lambda_orpo
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.policy = policy.to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.global_step = 0
        os.makedirs(config.save_dir, exist_ok=True)
    
    def compute_orpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        chosen_nll: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ORPO loss.
        
        L = NLL(chosen) + λ * L_orpo
        L_orpo = -log σ(log(odds(chosen)) - log(odds(rejected)))
        """
        # Log odds = log(p / (1-p)) ≈ log_prob for language models
        log_odds_chosen = chosen_logps
        log_odds_rejected = rejected_logps
        
        # ORPO loss
        orpo_logits = log_odds_chosen - log_odds_rejected
        orpo_loss = -F.logsigmoid(orpo_logits).mean()
        
        # Total loss = NLL + λ * ORPO
        total_loss = chosen_nll + self.lambda_orpo * orpo_loss
        
        metrics = {
            "loss": total_loss.item(),
            "nll": chosen_nll.item(),
            "orpo_loss": orpo_loss.item(),
            "accuracy": (chosen_logps > rejected_logps).float().mean().item(),
        }
        
        return total_loss, metrics


# =============================================================================
# Demo
# =============================================================================


