"""
Reward Model for DeepSeek

This module implements:
- Reward model architecture
- Bradley-Terry preference learning
- Training and evaluation

Based on: RLHF literature and DeepSeek alignment approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import os


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RewardConfig:
    """Configuration for reward model."""
    # Architecture
    hidden_size: int = 4096
    num_labels: int = 1
    pooling: str = "last"  # "last", "mean", "max"
    
    # Training
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    
    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    
    # Loss
    margin: float = 0.0  # Margin for ranking loss
    
    # Checkpointing
    save_steps: int = 500
    save_dir: str = "./reward_checkpoints"
    
    # Logging
    log_steps: int = 10


# =============================================================================
# Reward Model
# =============================================================================

class RewardModel(nn.Module):
    """
    Reward model for preference learning.
    
    Architecture:
    - Transformer backbone (shared with LLM)
    - Reward head (linear projection to scalar)
    
    The model learns to predict scalar rewards such that:
    P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: RewardConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 4, 1),
        )
        
        # Initialize reward head
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_last_hidden_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get hidden state of last non-padding token."""
        batch_size = hidden_states.size(0)
        
        # Find last non-padding position
        seq_lengths = attention_mask.sum(dim=1) - 1
        
        # Gather last hidden states
        last_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            seq_lengths,
        ]
        
        return last_hidden
    
    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool hidden states according to config."""
        if self.config.pooling == "last":
            return self.get_last_hidden_state(hidden_states, attention_mask)
        
        elif self.config.pooling == "mean":
            # Mean pooling over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            return pooled
        
        elif self.config.pooling == "max":
            # Max pooling over non-padding tokens
            mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states.masked_fill(~mask.bool(), float('-inf'))
            pooled = hidden_states.max(dim=1)[0]
            return pooled
        
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Compute scalar reward for each sequence.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            return_hidden: Whether to also return hidden states
        
        Returns:
            rewards: Scalar rewards (batch,)
            hidden: Optional hidden states (batch, hidden_size)
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last layer hidden states
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs.last_hidden_state
        
        # Pool hidden states
        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        
        # Compute reward
        rewards = self.reward_head(pooled).squeeze(-1)
        
        if return_hidden:
            return rewards, pooled
        
        return rewards
    
    def compute_preference_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry preference loss.
        
        L = -log(sigmoid(r_chosen - r_rejected - margin))
        """
        logits = chosen_rewards - rejected_rewards - margin
        loss = -F.logsigmoid(logits).mean()
        return loss


class RewardModelSimple(nn.Module):
    """
    Simplified reward model for demonstration.
    Uses a small transformer as backbone.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(4096, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self.hidden_size = hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Create attention mask for transformer
        src_key_padding_mask = ~attention_mask.bool()
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Get last token
        seq_lengths = (attention_mask.sum(dim=1) - 1).long()
        last_hidden = x[torch.arange(batch_size, device=x.device), seq_lengths]
        
        # Reward
        rewards = self.reward_head(last_hidden).squeeze(-1)
        
        return rewards


# =============================================================================
# Dataset
# =============================================================================

class PreferenceDataset(Dataset):
    """
    Dataset for preference learning.
    
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
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize chosen
        chosen_text = f"{prompt} {chosen}"
        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize rejected
        rejected_text = f"{prompt} {rejected}"
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# =============================================================================
# Trainer
# =============================================================================

class RewardTrainer:
    """
    Trainer for reward model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: RewardConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        
        self.model = model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # State
        self.global_step = 0
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        print(f"RewardTrainer initialized on {device}")
    
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute preference loss and metrics.
        """
        # Get rewards
        chosen_rewards = self.model(chosen_ids, chosen_mask)
        rejected_rewards = self.model(rejected_ids, rejected_mask)
        
        # Bradley-Terry loss with margin
        logits = chosen_rewards - rejected_rewards - self.config.margin
        loss = -F.logsigmoid(logits).mean()
        
        # Metrics
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        
        # Compute loss
        loss, metrics = self.compute_loss(
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
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
        tokenizer,
        eval_data: Optional[List[Dict]] = None,
    ):
        """Full training loop."""
        dataset = PreferenceDataset(
            train_data, tokenizer, max_length=self.config.max_length
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
                eval_metrics = self.evaluate(eval_data, tokenizer)
                print(f"  Eval - Acc: {eval_metrics['accuracy']:.2%}")
        
        self.save_checkpoint(final=True)
        print("\nTraining complete!")
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_data: List[Dict],
        tokenizer,
    ) -> Dict[str, float]:
        """Evaluate on held-out data."""
        self.model.eval()
        
        dataset = PreferenceDataset(
            eval_data, tokenizer, max_length=self.config.max_length
        )
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)
            
            chosen_rewards = self.model(chosen_ids, chosen_mask)
            rejected_rewards = self.model(rejected_ids, rejected_mask)
            
            correct = (chosen_rewards > rejected_rewards).sum().item()
            total_correct += correct
            total_samples += chosen_ids.size(0)
        
        return {"accuracy": total_correct / max(total_samples, 1)}
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.save_dir, f"reward_checkpoint_{suffix}.pt")
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config.__dict__,
        }, path)
        
        print(f"Checkpoint saved: {path}")
    
    def predict(
        self,
        text: str,
        tokenizer,
    ) -> float:
        """Predict reward for a single text."""
        self.model.eval()
        
        encoding = tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            reward = self.model(input_ids, attention_mask)
        
        return reward.item()


# =============================================================================
# Demo
# =============================================================================


