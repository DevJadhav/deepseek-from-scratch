"""
Reward Model - MLX Implementation
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward model."""
    hidden_size: int = 4096
    dropout: float = 0.1
    pooling: str = "last"  # "last", "mean", "max"


class RewardModel(nn.Module):
    """
    Reward model for preference learning.
    
    Predicts scalar rewards for input sequences.
    P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
    """
    
    def __init__(self, base_model: nn.Module, config: RewardConfig):
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
        
    def get_last_hidden_state(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        """Pool hidden states based on config."""
        if self.config.pooling == "last":
            # Get last non-padded position
            seq_lengths = mx.sum(attention_mask, axis=1).astype(mx.int32) - 1
            batch_size = hidden_states.shape[0]
            
            # Use gather for last token
            batch_idx = mx.arange(batch_size)
            last_hidden = hidden_states[batch_idx, seq_lengths]
            return last_hidden
            
        elif self.config.pooling == "mean":
            # Mean pool over non-padded positions
            mask = attention_mask[:, :, None]
            pooled = mx.sum(hidden_states * mask, axis=1) / mx.sum(mask, axis=1)
            return pooled
            
        elif self.config.pooling == "max":
            # Max pool
            large_neg = -1e9
            mask = attention_mask[:, :, None]
            masked = hidden_states * mask + (1 - mask) * large_neg
            return mx.max(masked, axis=1)
            
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq) token ids
            attention_mask: (batch, seq) attention mask
            
        Returns:
            (batch,) scalar rewards
        """
        # Get hidden states from base model
        outputs = self.base_model(input_ids)
        if isinstance(outputs, dict):
            hidden_states = outputs["hidden_states"]
        else:
            hidden_states = outputs
            
        # Pool hidden states
        pooled = self.get_last_hidden_state(hidden_states, attention_mask)
        
        # Get reward
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)

class RewardModelSimple(nn.Module):
    """
    Simplified Reward Model for demo purposes.
    Contains its own small transformer.
    """
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # Simple transformer block
        self.layer = nn.TransformerEncoderLayer(hidden_size, num_heads) # MLX has this? No.
        # MLX nn.TransformerEncoderLayer exists in newer versions or we build it.
        # Let's use a simple Sequential of Linear/Relu for demo if Transformer is complex to import
        # Or just use nn.MultiHeadAttention if available.
        
        # Actually MLX has nn.TransformerEncoderLayer.
        self.layers = [
            nn.TransformerEncoderLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ]
        self.head = nn.Linear(hidden_size, 1)
        
    def __call__(self, input_ids):
        x = self.embed(input_ids)
        # No mask for simplicity in demo
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # Pool (mean)
        x = mx.mean(x, axis=1)
        return self.head(x).squeeze(-1)
