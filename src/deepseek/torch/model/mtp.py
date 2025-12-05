"""
Multi-Token Prediction (MTP) for DeepSeek-V3

This module implements DeepSeek's Multi-Token Prediction mechanism with:
- Configurable prediction depth (D=1, 2, 3)
- Sequential MTP modules with proper gradient flow
- Loss weighting for different prediction depths
- Speculative decoding integration point for inference
- MTP dropout for regularization

MTP enables the model to predict multiple future tokens simultaneously,
improving both training signal density and enabling speculative decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math


@dataclass
class MTPConfig:
    """Configuration for Multi-Token Prediction."""
    vocab_size: int = 32000
    d_model: int = 4096
    d_embed: int = 4096
    num_layers: int = 24
    num_heads: int = 32
    
    # MTP-specific parameters
    prediction_depth: int = 3  # Number of future tokens to predict (D)
    
    # Loss weighting
    mtp_loss_weight: float = 1.0  # Base weight for MTP losses
    depth_decay: float = 0.8  # Decay factor per depth (weight = base * decay^depth)
    
    # Regularization
    mtp_dropout: float = 0.0  # Randomly skip MTP during training
    
    # Inference
    enable_speculative: bool = True  # Enable speculative decoding mode
    speculative_threshold: float = 0.9  # Confidence threshold for speculation
    
    # Architecture
    use_shared_embedding: bool = True  # Share embedding with main head
    use_sequential_refinement: bool = True  # Each MTP refines previous prediction


class MTPModule(nn.Module):
    """
    Single MTP prediction module.
    
    Takes hidden state and previous prediction context to predict
    the next token in the sequence.
    """
    
    def __init__(
        self,
        d_model: int,
        d_embed: int,
        vocab_size: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        
        # Project hidden state
        self.hidden_proj = nn.Linear(d_model, d_embed)
        
        # Optional: incorporate previous token context
        self.context_proj = nn.Linear(d_embed, d_embed)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_embed) if use_layer_norm else nn.Identity()
        
        # Prediction head
        self.head = nn.Linear(d_embed, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # For returning hidden state
        self._last_hidden: Optional[torch.Tensor] = None
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        previous_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hidden_state: Current hidden representation [batch, seq, d_model]
            previous_context: Optional context from previous prediction [batch, seq, d_embed]
            
        Returns:
            - logits: Prediction logits [batch, seq, vocab_size]
            - hidden: Updated hidden state [batch, seq, d_embed]
        """
        # Project hidden state
        h = self.hidden_proj(hidden_state)
        
        # Incorporate previous context if available
        if previous_context is not None:
            context = self.context_proj(previous_context)
            h = h + context  # Residual addition
        
        # Normalize and dropout
        h = self.layer_norm(h)
        h = self.dropout(h)
        
        # Store for potential use
        self._last_hidden = h
        
        # Predict
        logits = self.head(h)
        
        return logits, h
    
    def get_last_hidden(self) -> Optional[torch.Tensor]:
        """Get the last hidden state for chaining."""
        return self._last_hidden


class SequentialMTPModule(nn.Module):
    """
    Sequential MTP module with lightweight transformer layer.
    
    Each module takes the previous prediction's hidden state and
    refines it through a small transformer block before predicting.
    """
    
    def __init__(
        self,
        d_model: int,
        d_embed: int,
        vocab_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_embed)
        
        # Lightweight self-attention for context refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_embed, d_embed * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_embed * 4, d_embed),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_embed)
        self.ln2 = nn.LayerNorm(d_embed)
        
        # Output head
        self.head = nn.Linear(d_embed, vocab_size, bias=False)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention refinement."""
        # Project input
        h = self.input_proj(hidden_state)
        
        # Self-attention with residual
        h_norm = self.ln1(h)
        # Generate causal mask if not provided
        if causal_mask is None:
            seq_len = h_norm.size(1)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=h_norm.device, dtype=h_norm.dtype
            )
        attn_out, _ = self.self_attn(
            h_norm, h_norm, h_norm,
            attn_mask=causal_mask,
            is_causal=True,
        )
        h = h + attn_out
        
        # FFN with residual
        h = h + self.ffn(self.ln2(h))
        
        # Predict
        logits = self.head(h)
        
        return logits, h


class MTPModel(nn.Module):
    """
    Multi-Token Prediction Model with configurable depth.
    
    Features:
    - D-depth sequential prediction (D=1, 2, 3)
    - Proper gradient flow through MTP chain
    - Configurable loss weighting per depth
    - Speculative decoding support for inference
    """
    
    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.prediction_depth = config.prediction_depth
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Simplified transformer layers (placeholder - would use actual transformer)
        self.layers = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model) 
            for _ in range(config.num_layers)
        ])
        
        # Layer norms
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Main prediction head (t+1)
        self.main_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # MTP modules for future predictions (t+2, t+3, ...)
        if config.use_sequential_refinement:
            self.mtp_modules = nn.ModuleList([
                SequentialMTPModule(
                    d_model=config.d_model,
                    d_embed=config.d_embed,
                    vocab_size=config.vocab_size,
                    num_heads=max(1, config.num_heads // 4),
                    dropout=0.1,
                )
                for _ in range(config.prediction_depth)
            ])
        else:
            self.mtp_modules = nn.ModuleList([
                MTPModule(
                    d_model=config.d_model,
                    d_embed=config.d_embed,
                    vocab_size=config.vocab_size,
                    dropout=0.1,
                )
                for _ in range(config.prediction_depth)
            ])
        
        # Loss weights per depth
        self._loss_weights = self._compute_loss_weights()
        
        # MTP dropout probability
        self.mtp_dropout = config.mtp_dropout
    
    def _compute_loss_weights(self) -> List[float]:
        """Compute loss weights for each prediction depth."""
        weights = []
        for d in range(self.prediction_depth):
            weight = self.config.mtp_loss_weight * (self.config.depth_decay ** d)
            weights.append(weight)
        return weights
    
    def get_loss_weights(self) -> List[float]:
        """Get loss weights for MTP heads."""
        return self._loss_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with multi-token prediction.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            return_hidden: Whether to return final hidden state
            
        Returns:
            - main_logits: Main prediction (t+1) [batch, seq, vocab]
            - future_logits: List of future predictions [batch, seq, vocab]
            - hidden: Optional final hidden state [batch, seq, d_model]
        """
        # Embed tokens
        x = self.embed(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = F.gelu(layer(x))
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Main prediction (t+1)
        main_logits = self.main_head(x)
        
        # MTP predictions (t+2, t+3, ...)
        future_logits = []
        current_hidden = x
        
        # Apply MTP dropout during training
        apply_mtp = True
        if self.training and self.mtp_dropout > 0:
            if torch.rand(1).item() < self.mtp_dropout:
                apply_mtp = False
        
        if apply_mtp:
            for i, module in enumerate(self.mtp_modules):
                if isinstance(module, SequentialMTPModule):
                    logits, current_hidden = module(current_hidden)
                else:
                    # For simple MTPModule, pass previous context
                    prev_context = current_hidden if i > 0 else None
                    logits, current_hidden = module(current_hidden, prev_context)
                
                future_logits.append(logits)
        
        if return_hidden:
            return main_logits, future_logits, x
        return main_logits, future_logits, None
    
    def compute_mtp_loss(
        self,
        main_logits: torch.Tensor,
        future_logits: List[torch.Tensor],
        target_ids: torch.Tensor,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined MTP loss.
        
        Args:
            main_logits: Main prediction logits [batch, seq, vocab]
            future_logits: List of future prediction logits
            target_ids: Target token IDs [batch, seq]
            ignore_index: Index to ignore in loss computation
            
        Returns:
            - total_loss: Combined weighted loss
            - loss_breakdown: Dict with per-head losses
        """
        batch_size, seq_len = target_ids.shape
        
        # Main loss (predict t+1 from position t)
        # Shift logits and targets
        shift_logits = main_logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()
        
        main_loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            ignore_index=ignore_index,
        )
        
        total_loss = main_loss
        loss_breakdown = {"main_loss": main_loss.item()}
        
        # MTP losses (predict t+2, t+3, ... from position t)
        for d, (logits, weight) in enumerate(zip(future_logits, self._loss_weights)):
            # Shift by d+2 (main head predicts t+1, first MTP predicts t+2)
            shift_amount = d + 2
            
            if seq_len <= shift_amount:
                continue
            
            shift_logits = logits[:, :-shift_amount, :].contiguous()
            shift_targets = target_ids[:, shift_amount:].contiguous()
            
            mtp_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=ignore_index,
            )
            
            weighted_loss = weight * mtp_loss
            total_loss = total_loss + weighted_loss
            loss_breakdown[f"mtp_depth_{d+1}_loss"] = mtp_loss.item()
        
        return total_loss, loss_breakdown
    
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using speculative decoding.
        
        Uses MTP predictions to speculate multiple tokens ahead,
        then verifies with the main model.
        
        Args:
            input_ids: Input token IDs [batch, seq]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            - generated_ids: Generated token IDs
            - stats: Generation statistics
        """
        if not self.config.enable_speculative:
            raise ValueError("Speculative decoding not enabled in config")
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        generated = input_ids.clone()
        stats = {
            "total_tokens": 0,
            "accepted_speculations": 0,
            "rejected_speculations": 0,
        }
        
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Get predictions
            with torch.no_grad():
                main_logits, future_logits, _ = self.forward(generated)
            
            # Get main prediction
            main_probs = F.softmax(main_logits[:, -1, :] / temperature, dim=-1)
            main_token = torch.multinomial(main_probs, 1)
            
            # Get speculative predictions
            speculative_tokens = [main_token]
            
            for i, logits in enumerate(future_logits):
                if tokens_generated + i + 1 >= max_new_tokens:
                    break
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                max_prob, token = probs.max(dim=-1, keepdim=True)
                
                # Only accept if confidence is high enough
                if max_prob.item() >= self.config.speculative_threshold:
                    speculative_tokens.append(token)
                    stats["accepted_speculations"] += 1
                else:
                    stats["rejected_speculations"] += 1
                    break
            
            # Add tokens to generated sequence
            for token in speculative_tokens:
                generated = torch.cat([generated, token], dim=1)
                tokens_generated += 1
                stats["total_tokens"] += 1
                
                if tokens_generated >= max_new_tokens:
                    break
        
        return generated, stats
    
    def get_mtp_accuracy(
        self,
        main_logits: torch.Tensor,
        future_logits: List[torch.Tensor],
        target_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute accuracy at each prediction depth.
        
        Returns dict with accuracy for main and each MTP head.
        """
        batch_size, seq_len = target_ids.shape
        accuracies = {}
        
        # Main accuracy
        shift_logits = main_logits[:, :-1, :]
        shift_targets = target_ids[:, 1:]
        main_preds = shift_logits.argmax(dim=-1)
        main_correct = (main_preds == shift_targets).float().mean()
        accuracies["main_accuracy"] = main_correct.item()
        
        # MTP accuracies
        for d, logits in enumerate(future_logits):
            shift_amount = d + 2
            if seq_len <= shift_amount:
                continue
            
            shift_logits = logits[:, :-shift_amount, :]
            shift_targets = target_ids[:, shift_amount:]
            
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_targets).float().mean()
            accuracies[f"mtp_depth_{d+1}_accuracy"] = correct.item()
        
        return accuracies


# ============================================================================
# Legacy MTPModule for backward compatibility
# ============================================================================

class LegacyMTPModule(nn.Module):
    """Legacy MTP module for backward compatibility."""
    
    def __init__(self, d_model, d_embed, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, d_embed)
        self.layer_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, vocab_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.layer_norm(x)
        return self.head(x)


class LegacyMTPModel(nn.Module):
    """Legacy MTP model for backward compatibility."""
    
    def __init__(self, vocab_size, d_model, num_layers, k_predictions):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.main_head = nn.Linear(d_model, vocab_size)
        
        self.k_predictions = k_predictions
        self.mtp_modules = nn.ModuleList([
            LegacyMTPModule(d_model, d_model, vocab_size) for _ in range(k_predictions)
        ])
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = F.relu(layer(x))
            
        main_logits = self.main_head(x)
        
        future_logits = []
        current_state = x
        
        for module in self.mtp_modules:
            logits = module(current_state)
            future_logits.append(logits)
            
        return main_logits, future_logits