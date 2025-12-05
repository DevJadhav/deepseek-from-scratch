"""
Multi-Token Prediction (MTP) for MLX

This module implements DeepSeek's Multi-Token Prediction mechanism with:
- Configurable prediction depth (D=1, 2, 3)
- Sequential MTP modules with proper gradient flow
- Loss weighting for different prediction depths
- Speculative decoding integration point for inference
- MTP dropout for regularization

Optimized for Apple Silicon using MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class MTPConfig:
    """Configuration for Multi-Token Prediction."""
    vocab_size: int = 32000
    d_model: int = 4096
    d_embed: Optional[int] = None  # If None, defaults to d_model
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
    
    def __post_init__(self):
        """Set d_embed to d_model if not specified."""
        if self.d_embed is None:
            self.d_embed = self.d_model
    
    def get_loss_weights(self) -> List[float]:
        """Get loss weights for each prediction depth."""
        return [
            self.mtp_loss_weight * (self.depth_decay ** d)
            for d in range(self.prediction_depth)
        ]
    
    @classmethod
    def small(cls) -> "MTPConfig":
        """Small config for testing."""
        return cls(
            vocab_size=1000,
            d_model=256,
            d_embed=256,
            num_layers=4,
            num_heads=4,
            prediction_depth=2,
        )
    
    @classmethod
    def medium(cls) -> "MTPConfig":
        """Medium config."""
        return cls(
            vocab_size=16000,
            d_model=1024,
            d_embed=1024,
            num_layers=12,
            num_heads=16,
            prediction_depth=3,
        )


@dataclass
class MTPMetrics:
    """Metrics for tracking MTP performance."""
    depth_accuracy: List[float] = field(default_factory=list)
    depth_loss: List[float] = field(default_factory=list)
    total_samples: int = 0
    speculation_acceptance_rate: float = 0.0
    
    def __post_init__(self):
        if not self.depth_accuracy:
            self.depth_accuracy = []
        if not self.depth_loss:
            self.depth_loss = []
    
    def reset(self, depth: int = 3):
        """Reset metrics for new tracking period."""
        self.depth_accuracy = [0.0] * depth
        self.depth_loss = [0.0] * depth
        self.total_samples = 0
        self.speculation_acceptance_rate = 0.0
    
    def record_loss(self, depth_idx: int, loss: float):
        """Record loss for a specific MTP depth."""
        if depth_idx < len(self.depth_loss):
            # Use exponential moving average
            alpha = 0.1
            self.depth_loss[depth_idx] = (1 - alpha) * self.depth_loss[depth_idx] + alpha * loss
    
    def update(self, depth_losses: List[float], depth_correct: List[int], batch_size: int):
        """Update metrics with new batch results."""
        for i, loss in enumerate(depth_losses):
            if i < len(self.depth_loss):
                old_weight = self.total_samples / max(1, self.total_samples + batch_size)
                new_weight = batch_size / max(1, self.total_samples + batch_size)
                self.depth_loss[i] = old_weight * self.depth_loss[i] + new_weight * loss
        
        for i, correct in enumerate(depth_correct):
            if i < len(self.depth_accuracy):
                acc = correct / max(1, batch_size)
                old_weight = self.total_samples / max(1, self.total_samples + batch_size)
                new_weight = batch_size / max(1, self.total_samples + batch_size)
                self.depth_accuracy[i] = old_weight * self.depth_accuracy[i] + new_weight * acc
        
        self.total_samples += batch_size


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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        
        # Project hidden state
        self.hidden_proj = nn.Linear(d_model, d_embed)
        
        # Context projection for previous token info
        self.context_proj = nn.Linear(d_embed, d_embed)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_embed)
        
        # Prediction head
        self.head = nn.Linear(d_embed, vocab_size)
        
        self._last_hidden: Optional[mx.array] = None
    
    def __call__(
        self,
        hidden_state: mx.array,
        previous_context: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
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
        
        # Normalize
        h = self.layer_norm(h)
        
        # Store for potential use
        self._last_hidden = h
        
        # Predict
        logits = self.head(h)
        
        return logits, h
    
    def get_last_hidden(self) -> Optional[mx.array]:
        """Get the last hidden state for chaining."""
        return self._last_hidden


class SequentialMTPModule(nn.Module):
    """
    Sequential MTP module with lightweight self-attention for refinement.
    """
    
    def __init__(
        self,
        d_model: int,
        d_embed: int,
        vocab_size: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_embed)
        
        # Simple self-attention for context refinement
        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.v_proj = nn.Linear(d_embed, d_embed)
        self.o_proj = nn.Linear(d_embed, d_embed)
        self.num_heads = num_heads
        self.head_dim = d_embed // num_heads
        
        # Feed-forward network
        self.ffn_up = nn.Linear(d_embed, d_embed * 4)
        self.ffn_down = nn.Linear(d_embed * 4, d_embed)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_embed)
        self.ln2 = nn.LayerNorm(d_embed)
        
        # Output head
        self.head = nn.Linear(d_embed, vocab_size)
    
    def __call__(
        self,
        hidden_state: mx.array,
        causal_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass with attention refinement."""
        batch_size, seq_len, _ = hidden_state.shape
        
        # Project input
        h = self.input_proj(hidden_state)
        
        # Self-attention with residual
        h_norm = self.ln1(h)
        
        # Compute Q, K, V
        q = self.q_proj(h_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        # Causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        scores = scores + mask
        
        # Softmax and output
        attn_weights = mx.softmax(scores, axis=-1)
        attn_out = attn_weights @ v
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_embed)
        attn_out = self.o_proj(attn_out)
        
        h = h + attn_out
        
        # FFN with residual
        h_norm = self.ln2(h)
        ffn_out = self.ffn_down(nn.gelu(self.ffn_up(h_norm)))
        h = h + ffn_out
        
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
        
        # Simplified transformer layers
        self.layers = [nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers)]
        
        # Layer norms
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Main prediction head (t+1)
        self.main_head = nn.Linear(config.d_model, config.vocab_size)
        
        # MTP modules for future predictions
        if config.use_sequential_refinement:
            self.mtp_modules = [
                SequentialMTPModule(
                    d_model=config.d_model,
                    d_embed=config.d_embed,
                    vocab_size=config.vocab_size,
                    num_heads=max(1, config.num_heads // 4),
                    dropout=0.1,
                )
                for _ in range(config.prediction_depth)
            ]
        else:
            self.mtp_modules = [
                MTPModule(
                    d_model=config.d_model,
                    d_embed=config.d_embed,
                    vocab_size=config.vocab_size,
                    dropout=0.1,
                )
                for _ in range(config.prediction_depth)
            ]
        
        # Loss weights
        self._loss_weights = config.get_loss_weights()
        
        # Metrics
        self._metrics = MTPMetrics()
        self._metrics.reset(config.prediction_depth)
        
        # Training mode
        self._is_training = True
    
    def get_loss_weights(self) -> List[float]:
        """Get loss weights for MTP heads."""
        return self._loss_weights
    
    def get_metrics(self) -> MTPMetrics:
        """Get metrics tracker."""
        return self._metrics
    
    def set_training(self, mode: bool = True):
        """Set training mode."""
        self._is_training = mode
    
    def __call__(
        self,
        input_ids: mx.array,
        return_hidden: bool = False,
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Forward pass with multi-token prediction.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            return_hidden: Whether to return final hidden state
            
        Returns:
            - main_logits: Main prediction (t+1) [batch, seq, vocab]
            - future_logits: List of future predictions [batch, seq, vocab]
        """
        # Embed tokens
        x = self.embed(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Main prediction (t+1)
        hidden = self.ln_f(x)
        main_logits = self.main_head(hidden)
        
        # MTP Forward (Sequential)
        future_logits = []
        current_hidden = x
        
        for module in self.mtp_modules:
            logits, new_hidden = module(current_hidden)
            future_logits.append(logits)
            current_hidden = new_hidden
        
        return main_logits, future_logits
    
    def forward_speculative(
        self,
        input_ids: mx.array,
    ) -> Tuple[mx.array, List[Tuple[mx.array, float]]]:
        """
        Forward pass for speculative decoding inference.
        
        Returns main prediction and speculative candidates with confidence.
        """
        main_logits, mtp_logits = self(input_ids)
        
        speculative_candidates = []
        for logits in mtp_logits:
            # Get probabilities
            probs = mx.softmax(logits, axis=-1)
            
            # Get max probability as confidence
            confidence = float(mx.max(probs, axis=-1).mean().item())
            
            speculative_candidates.append((logits, confidence))
        
        return main_logits, speculative_candidates

    def compute_loss(
        self,
        input_ids: mx.array,
        targets: mx.array,
    ) -> Tuple[mx.array, Dict[str, float]]:
        """
        Compute MTP loss with weighted contributions from each depth.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            targets: Target token IDs [batch, seq_len]
            
        Returns:
            - total_loss: Weighted sum of all MTP losses
            - loss_dict: Dictionary with individual losses
        """
        main_logits, future_logits = self(input_ids)
        
        batch_size, seq_len, vocab_size = main_logits.shape
        
        # Main loss (t+1)
        main_targets = targets[:, 1:]  # Shift targets by 1
        main_logits_shift = main_logits[:, :-1]  # Shift logits
        
        # Cross-entropy loss
        main_loss = mx.mean(
            nn.losses.cross_entropy(
                main_logits_shift.reshape(-1, vocab_size),
                main_targets.reshape(-1),
            )
        )
        
        total_loss = self._loss_weights[0] * main_loss
        loss_dict = {"main_loss": float(main_loss.item())}
        
        # MTP losses for future predictions
        for i, logits in enumerate(future_logits):
            shift = i + 2  # t+2, t+3, etc.
            if shift >= seq_len:
                break
                
            # Shift targets for future prediction
            mtp_targets = targets[:, shift:]
            mtp_logits = logits[:, :-shift]
            
            if mtp_logits.shape[1] == 0:
                continue
            
            mtp_loss = mx.mean(
                nn.losses.cross_entropy(
                    mtp_logits.reshape(-1, vocab_size),
                    mtp_targets.reshape(-1),
                )
            )
            
            weight = self._loss_weights[i + 1] if i + 1 < len(self._loss_weights) else 0.1
            total_loss = total_loss + weight * mtp_loss
            loss_dict[f"mtp_{i}_loss"] = float(mtp_loss.item())
            
            # Update metrics
            self._metrics.record_loss(i, float(mtp_loss.item()))
        
        loss_dict["total_loss"] = float(total_loss.item())
        return total_loss, loss_dict


# Create an alias for V3 compatibility
MTPModelV3 = MTPModel


# Legacy compatibility
class MTPModelLegacy(nn.Module):
    """Legacy MTP model for backward compatibility."""
    
    def __init__(self, vocab_size, d_model, num_layers, k_predictions):
        super().__init__()
        self.k_predictions = k_predictions
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        
        self.head = nn.Linear(d_model, vocab_size)
        
        self.mtp_modules = [
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, vocab_size)
            ) for _ in range(k_predictions)
        ]
        
    def __call__(self, x):
        h = self.embedding(x)
        
        for layer in self.layers:
            h = layer(h)
            
        main_logits = self.head(h)
        
        future_logits = []
        current_h = h
        
        for module in self.mtp_modules:
            combined = mx.concatenate([current_h, h], axis=-1)
            logits = module(combined)
            future_logits.append(logits)
            current_h = h
            
        return main_logits, future_logits


def demo_mtp():
    """Demonstrate MTP model."""
    print("=" * 60)
    print("Multi-Token Prediction (MTP) Demo")
    print("=" * 60)
    
    # Create config
    config = MTPConfig.small()
    print(f"\nConfig: vocab={config.vocab_size}, d_model={config.d_model}, "
          f"depth={config.prediction_depth}")
    
    # Create model
    model = MTPModel(config)
    
    # Test input
    batch_size = 2
    seq_len = 16
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    main_logits, future_logits = model(input_ids)
    
    print(f"Main logits shape: {main_logits.shape}")
    print(f"Future predictions: {len(future_logits)}")
    for i, logits in enumerate(future_logits):
        print(f"  MTP[{i}] (t+{i+2}) shape: {logits.shape}")
    
    # Loss weights
    weights = model.get_loss_weights()
    print(f"\nLoss weights: {weights}")
    
    # Speculative decoding demo
    print("\nSpeculative decoding mode:")
    main_logits, candidates = model.forward_speculative(input_ids)
    for i, (logits, conf) in enumerate(candidates):
        print(f"  MTP[{i}] confidence: {conf:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_mtp()

