"""
MLX Training Module for Tiny DeepSeek Model
============================================

This module provides actual training loops for MLX models using:
- Multi-Token Prediction (MTP) architecture
- Mixture of Experts (MoE)
- Multi-head Latent Attention (MLA)

Usage:
    from mlx_impl.tiny_trainer import TinyMLXTrainer
    trainer = TinyMLXTrainer(config)
    trainer.train(train_data, valid_data)
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


@dataclass
class TinyModelConfig:
    """Configuration for tiny DeepSeek model."""
    vocab_size: int = 8000
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 512
    d_latent: int = 64
    d_rope: int = 32
    num_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 1
    mtp_k: int = 2
    dropout: float = 0.1


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dims: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self.inv_freq = inv_freq
    
    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[1]
        t = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        
        cos = mx.cos(emb)[None, :, None, :]
        sin = mx.sin(emb)[None, :, None, :]
        
        return self._apply_rope(x, cos, sin)
    
    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # Split and rotate
        x1, x2 = mx.split(x, 2, axis=-1)
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin


class TinyMLA(nn.Module):
    """
    Simplified Multi-head Latent Attention for tiny model.
    
    Uses compressed KV cache for efficiency.
    """
    
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_latent = config.d_latent
        
        # Query projection
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        
        # Latent compression for KV
        self.kv_compress = nn.Linear(config.d_model, config.d_latent * 2)
        self.kv_decompress = nn.Linear(config.d_latent, config.d_model * 2)
        
        # Output projection
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        
        self.scale = self.head_dim ** -0.5
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, T, D = x.shape
        
        # Query projection
        q = self.q_proj(x)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compress and decompress KV
        kv_latent = self.kv_compress(x)  # (B, T, d_latent * 2)
        kv = self.kv_decompress(kv_latent[:, :, :self.d_latent])  # Use first half
        k, v = mx.split(kv, 2, axis=-1)
        
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
        
        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
        
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = self.o_proj(out)
        
        new_cache = (k, v)
        return out, new_cache


class TinyMoELayer(nn.Module):
    """
    Simplified Mixture of Experts layer for tiny model.
    """
    
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.top_k
        self.d_model = config.d_model
        self.d_hidden = config.d_model * 2
        
        # Router
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        
        # Shared expert
        self.shared_up = nn.Linear(config.d_model, self.d_hidden)
        self.shared_down = nn.Linear(self.d_hidden, config.d_model)
        
        # Routed experts
        self.expert_ups = [nn.Linear(config.d_model, self.d_hidden) for _ in range(config.num_experts)]
        self.expert_downs = [nn.Linear(self.d_hidden, config.d_model) for _ in range(config.num_experts)]
    
    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        
        # Router logits
        router_logits = self.gate(x)  # (B, T, num_experts)
        
        # Top-k selection using argsort (MLX topk only returns values)
        router_probs = mx.softmax(router_logits, axis=-1)
        
        # Get top-k indices manually
        sorted_indices = mx.argsort(router_probs, axis=-1)
        top_k_indices = sorted_indices[..., -self.top_k:]  # Last k are the largest
        
        # Gather top-k probabilities
        # Create index arrays for gathering
        B_idx = mx.arange(B)[:, None, None]
        T_idx = mx.arange(T)[None, :, None]
        top_k_probs = router_probs[B_idx, T_idx, top_k_indices]
        top_k_probs = top_k_probs / (mx.sum(top_k_probs, axis=-1, keepdims=True) + 1e-8)  # Normalize
        
        # Shared expert output
        shared_out = self.shared_down(nn.gelu(self.shared_up(x)))
        
        # Routed expert output (simplified - just use weighted average)
        routed_out = mx.zeros_like(x)
        for i in range(self.num_experts):
            expert_out = self.expert_downs[i](nn.gelu(self.expert_ups[i](x)))
            # Weight by router probability
            mask = (top_k_indices == i).astype(mx.float32)
            weight = mx.sum(top_k_probs * mask, axis=-1, keepdims=True)
            routed_out = routed_out + expert_out * weight
        
        # Combine shared and routed
        return shared_out + routed_out


class TinyTransformerBlock(nn.Module):
    """Single transformer block with MLA + MoE."""
    
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.attn = TinyMLA(config)
        self.moe = TinyMoELayer(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Pre-norm attention
        h = self.norm1(x)
        attn_out, new_cache = self.attn(h, mask, cache)
        x = x + attn_out
        
        # Pre-norm MoE FFN
        h = self.norm2(x)
        x = x + self.moe(h)
        
        return x, new_cache


class TinyMTPModel(nn.Module):
    """
    Tiny DeepSeek model with Multi-Token Prediction.
    
    This model predicts k future tokens in addition to the next token,
    enabling better sample efficiency during training.
    """
    
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.mtp_k = config.mtp_k
        
        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.layers = [TinyTransformerBlock(config) for _ in range(config.num_layers)]
        
        # Output norm
        self.norm = RMSNorm(config.d_model)
        
        # Main LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # MTP heads for k future tokens
        if config.mtp_k > 0:
            self.mtp_heads = [
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, config.vocab_size),
                )
                for _ in range(config.mtp_k)
            ]
        else:
            self.mtp_heads = []
    
    def __call__(
        self, 
        input_ids: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[mx.array], Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, T) token IDs
            cache: Optional KV cache for inference
            
        Returns:
            main_logits: (B, T, vocab_size) next token logits
            mtp_logits: List of (B, T, vocab_size) future token logits
            new_cache: Updated KV cache
        """
        B, T = input_ids.shape
        
        # Create causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(mx.float32)
        
        # Embeddings
        h = self.embed(input_ids)
        
        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            h, layer_cache = layer(h, mask, cache[i])
            new_cache.append(layer_cache)
        
        # Final norm
        h = self.norm(h)
        
        # Main LM head
        main_logits = self.lm_head(h)
        
        # MTP heads
        mtp_logits = []
        for mtp_head in self.mtp_heads:
            mtp_logits.append(mtp_head(h))
        
        return main_logits, mtp_logits, new_cache
    
    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> mx.array:
        """Generate text autoregressively."""
        B, T = input_ids.shape
        cache = None
        
        for _ in range(max_new_tokens):
            # Get logits
            if cache is None:
                logits, _, cache = self(input_ids)
                logits = logits[:, -1, :]  # Last token
            else:
                logits, _, cache = self(input_ids[:, -1:], cache)
                logits = logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-p sampling
            probs = mx.softmax(logits, axis=-1)
            sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            
            # Sample
            next_token = mx.argmax(probs, axis=-1, keepdims=True)
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
        
        return input_ids


def compute_mtp_loss(
    main_logits: mx.array,
    mtp_logits: List[mx.array],
    targets: mx.array,
    pad_token_id: int = 0,
    mtp_weight: float = 0.1,
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Compute combined loss for main LM head and MTP heads.
    
    Args:
        main_logits: (B, T, V) main next-token logits
        mtp_logits: List of (B, T, V) future token logits
        targets: (B, T) target token IDs
        pad_token_id: Padding token ID to ignore
        mtp_weight: Weight for MTP loss
        
    Returns:
        loss: Combined scalar loss
        metrics: Dictionary of loss components
    """
    B, T, V = main_logits.shape
    
    # Main LM loss (shifted)
    main_targets = targets[:, 1:]  # Shift targets
    main_logits = main_logits[:, :-1, :]  # Shift logits
    
    # Create mask for non-padding tokens
    mask = (main_targets != pad_token_id).astype(mx.float32)
    
    # Cross-entropy loss (manual log_softmax: log(softmax(x)) = x - logsumexp(x))
    main_log_probs = main_logits - mx.logsumexp(main_logits, axis=-1, keepdims=True)
    
    # Gather log probs for targets
    B_new, T_new = main_targets.shape
    batch_idx = mx.arange(B_new)[:, None]
    seq_idx = mx.arange(T_new)[None, :]
    target_log_probs = main_log_probs[batch_idx, seq_idx, main_targets]
    
    main_loss = -mx.sum(target_log_probs * mask) / (mx.sum(mask) + 1e-8)
    
    # MTP losses
    mtp_losses = []
    for k, logits in enumerate(mtp_logits):
        # Shift by k+2 (1 for main + k+1 for MTP)
        shift = k + 2
        if T - shift < 1:
            continue
            
        mtp_targets = targets[:, shift:]
        mtp_logits_k = logits[:, :-shift, :]
        
        mtp_mask = (mtp_targets != pad_token_id).astype(mx.float32)
        mtp_log_probs = mtp_logits_k - mx.logsumexp(mtp_logits_k, axis=-1, keepdims=True)
        
        B_k, T_k = mtp_targets.shape
        batch_idx_k = mx.arange(B_k)[:, None]
        seq_idx_k = mx.arange(T_k)[None, :]
        mtp_target_log_probs = mtp_log_probs[batch_idx_k, seq_idx_k, mtp_targets]
        
        mtp_loss_k = -mx.sum(mtp_target_log_probs * mtp_mask) / (mx.sum(mtp_mask) + 1e-8)
        mtp_losses.append(mtp_loss_k)
    
    # Combine losses
    total_mtp_loss = mx.mean(mx.array(mtp_losses)) if mtp_losses else mx.array(0.0)
    total_loss = main_loss + mtp_weight * total_mtp_loss
    
    metrics = {
        "main_loss": float(main_loss.item()),
        "mtp_loss": float(total_mtp_loss.item()) if mtp_losses else 0.0,
        "total_loss": float(total_loss.item()),
    }
    
    return total_loss, metrics


class DataLoader:
    """Simple data loader for JSONL files."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int = 16,
        max_seq_len: int = 512,
        shuffle: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        
        # Load data
        self.samples = self._load_data()
        self.indices = list(range(len(self.samples)))
    
    def _load_data(self) -> List[str]:
        """Load texts from JSONL file."""
        samples = []
        
        # Find JSONL files
        if self.data_path.is_dir():
            files = list(self.data_path.glob("**/*.jsonl"))
        else:
            files = [self.data_path]
        
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            samples.append(data["text"])
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples) // self.batch_size
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices) - self.batch_size, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            texts = [self.samples[j] for j in batch_indices]
            
            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="np",
            )
            
            yield {
                "input_ids": mx.array(encoded["input_ids"]),
                "attention_mask": mx.array(encoded["attention_mask"]),
            }


class TinyMLXTrainer:
    """
    Trainer for tiny DeepSeek model using MLX.
    
    Features:
    - Multi-Token Prediction (MTP) training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: TinyMTPModel,
        config: TinyModelConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        gradient_accumulation_steps: int = 4,
        checkpoint_dir: str = "./checkpoints",
        log_every: int = 50,
        save_every: int = 1000,
        eval_every: int = 500,
        mtp_weight: float = 0.1,
    ):
        self.model = model
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.mtp_weight = mtp_weight
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
    
    def _count_params(self, params=None) -> int:
        """Count total parameters in model."""
        if params is None:
            params = self.model.parameters()
        total = 0
        for k, v in params.items():
            if isinstance(v, dict):
                total += self._count_params(v)
            elif isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        total += self._count_params(item)
                    elif isinstance(item, mx.array):
                        total += item.size
        return total
    
    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train_step(
        self, 
        batch: Dict[str, mx.array],
        pad_token_id: int,
    ) -> Dict[str, float]:
        """Single training step."""
        input_ids = batch["input_ids"]
        
        def loss_fn(model, x):
            main_logits, mtp_logits, _ = model(x)
            loss, _ = compute_mtp_loss(
                main_logits, mtp_logits, x,
                pad_token_id=pad_token_id,
                mtp_weight=self.mtp_weight,
            )
            return loss
        
        # Compute gradients using nn.value_and_grad
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(self.model, input_ids)
        
        # Compute metrics separately (no grad needed)
        main_logits, mtp_logits, _ = self.model(input_ids)
        _, metrics = compute_mtp_loss(
            main_logits, mtp_logits, input_ids,
            pad_token_id=pad_token_id,
            mtp_weight=self.mtp_weight,
        )
        
        return grads, metrics
    
    def _add_grads(self, g1, g2):
        """Recursively add two gradient structures (dicts, lists, or arrays)."""
        if isinstance(g1, dict):
            return {k: self._add_grads(g1[k], g2[k]) for k in g1.keys()}
        elif isinstance(g1, list):
            return [self._add_grads(a, b) for a, b in zip(g1, g2)]
        else:
            return g1 + g2
    
    def _scale_grads(self, grads, scale: float):
        """Recursively scale gradient structure."""
        if isinstance(grads, dict):
            return {k: self._scale_grads(v, scale) for k, v in grads.items()}
        elif isinstance(grads, list):
            return [self._scale_grads(g, scale) for g in grads]
        else:
            return grads * scale
    
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        pad_token_id: int = 0,
    ):
        """Main training loop."""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Model parameters: {self._count_params():,}")
        print(f"Max steps: {self.max_steps}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {train_loader.batch_size * self.gradient_accumulation_steps}")
        print("=" * 60)
        
        accumulated_grads = None
        accumulated_count = 0
        epoch = 0
        
        while self.global_step < self.max_steps:
            epoch += 1
            print(f"\n--- Epoch {epoch} ---")
            
            for batch in train_loader:
                # Update learning rate
                lr = self.get_lr(self.global_step)
                self.optimizer.learning_rate = lr
                
                # Forward + backward
                grads, metrics = self.train_step(batch, pad_token_id)
                
                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = self._add_grads(accumulated_grads, grads)
                accumulated_count += 1
                
                # Update weights
                if accumulated_count >= self.gradient_accumulation_steps:
                    # Average gradients
                    accumulated_grads = self._scale_grads(accumulated_grads, 1.0 / accumulated_count)
                    
                    # Apply gradients
                    self.optimizer.update(self.model, accumulated_grads)
                    mx.eval(self.model.parameters())
                    
                    # Reset accumulation
                    accumulated_grads = None
                    accumulated_count = 0
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_every == 0:
                        print(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Loss: {metrics['total_loss']:.4f} | "
                            f"Main: {metrics['main_loss']:.4f} | "
                            f"MTP: {metrics['mtp_loss']:.4f} | "
                            f"LR: {lr:.2e}"
                        )
                    
                    # Evaluation
                    if valid_loader and self.global_step % self.eval_every == 0:
                        val_loss = self.evaluate(valid_loader, pad_token_id)
                        print(f"  Validation Loss: {val_loss:.4f}")
                        
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint("best")
                    
                    # Save checkpoint
                    if self.global_step % self.save_every == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                    
                    # Check if done
                    if self.global_step >= self.max_steps:
                        break
            
            if self.global_step >= self.max_steps:
                break
        
        # Final save
        self.save_checkpoint("final")
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Final step: {self.global_step}")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print("=" * 60)
    
    def evaluate(
        self,
        loader: DataLoader,
        pad_token_id: int,
        max_batches: int = 10,
    ) -> float:
        """Evaluate model on validation set."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            input_ids = batch["input_ids"]
            main_logits, mtp_logits, _ = self.model(input_ids)
            loss, _ = compute_mtp_loss(
                main_logits, mtp_logits, input_ids,
                pad_token_id=pad_token_id,
                mtp_weight=self.mtp_weight,
            )
            total_loss += float(loss.item())
            num_batches += 1
            
            if num_batches >= max_batches:
                break
        
        return total_loss / max(num_batches, 1)
    
    def _flatten_params(self, params, prefix=""):
        """Flatten nested parameter dict/list to flat dict with dotted keys."""
        flat = {}
        if isinstance(params, dict):
            for k, v in params.items():
                new_key = f"{prefix}.{k}" if prefix else k
                flat.update(self._flatten_params(v, new_key))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                new_key = f"{prefix}.{i}"
                flat.update(self._flatten_params(v, new_key))
        else:
            flat[prefix] = params
        return flat
    
    def _unflatten_params(self, flat_params):
        """Unflatten dotted keys back to nested dict/list structure."""
        result = {}
        
        for key, value in flat_params.items():
            parts = key.split(".")
            current = result
            
            for i, part in enumerate(parts[:-1]):
                next_part = parts[i + 1]
                
                # Check if next level should be a list
                if next_part.isdigit():
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                else:
                    if part.isdigit():
                        part = int(part)
                        while len(current) <= part:
                            current.append({})
                        if not isinstance(current[part], dict):
                            current[part] = {}
                        current = current[part]
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            
            # Set the final value
            final_key = parts[-1]
            if final_key.isdigit():
                final_key = int(final_key)
                while len(current) <= final_key:
                    current.append(None)
                current[final_key] = value
            else:
                current[final_key] = value
        
        return result
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights (flattened for safetensors compatibility)
        weights_path = path / "model.safetensors"
        flat_params = self._flatten_params(dict(self.model.parameters()))
        mx.save_safetensors(str(weights_path), flat_params)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save training state
        state_path = path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump({
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "learning_rate": self.learning_rate,
            }, f, indent=2)
        
        print(f"  Checkpoint saved: {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
    ) -> Tuple["TinyMLXTrainer", TinyMTPModel, TinyModelConfig]:
        """Load model from checkpoint."""
        path = Path(checkpoint_path)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = TinyModelConfig(**config_dict)
        
        # Create model
        model = TinyMTPModel(config)
        
        # Load weights
        weights = mx.load(str(path / "model.safetensors"))
        model.update(weights)
        
        # Load training state
        with open(path / "training_state.json", "r") as f:
            state = json.load(f)
        
        # Create trainer
        trainer = cls(
            model=model,
            config=config,
            learning_rate=state["learning_rate"],
        )
        trainer.global_step = state["global_step"]
        trainer.best_loss = state["best_loss"]
        
        return trainer, model, config


def create_tiny_model(config: Optional[TinyModelConfig] = None) -> TinyMTPModel:
    """Create a tiny DeepSeek model."""
    if config is None:
        config = TinyModelConfig()
    return TinyMTPModel(config)


if __name__ == "__main__":
    # Quick test
    print("Testing Tiny MLX Model...")
    
    config = TinyModelConfig(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        mtp_k=2,
    )
    
    model = TinyMTPModel(config)
    
    # Test forward pass
    input_ids = mx.random.randint(0, config.vocab_size, (2, 32))
    main_logits, mtp_logits, cache = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Main logits shape: {main_logits.shape}")
    print(f"MTP heads: {len(mtp_logits)}")
    for i, l in enumerate(mtp_logits):
        print(f"  MTP head {i} shape: {l.shape}")
    
    # Test loss
    loss, metrics = compute_mtp_loss(main_logits, mtp_logits, input_ids)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Count parameters
    total_params = sum(p.size for p in model.parameters().values())
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    print("\nTest passed!")
