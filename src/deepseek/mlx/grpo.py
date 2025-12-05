"""
GRPO (Group Relative Policy Optimization) - MLX Implementation
"""

import mlx.core as mx
import mlx.nn as nn


class GRPOTrainer:
    """Group Relative Policy Optimization trainer."""
    
    def __init__(self, beta: float = 0.01):
        self.beta = beta
        
    def compute_loss(
        self,
        logits: mx.array,
        input_ids: mx.array,
        rewards: mx.array,
        ref_logits: mx.array,
    ) -> mx.array:
        """
        Compute GRPO loss.
        
        Args:
            logits: (G, Seq, Vocab) policy logits
            input_ids: (G, Seq) token ids
            rewards: (G,) scalar rewards
            ref_logits: (G, Seq, Vocab) reference model logits
            
        Returns:
            Scalar loss
        """
        G, Seq, Vocab = logits.shape
        
        # 1. Compute advantages (normalized rewards)
        mean_r = mx.mean(rewards)
        std_r = mx.std(rewards) + 1e-8
        advantages = (rewards - mean_r) / std_r  # (G,)
        
        # 2. Compute policy log probs
        log_probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(log_probs + 1e-10)
        
        # Gather log probs for actual tokens
        # Using advanced indexing
        batch_idx = mx.arange(G)[:, None]  # (G, 1)
        seq_idx = mx.arange(Seq)[None, :]  # (1, Seq)
        token_log_probs = log_probs[batch_idx, seq_idx, input_ids]  # (G, Seq)
        seq_log_probs = mx.sum(token_log_probs, axis=1)  # (G,)
        
        # 3. Compute KL divergence
        ref_log_probs = mx.softmax(ref_logits, axis=-1)
        ref_log_probs = mx.log(ref_log_probs + 1e-10)
        
        probs = mx.softmax(logits, axis=-1)
        kl = mx.sum(probs * (log_probs - ref_log_probs), axis=-1)  # (G, Seq)
        mean_kl = mx.mean(kl, axis=1)  # (G,)
        
        # 4. Compute loss
        loss = -(advantages * seq_log_probs) + self.beta * mean_kl
        return mx.mean(loss)


class GroupSampler:
    """Group sampler for GRPO."""
    
    def __init__(self, group_size: int):
        self.group_size = group_size
        
    def sample(self, prompt: str) -> list:
        return [f"Sampled output {i} for prompt..." for i in range(self.group_size)]
