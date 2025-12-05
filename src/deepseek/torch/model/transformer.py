"""
DeepSeek Transformer Model

This module implements the DeepSeek transformer architecture with:
- Multi-Latent Attention (MLA) with Flash Attention support
- Mixture of Experts (MoE) layers
- Gradient checkpointing for memory efficiency
- RMSNorm for normalization
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional
from dataclasses import dataclass
from deepseek.torch.model.mla import DeepSeekAttention, FlashAttentionConfig
from deepseek.torch.model.moe import DeepSeekMoE, StandardMoE


@dataclass
class GradientCheckpointConfig:
    """Configuration for gradient checkpointing."""
    enabled: bool = True
    # Checkpoint every N layers (1 = every layer, 2 = every other layer)
    checkpoint_every_n_layers: int = 1
    # Checkpoint MoE expert forward passes
    checkpoint_moe: bool = True
    # Use reentrant checkpointing (False recommended for torch.compile)
    use_reentrant: bool = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class DeepSeekLayer(nn.Module):
    """
    Single transformer layer with MLA attention and optional MoE.
    
    Supports gradient checkpointing for memory-efficient training.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        d_rope: int,
        d_hidden: int,
        num_experts: int,
        num_shared: int,
        num_routed: int,
        top_k: int,
        use_moe: bool = True,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
        checkpoint_config: Optional[GradientCheckpointConfig] = None,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.checkpoint_config = checkpoint_config or GradientCheckpointConfig(enabled=False)
        
        # Attention with Flash Attention support
        self.attn = DeepSeekAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=d_latent,
            d_rope=d_rope,
            dropout=dropout,
            attention_config=attention_config,
        )
        self.attn_norm = RMSNorm(d_model)
        
        # Feed Forward (MoE or Standard MLP)
        if use_moe:
            self.mlp = DeepSeekMoE(d_model, d_hidden, num_experts, num_shared, num_routed, top_k)
        else:
            # Standard SwiGLU-style MLP
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_model)
            )
        self.mlp_norm = RMSNorm(d_model)
    
    def _attention_forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention sublayer with pre-norm."""
        return self.attn(self.attn_norm(x), mask)
    
    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP sublayer with pre-norm."""
        return self.mlp(self.mlp_norm(x))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention block with residual
        if self.checkpoint_config.enabled and self.training:
            attn_out = checkpoint(
                self._attention_forward,
                x,
                mask,
                use_reentrant=self.checkpoint_config.use_reentrant,
            )
        else:
            attn_out = self._attention_forward(x, mask)
        x = x + attn_out
        
        # MLP/MoE block with residual
        if self.checkpoint_config.enabled and self.checkpoint_config.checkpoint_moe and self.training:
            mlp_out = checkpoint(
                self._mlp_forward,
                x,
                use_reentrant=self.checkpoint_config.use_reentrant,
            )
        else:
            mlp_out = self._mlp_forward(x)
        x = x + mlp_out
        
        return x


class DeepSeekModel(nn.Module):
    """
    Complete DeepSeek model with configurable attention and checkpointing.
    
    Features:
    - Flash Attention integration
    - Selective gradient checkpointing
    - torch.compile compatibility
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_latent: int = 128,
        d_rope: int = 32,
        d_hidden: int = 2048,
        num_experts: int = 8,
        num_shared: int = 1,
        num_routed: int = 8,
        top_k: int = 2,
        use_moe: bool = True,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
        checkpoint_config: Optional[GradientCheckpointConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Store configs
        self.attention_config = attention_config
        self.checkpoint_config = checkpoint_config or GradientCheckpointConfig()
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Determine if this layer should be checkpointed
            layer_checkpoint_config = None
            if self.checkpoint_config.enabled:
                should_checkpoint = (layer_idx % self.checkpoint_config.checkpoint_every_n_layers == 0)
                layer_checkpoint_config = GradientCheckpointConfig(
                    enabled=should_checkpoint,
                    checkpoint_moe=self.checkpoint_config.checkpoint_moe,
                    use_reentrant=self.checkpoint_config.use_reentrant,
                )
            
            layer = DeepSeekLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_latent=d_latent,
                d_rope=d_rope,
                d_hidden=d_hidden,
                num_experts=num_experts,
                num_shared=num_shared,
                num_routed=num_routed,
                top_k=top_k,
                use_moe=use_moe,
                dropout=dropout,
                attention_config=attention_config,
                checkpoint_config=layer_checkpoint_config,
            )
            self.layers.append(layer)
        
        # Output
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (embed and head share weights)
        self.head.weight = self.embed.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        logits = self.head(x)
        return logits
    
    def configure_gradient_checkpointing(
        self,
        enabled: bool = True,
        checkpoint_every_n_layers: int = 1,
        checkpoint_moe: bool = True,
    ) -> None:
        """
        Configure gradient checkpointing at runtime.
        
        Args:
            enabled: Whether to enable checkpointing
            checkpoint_every_n_layers: Checkpoint every N layers
            checkpoint_moe: Whether to checkpoint MoE layers
        """
        for layer_idx, layer in enumerate(self.layers):
            should_checkpoint = enabled and (layer_idx % checkpoint_every_n_layers == 0)
            layer.checkpoint_config = GradientCheckpointConfig(
                enabled=should_checkpoint,
                checkpoint_moe=checkpoint_moe,
                use_reentrant=False,  # Always use non-reentrant for torch.compile
            )
