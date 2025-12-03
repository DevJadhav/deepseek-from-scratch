import torch
import torch.nn as nn
from typing import Optional
from deepseek.model.mla import DeepSeekAttention
from deepseek.model.moe import DeepSeekMoE, StandardMoE

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class DeepSeekLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, d_rope, 
                 d_hidden, num_experts, num_shared, num_routed, top_k,
                 use_moe=True):
        super().__init__()
        
        # Attention
        self.attn = DeepSeekAttention(d_model, num_heads, d_latent, d_rope)
        self.attn_norm = RMSNorm(d_model)
        
        # Feed Forward (MoE or Standard MLP)
        if use_moe:
            self.mlp = DeepSeekMoE(d_model, d_hidden, num_experts, num_shared, num_routed, top_k)
        else:
            # Standard MLP (SwiGLU usually)
            # For simplicity, using StandardMoE with 1 expert or just a Sequential
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_model)
            )
        self.mlp_norm = RMSNorm(d_model)
        
    def forward(self, x, mask=None):
        # Pre-Norm
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask)
        x = residual + x
        
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class DeepSeekModel(nn.Module):
    def __init__(self, vocab_size, num_layers, gradient_checkpointing=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        d_model = kwargs.get('d_model', 512)
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DeepSeekLayer(**kwargs) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Gradient Checkpointing
                # Note: checkpoint requires inputs to have requires_grad=True for at least one input
                # Embedding output usually has it if it's trainable.
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
            
        x = self.norm(x)
        logits = self.head(x)
        return logits
