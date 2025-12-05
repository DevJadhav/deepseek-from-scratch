"""
ANE-Optimized Transformer Layer

This module implements a single transformer layer optimized for Apple Neural Engine,
combining ANE-optimized attention (MLA) with ANE-optimized MoE feed-forward.

Key Features:
- Pre-norm architecture with ANERMSNorm
- Multi-Latent Attention with chunked processing
- Mixture of Experts with automatic strategy selection
- FP16 computation with INT8 weights
- 128-token chunk size for ANE constraints
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum


@dataclass
class ANEDeepSeekLayerConfig:
    """Configuration for a single ANE-optimized transformer layer."""
    
    # Model dimensions
    d_model: int = 4096
    num_heads: int = 32
    
    # Attention config
    d_latent: int = 512  # Latent dimension for KV compression
    d_rope: int = 64  # RoPE dimension
    chunk_size: int = 128  # ANE-friendly chunk size
    
    # MoE config
    d_hidden: int = 11008  # Hidden dimension in expert FFN
    num_experts: int = 32  # Number of routed experts
    num_shared: int = 2  # Number of shared (always-active) experts
    top_k: int = 4  # Number of experts per token
    use_moe: bool = True  # Whether to use MoE (False = dense FFN)
    
    # Precision config
    use_fp16: bool = True  # FP16 activations
    use_int8_weights: bool = True  # INT8 quantized weights
    
    # Dropout
    dropout: float = 0.0
    
    @classmethod
    def tiny(cls) -> "ANEDeepSeekLayerConfig":
        """Tiny configuration for testing."""
        return cls(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
            d_hidden=512,
            num_experts=8,
            num_shared=1,
            top_k=2,
            use_moe=True,
            use_fp16=True,
            use_int8_weights=False,
        )
    
    @classmethod
    def small(cls) -> "ANEDeepSeekLayerConfig":
        """Small configuration for development."""
        return cls(
            d_model=1024,
            num_heads=16,
            d_latent=256,
            d_rope=64,
            d_hidden=4096,
            num_experts=16,
            num_shared=2,
            top_k=4,
            use_moe=True,
            use_fp16=True,
            use_int8_weights=True,
        )
    
    @classmethod
    def deepseek_v3(cls) -> "ANEDeepSeekLayerConfig":
        """DeepSeek-V3 configuration (scaled for ANE)."""
        return cls(
            d_model=4096,
            num_heads=32,
            d_latent=512,
            d_rope=64,
            d_hidden=11008,
            num_experts=32,  # Reduced from 256 for ANE
            num_shared=2,
            top_k=4,
            use_moe=True,
            use_fp16=True,
            use_int8_weights=True,
        )


class ANEDeepSeekLayer(nn.Module):
    """
    Single ANE-optimized transformer layer.
    
    Architecture (Pre-norm):
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MoE/FFN -> +
        |__________________________|   |_______________________|
        
    Features:
    - Multi-Latent Attention with 14x KV compression
    - Chunked attention processing (128 tokens)
    - MoE with automatic strategy selection
    - FP16 computation pipeline
    """
    
    def __init__(
        self,
        config: ANEDeepSeekLayerConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Import ANE components
        from ..layers.base import ANERMSNorm
        from ..attention.mla import ANEMultiLatentAttention, ANEMLAConfig
        from ..moe.moe import ANEMoE, ANEMoEConfig
        from ..moe.expert import ANEExpertConfig
        
        # Pre-norm layers
        self.attn_norm = ANERMSNorm(config.d_model, use_fp16=config.use_fp16)
        self.ffn_norm = ANERMSNorm(config.d_model, use_fp16=config.use_fp16)
        
        # Multi-Latent Attention
        mla_config = ANEMLAConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_latent=config.d_latent,
            d_rope=config.d_rope,
            chunk_size=config.chunk_size,
            dropout=config.dropout,
        )
        self.attention = ANEMultiLatentAttention(mla_config)
        
        # MoE or Dense FFN
        if config.use_moe:
            # Create MoE config with routed/shared hidden dimension calculations
            routed_hidden_mult = config.d_hidden / config.d_model
            moe_config = ANEMoEConfig(
                d_model=config.d_model,
                num_routed_experts=config.num_experts,
                num_shared_experts=config.num_shared,
                top_k=config.top_k,
                routed_hidden_mult=routed_hidden_mult,
                shared_hidden_mult=routed_hidden_mult * 2,  # Shared experts are larger
                use_fp16=config.use_fp16,
                use_quantized_weights=config.use_int8_weights,
                dropout=config.dropout,
            )
            self.ffn = ANEMoE(moe_config)
        else:
            # Dense FFN with SwiGLU activation
            self.ffn = self._create_dense_ffn(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
    def _create_dense_ffn(self, config: ANEDeepSeekLayerConfig) -> nn.Module:
        """Create a dense FFN (no MoE) with SwiGLU activation."""
        from ..layers.base import ANELinear
        from ..utils.quantization import QuantizationType
        from ..layers.activations import ANESiLU
        
        # Determine quantization type
        quant_type = (
            QuantizationType.INT8_PER_CHANNEL 
            if config.use_int8_weights 
            else QuantizationType.NONE
        )
        
        class DenseFFN(nn.Module):
            def __init__(self, d_model: int, d_hidden: int, use_fp16: bool, quantization: QuantizationType):
                super().__init__()
                # SwiGLU: gate * silu(x) where gate and x are projected from input
                self.gate_proj = ANELinear(
                    d_model, d_hidden,
                    use_fp16=use_fp16,
                    quant_type=quantization,
                )
                self.up_proj = ANELinear(
                    d_model, d_hidden,
                    use_fp16=use_fp16,
                    quant_type=quantization,
                )
                self.down_proj = ANELinear(
                    d_hidden, d_model,
                    use_fp16=use_fp16,
                    quant_type=quantization,
                )
                self.activation = ANESiLU(use_fp16=use_fp16)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate = self.activation(self.gate_proj(x))
                up = self.up_proj(x)
                return self.down_proj(gate * up)
        
        return DenseFFN(
            config.d_model,
            config.d_hidden,
            config.use_fp16,
            quant_type,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len] or [batch, 1, seq_len, seq_len]
            use_cache: Whether to use/return KV cache
            position_offset: Position offset for RoPE (used with KV cache)
            
        Returns:
            Tuple of:
            - Output tensor [batch, seq_len, d_model]
            - Cached KV latent (if use_cache=True)
        """
        # Cast to FP16 if needed
        if self.config.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        # Attention block with pre-norm and residual
        residual = x
        x = self.attn_norm(x)
        attn_out, cached_latent = self.attention(
            x,
            mask=attention_mask,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        x = residual + self.dropout(attn_out)
        
        # FFN/MoE block with pre-norm and residual
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        # MoE returns (output, aux_loss) tuple, dense FFN returns tensor
        if isinstance(ffn_out, tuple):
            ffn_out = ffn_out[0]  # Extract output, discard aux_loss during forward
        x = residual + self.dropout(ffn_out)
        
        return x, cached_latent
    
    def get_memory_footprint(self) -> dict:
        """Estimate memory footprint of this layer."""
        config = self.config
        
        # Attention params
        # Q/K/V projections + output projection
        attn_params = (
            config.d_model * (config.d_latent + config.d_rope) +  # KV down
            config.d_latent * (config.num_heads * config.d_latent // config.num_heads) +  # K up
            config.d_latent * (config.num_heads * config.d_latent // config.num_heads) +  # V up
            config.d_model * config.d_model  # Output
        )
        
        # MoE params
        if config.use_moe:
            expert_params = 3 * config.d_model * config.d_hidden  # gate + up + down
            total_expert_params = expert_params * (config.num_experts + config.num_shared)
            router_params = config.d_model * config.num_experts
            ffn_params = total_expert_params + router_params
        else:
            ffn_params = 3 * config.d_model * config.d_hidden
        
        # Norm params
        norm_params = 2 * config.d_model
        
        total = attn_params + ffn_params + norm_params
        
        # Calculate size in bytes
        weight_bytes = 1 if config.use_int8_weights else 2
        activation_bytes = 2 if config.use_fp16 else 4
        
        return {
            "attention_params": attn_params,
            "ffn_params": ffn_params,
            "norm_params": norm_params,
            "total_params": total,
            "weight_size_mb": total * weight_bytes / (1024 * 1024),
            "activation_size_mb": None,  # Depends on sequence length
        }


class ANEDeepSeekLayerWithCache(ANEDeepSeekLayer):
    """
    ANE transformer layer with built-in KV cache management.
    
    This variant automatically manages the KV cache for generation,
    making it easier to use for autoregressive decoding.
    """
    
    def __init__(
        self,
        config: ANEDeepSeekLayerConfig,
        layer_idx: int = 0,
        max_seq_len: int = 8192,
    ):
        super().__init__(config, layer_idx)
        self.max_seq_len = max_seq_len
        
        # Initialize KV cache
        from ..attention.kv_cache import ANEKVCache
        self.kv_cache = ANEKVCache(
            max_seq_len=max_seq_len,
            num_heads=config.num_heads,
            head_dim=config.d_model // config.num_heads,
            use_fp16=config.use_fp16,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with automatic cache management.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            use_cache: Whether to use/update the KV cache
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if use_cache:
            kv = (self.kv_cache.key_cache, self.kv_cache.value_cache)
        else:
            kv = None
            
        output, new_kv = super().forward(
            x,
            kv_cache=kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        if use_cache and new_kv is not None:
            self.kv_cache.key_cache, self.kv_cache.value_cache = new_kv
            
        return output
    
    def reset_cache(self):
        """Reset the KV cache for a new generation."""
        self.kv_cache.reset()
