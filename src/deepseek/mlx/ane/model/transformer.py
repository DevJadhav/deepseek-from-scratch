"""
ANE-Optimized DeepSeek Transformer Model

This module implements the complete DeepSeek transformer model optimized for
Apple Neural Engine (ANE) inference on Apple Silicon.

Key Features:
- Full transformer architecture with embeddings, layers, and output head
- Configurable for different model sizes (tiny, small, standard, V3)
- Unified memory KV cache for zero-copy inference
- INT8 weights with FP16 activations
- Chunked attention for long context
- MoE with automatic strategy selection

Usage:
    from ane_impl.model import ANEDeepSeekConfig, ANEDeepSeekModel
    
    # Create model
    config = ANEDeepSeekConfig.tiny()
    model = ANEDeepSeekModel(config)
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    logits = model(input_ids)
    
    # Generation
    gen_config = ANEGenerationConfig(max_new_tokens=100)
    output_ids = model.generate(input_ids, gen_config)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Callable

from .layer import ANEDeepSeekLayer, ANEDeepSeekLayerConfig


@dataclass
class ANEDeepSeekConfig:
    """Configuration for the complete ANE-optimized DeepSeek model."""
    
    # Vocabulary
    vocab_size: int = 102400
    
    # Architecture
    num_layers: int = 32
    d_model: int = 4096
    num_heads: int = 32
    
    # Attention
    d_latent: int = 512  # Latent dimension for KV compression
    d_rope: int = 64  # RoPE dimension
    chunk_size: int = 128  # ANE-friendly chunk size
    max_seq_len: int = 8192  # Maximum sequence length
    
    # MoE
    d_hidden: int = 11008  # Hidden dimension in expert FFN
    num_experts: int = 32  # Number of routed experts (reduced from 256)
    num_shared: int = 2  # Number of shared experts
    top_k: int = 4  # Experts per token
    use_moe: bool = True  # Whether to use MoE
    
    # First N layers use dense FFN (no MoE)
    num_dense_layers: int = 1
    
    # Precision
    use_fp16: bool = True  # FP16 activations
    use_int8_weights: bool = True  # INT8 quantized weights
    
    # Dropout
    dropout: float = 0.0
    
    # Generation defaults
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    
    @classmethod
    def tiny(cls) -> "ANEDeepSeekConfig":
        """Tiny configuration for testing (< 50M params)."""
        return cls(
            vocab_size=32000,
            num_layers=4,
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
            chunk_size=128,
            max_seq_len=2048,
            d_hidden=512,
            num_experts=8,
            num_shared=1,
            top_k=2,
            use_moe=True,
            num_dense_layers=1,
            use_fp16=True,
            use_int8_weights=False,
        )
    
    @classmethod
    def small(cls) -> "ANEDeepSeekConfig":
        """Small configuration for development (~500M params)."""
        return cls(
            vocab_size=32000,
            num_layers=12,
            d_model=1024,
            num_heads=16,
            d_latent=256,
            d_rope=64,
            chunk_size=128,
            max_seq_len=4096,
            d_hidden=4096,
            num_experts=16,
            num_shared=2,
            top_k=4,
            use_moe=True,
            num_dense_layers=1,
            use_fp16=True,
            use_int8_weights=True,
        )
    
    @classmethod
    def standard(cls) -> "ANEDeepSeekConfig":
        """Standard configuration (~7B params)."""
        return cls(
            vocab_size=102400,
            num_layers=32,
            d_model=4096,
            num_heads=32,
            d_latent=512,
            d_rope=64,
            chunk_size=128,
            max_seq_len=8192,
            d_hidden=11008,
            num_experts=32,
            num_shared=2,
            top_k=4,
            use_moe=True,
            num_dense_layers=1,
            use_fp16=True,
            use_int8_weights=True,
        )
    
    @classmethod
    def deepseek_v3(cls) -> "ANEDeepSeekConfig":
        """DeepSeek-V3 style configuration (scaled for ANE)."""
        return cls(
            vocab_size=102400,
            num_layers=48,
            d_model=5120,
            num_heads=40,
            d_latent=512,
            d_rope=64,
            chunk_size=128,
            max_seq_len=8192,
            d_hidden=13824,
            num_experts=64,  # Reduced from 256
            num_shared=2,
            top_k=6,
            use_moe=True,
            num_dense_layers=2,
            use_fp16=True,
            use_int8_weights=True,
        )
    
    def get_layer_config(self, layer_idx: int) -> ANEDeepSeekLayerConfig:
        """Get configuration for a specific layer."""
        # First N layers are dense (no MoE)
        use_moe = self.use_moe and (layer_idx >= self.num_dense_layers)
        
        return ANEDeepSeekLayerConfig(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_latent=self.d_latent,
            d_rope=self.d_rope,
            chunk_size=self.chunk_size,
            d_hidden=self.d_hidden,
            num_experts=self.num_experts,
            num_shared=self.num_shared,
            top_k=self.top_k,
            use_moe=use_moe,
            use_fp16=self.use_fp16,
            use_int8_weights=self.use_int8_weights,
            dropout=self.dropout,
        )


@dataclass
class ANEGenerationConfig:
    """Configuration for text generation."""
    
    # Length control
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    
    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    # Stopping
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    
    # Generation mode
    do_sample: bool = True  # False for greedy decoding
    
    # Cache control
    use_cache: bool = True
    
    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")


class ANEDeepSeekModel(nn.Module):
    """
    Complete DeepSeek model optimized for Apple Neural Engine.
    
    Architecture:
    - ANEEmbedding: Token embeddings (CPU lookup → ANE transfer)
    - ANEDeepSeekLayer × N: Transformer layers (ANE execution)
    - ANERMSNorm: Final normalization
    - Linear head: Output projection (weight tied with embeddings)
    
    Features:
    - Unified memory KV cache (zero-copy on Apple Silicon)
    - INT8 weights, FP16 activations
    - Chunked attention for long context
    - MoE with automatic strategy selection
    - Autoregressive generation support
    """
    
    def __init__(self, config: ANEDeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Import ANE components
        from ..layers.base import ANEEmbedding, ANERMSNorm
        
        # Token embeddings
        self.embed_tokens = ANEEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            use_fp16=config.use_fp16,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ANEDeepSeekLayer(
                config=config.get_layer_config(layer_idx),
                layer_idx=layer_idx,
            )
            for layer_idx in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = ANERMSNorm(config.d_model, use_fp16=config.use_fp16)
        
        # Output projection (weight tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weight tying
        self._init_weights()
        self._tie_weights()
        
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                
    def _tie_weights(self):
        """Tie embedding and output projection weights."""
        # Get the raw embedding weight tensor
        if hasattr(self.embed_tokens, 'weight'):
            self.lm_head.weight = self.embed_tokens.weight
        elif hasattr(self.embed_tokens, 'embedding') and hasattr(self.embed_tokens.embedding, 'weight'):
            self.lm_head.weight = self.embed_tokens.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None] | None]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (1 = attend, 0 = mask)
            position_ids: Position indices [batch, seq_len]
            past_key_values: List of cached latent tensors for each layer
            use_cache: Whether to return updated KV caches
            
        Returns:
            Tuple of:
            - logits: Output logits [batch, seq_len, vocab_size]
            - new_past_key_values: Updated KV caches (if use_cache=True)
        """
        batch_size, seq_len = input_ids.shape
        
        # Calculate cache length from past_key_values
        cache_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            # past_key_values is a list of cached latent tensors
            first_cache = past_key_values[0]
            if first_cache is not None:
                cache_len = first_cache.shape[1]  # [batch, seq_len, d_latent]
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                cache_len, cache_len + seq_len,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Create causal attention mask if needed
        if attention_mask is None:
            # Default: causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
                diagonal=1
            )
            attention_mask = ~causal_mask  # True = attend
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Cast to FP16 if needed
        if self.config.use_fp16:
            hidden_states = hidden_states.half()
        
        # Calculate position offset for KV cache
        position_offset = 0
        if past_key_values is not None and len(past_key_values) > 0:
            # Get sequence length from first cached latent
            if past_key_values[0] is not None:
                position_offset = past_key_values[0].shape[1]
        
        # Process through layers
        new_past_key_values = [] if use_cache else None
        
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                position_offset=position_offset,
            )
            
            if use_cache:
                new_past_key_values.append(new_kv)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        # Cast to float32 for stable softmax
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()
        logits = self.lm_head(hidden_states)
        
        return logits, new_past_key_values
    
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: ANEGenerationConfig | None = None,
        stopping_criteria: list[Callable[[torch.Tensor, torch.Tensor], bool]] | None = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Initial token indices [batch, seq_len]
            generation_config: Generation configuration
            stopping_criteria: List of stopping functions
            
        Returns:
            Generated token indices [batch, total_len]
        """
        if generation_config is None:
            generation_config = ANEGenerationConfig()
            generation_config.eos_token_id = self.config.eos_token_id
            generation_config.pad_token_id = self.config.pad_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Determine EOS token(s)
        eos_token_ids = generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        elif eos_token_ids is None:
            eos_token_ids = [self.config.eos_token_id]
        
        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Initial forward pass to get KV cache
        past_key_values = None
        current_ids = input_ids
        generated_ids = input_ids.clone()
        
        for step in range(generation_config.max_new_tokens):
            # Forward pass
            logits, past_key_values = self.forward(
                current_ids,
                past_key_values=past_key_values,
                use_cache=generation_config.use_cache,
            )
            
            # Get logits for next token
            next_token_logits = logits[:, -1, :]  # [batch, vocab]
            
            # Apply temperature
            if generation_config.temperature != 1.0:
                next_token_logits = next_token_logits / generation_config.temperature
            
            # Apply repetition penalty
            if generation_config.repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_id in generated_ids[batch_idx].unique():
                        if next_token_logits[batch_idx, token_id] < 0:
                            next_token_logits[batch_idx, token_id] *= generation_config.repetition_penalty
                        else:
                            next_token_logits[batch_idx, token_id] /= generation_config.repetition_penalty
            
            # Sample or greedy decode
            if generation_config.do_sample:
                # Apply top-k filtering
                if generation_config.top_k > 0:
                    top_k = min(generation_config.top_k, next_token_logits.shape[-1])
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative prob above threshold
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            current_ids = next_token
            
            # Check for EOS
            for eos_id in eos_token_ids:
                done = done | (next_token.squeeze(-1) == eos_id)
            
            # Check stopping criteria
            if stopping_criteria:
                for criterion in stopping_criteria:
                    if criterion(generated_ids, None):
                        done = done | True
            
            # Stop if all sequences are done
            if done.all():
                break
            
            # Check minimum length
            if step < generation_config.min_new_tokens:
                # Don't stop early
                done = torch.zeros_like(done)
        
        return generated_ids
    
    def get_memory_footprint(self) -> dict:
        """Estimate total memory footprint of the model."""
        config = self.config
        
        # Embedding params
        embed_params = config.vocab_size * config.d_model
        
        # Layer params (sum across all layers)
        layer_params = 0
        for layer_idx in range(config.num_layers):
            layer_config = config.get_layer_config(layer_idx)
            layer = ANEDeepSeekLayer(layer_config, layer_idx)
            layer_info = layer.get_memory_footprint()
            layer_params += layer_info["total_params"]
        
        # Final norm params
        norm_params = config.d_model
        
        # Output head (tied with embeddings, so not counted again)
        head_params = 0  # Weight-tied
        
        total = embed_params + layer_params + norm_params + head_params
        
        # Calculate size
        weight_bytes = 1 if config.use_int8_weights else 2
        
        return {
            "embedding_params": embed_params,
            "layer_params": layer_params,
            "norm_params": norm_params,
            "total_params": total,
            "weight_size_mb": total * weight_bytes / (1024 * 1024),
            "weight_size_gb": total * weight_bytes / (1024 * 1024 * 1024),
        }
    
    def get_num_params(self, count_embeddings: bool = True) -> int:
        """Count total number of parameters."""
        total = 0
        for name, param in self.named_parameters():
            if not count_embeddings and 'embed' in name:
                continue
            # Don't double-count weight-tied params
            if name == 'lm_head.weight':
                continue
            total += param.numel()
        return total


class ANEDeepSeekForCausalLM(ANEDeepSeekModel):
    """
    Wrapper for causal language modeling with ANE-optimized DeepSeek.
    
    This class provides a HuggingFace-compatible interface for the model,
    making it easier to integrate with existing pipelines and tokenizers.
    """
    
    def __init__(self, config: ANEDeepSeekConfig):
        super().__init__(config)
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation step."""
        # Only use last token if we have a cache
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }


def create_model_from_config(config_name: str) -> ANEDeepSeekModel:
    """Factory function to create model from config name."""
    configs = {
        "tiny": ANEDeepSeekConfig.tiny,
        "small": ANEDeepSeekConfig.small,
        "standard": ANEDeepSeekConfig.standard,
        "deepseek_v3": ANEDeepSeekConfig.deepseek_v3,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]()
    return ANEDeepSeekModel(config)
