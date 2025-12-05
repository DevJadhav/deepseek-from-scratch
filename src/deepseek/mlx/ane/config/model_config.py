"""
ANE Model Configuration

Configuration classes for ANE-optimized models:
- ANEModelConfig: Base configuration for ANE models
- ANELayerConfig: Per-layer configuration options
- ANEQuantConfig: Quantization configuration
- TinyANEConfig: Configuration for testing with small models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from ..utils.quantization import QuantizationType


class ANEPrecision(Enum):
    """Precision modes for ANE computation."""
    FP16 = "fp16"  # FP16 activations, INT8 weights
    INT8 = "int8"  # INT8 activations and weights
    MIXED = "mixed"  # FP16 for attention, INT8 for FFN


@dataclass
class ANEQuantConfig:
    """Quantization configuration for ANE models."""

    # Weight quantization
    weight_quant_type: QuantizationType = QuantizationType.INT8_PER_CHANNEL
    weight_block_size: int = 128  # Block size for INT4 quantization

    # Activation quantization (typically FP16)
    activation_dtype: Literal["fp16", "fp32"] = "fp16"

    # KV Cache quantization
    kv_cache_dtype: Literal["fp16", "int8", "k8v4"] = "fp16"

    # Which layers to quantize
    quantize_embeddings: bool = False  # Embeddings stay FP16
    quantize_attention: bool = True
    quantize_ffn: bool = True
    quantize_output: bool = False  # Output projection stays FP16

    def __post_init__(self):
        """Validate configuration."""
        if self.weight_block_size < 32 or self.weight_block_size > 256:
            raise ValueError(
                f"weight_block_size must be between 32 and 256, "
                f"got {self.weight_block_size}"
            )


@dataclass
class ANELayerConfig:
    """Per-layer configuration for ANE optimization."""

    # Dimension constraints
    pad_to_multiple: int = 16  # Pad dimensions to multiples of this
    prefer_power_of_2: bool = False  # Prefer power-of-2 dimensions

    # Tiling for large operations
    tile_size: int = 128  # Tile size for tiled matmul

    # Attention chunking
    attention_chunk_size: int = 128  # Chunk size for chunked attention

    # Memory layout
    use_channel_last: bool = True  # Use NHWC layout


@dataclass
class ANEModelConfig:
    """
    Configuration for ANE-optimized DeepSeek model.

    This configuration defines the model architecture and ANE-specific
    optimizations.
    """

    # Model architecture
    vocab_size: int = 102400
    num_layers: int = 32
    d_model: int = 4096  # Model dimension
    num_heads: int = 32  # Number of attention heads
    d_head: int = 128  # Head dimension (d_model / num_heads)

    # MLA (Multi-Latent Attention) configuration
    d_latent: int = 512  # Latent dimension for KV compression
    d_rope: int = 64  # RoPE dimension (decoupled positional encoding)
    use_mla: bool = True  # Use Multi-Latent Attention

    # FFN configuration
    d_hidden: int = 11008  # FFN hidden dimension (typically ~2.7x d_model)
    ffn_type: Literal["swiglu", "gelu", "relu"] = "swiglu"

    # MoE configuration
    num_experts: int = 0  # 0 = dense model, >0 = MoE
    num_shared_experts: int = 2  # Shared experts (always active)
    num_routed_experts: int = 6  # Routed experts per token
    top_k: int = 4  # Number of experts to route to

    # Sequence length
    max_seq_len: int = 8192  # Maximum sequence length

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Dropout (0 for inference)
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # ANE-specific configuration
    layer_config: ANELayerConfig = field(default_factory=ANELayerConfig)
    quant_config: ANEQuantConfig = field(default_factory=ANEQuantConfig)

    # Precision
    precision: ANEPrecision = ANEPrecision.FP16

    # RoPE scaling
    rope_theta: float = 10000.0
    rope_scaling_type: Literal["none", "linear", "ntk", "yarn"] = "none"
    rope_scaling_factor: float = 1.0

    def __post_init__(self):
        """Validate and compute derived values."""
        # Compute head dimension if not explicitly set
        if self.d_head is None:
            self.d_head = self.d_model // self.num_heads

        # Validate dimensions
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.d_model // self.num_heads != self.d_head:
            raise ValueError(
                f"d_model / num_heads ({self.d_model // self.num_heads}) "
                f"doesn't match d_head ({self.d_head})"
            )

        # Validate MLA configuration
        if self.use_mla:
            if self.d_latent <= 0:
                raise ValueError("d_latent must be positive when using MLA")
            if self.d_latent >= self.num_heads * self.d_head:
                raise ValueError(
                    f"d_latent ({self.d_latent}) should be smaller than "
                    f"num_heads * d_head ({self.num_heads * self.d_head}) "
                    "for compression benefit"
                )

    @property
    def head_dim(self) -> int:
        """Get head dimension."""
        return self.d_head

    @property
    def kv_dim(self) -> int:
        """Get KV dimension (for MLA, this is d_latent)."""
        return self.d_latent if self.use_mla else self.num_heads * self.d_head

    @property
    def is_moe(self) -> bool:
        """Check if model uses Mixture of Experts."""
        return self.num_experts > 0

    @property
    def memory_estimate_gb(self) -> float:
        """Estimate model memory in GB (FP16 weights)."""
        # Embedding: vocab_size * d_model
        embed_params = self.vocab_size * self.d_model

        # Per-layer params (approximate)
        # Attention: Q, K, V, O projections
        if self.use_mla:
            attn_params = (
                self.d_model * self.num_heads * self.d_head +  # Q
                self.d_model * self.d_latent +  # KV down
                self.d_latent * self.num_heads * self.d_head * 2 +  # K, V up
                self.num_heads * self.d_head * self.d_model  # O
            )
        else:
            attn_params = 4 * self.d_model * self.d_model

        # FFN (SwiGLU has 3 projections)
        ffn_params = 3 * self.d_model * self.d_hidden

        # MoE
        if self.is_moe:
            expert_params = ffn_params * (self.num_shared_experts + self.num_experts)
            router_params = self.d_model * self.num_experts
            ffn_params = expert_params + router_params
        else:
            ffn_params = ffn_params

        # Norms
        norm_params = 2 * self.d_model  # Pre-attention and pre-FFN norm

        layer_params = attn_params + ffn_params + norm_params
        total_params = embed_params + self.num_layers * layer_params

        # FP16 = 2 bytes per param
        return total_params * 2 / (1024**3)


def TinyANEConfig() -> ANEModelConfig:
    """
    Create a tiny model configuration for testing.

    This configuration creates a small model suitable for:
    - Unit testing
    - Debugging ANE integration
    - Quick validation of changes
    """
    return ANEModelConfig(
        vocab_size=1024,
        num_layers=2,
        d_model=256,
        num_heads=4,
        d_head=64,
        d_latent=64,
        d_rope=32,
        use_mla=True,
        d_hidden=512,
        ffn_type="swiglu",
        num_experts=0,
        max_seq_len=512,
        dropout=0.0,
        attention_dropout=0.0,
        layer_config=ANELayerConfig(
            pad_to_multiple=16,
            tile_size=64,
            attention_chunk_size=64,
        ),
        quant_config=ANEQuantConfig(
            weight_quant_type=QuantizationType.INT8_PER_CHANNEL,
            activation_dtype="fp16",
        ),
        precision=ANEPrecision.FP16,
    )


def SmallANEConfig() -> ANEModelConfig:
    """
    Create a small model configuration.

    Similar to LLaMA-7B scale but with ANE optimizations.
    Suitable for testing on Apple Silicon devices.
    """
    return ANEModelConfig(
        vocab_size=32000,
        num_layers=16,
        d_model=2048,
        num_heads=16,
        d_head=128,
        d_latent=256,
        d_rope=64,
        use_mla=True,
        d_hidden=5632,
        ffn_type="swiglu",
        num_experts=0,
        max_seq_len=4096,
        dropout=0.0,
        layer_config=ANELayerConfig(
            pad_to_multiple=16,
            tile_size=128,
            attention_chunk_size=128,
        ),
        quant_config=ANEQuantConfig(
            weight_quant_type=QuantizationType.INT8_PER_CHANNEL,
            activation_dtype="fp16",
        ),
        precision=ANEPrecision.FP16,
    )


def DeepSeekV3ANEConfig() -> ANEModelConfig:
    """
    Create DeepSeek-V3 style configuration for ANE.

    Note: The full 256-expert MoE is likely too large for ANE.
    Consider using distilled/reduced expert counts for ANE inference.
    """
    return ANEModelConfig(
        vocab_size=102400,
        num_layers=32,
        d_model=4096,
        num_heads=32,
        d_head=128,
        d_latent=512,
        d_rope=64,
        use_mla=True,
        d_hidden=11008,
        ffn_type="swiglu",
        # Reduced MoE for ANE compatibility
        num_experts=32,  # Reduced from 256
        num_shared_experts=2,
        num_routed_experts=6,
        top_k=4,
        max_seq_len=8192,
        dropout=0.0,
        layer_config=ANELayerConfig(
            pad_to_multiple=16,
            tile_size=128,
            attention_chunk_size=128,
        ),
        quant_config=ANEQuantConfig(
            weight_quant_type=QuantizationType.INT8_PER_CHANNEL,
            activation_dtype="fp16",
            kv_cache_dtype="fp16",
        ),
        precision=ANEPrecision.FP16,
        rope_scaling_type="ntk",
        rope_scaling_factor=2.0,
    )
