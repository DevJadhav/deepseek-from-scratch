import mlx.core as mx
import mlx.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


# ============================================================================
# Extended RoPE with NTK-aware Scaling and YaRN Interpolation
# ============================================================================

class RoPEScalingType(Enum):
    """RoPE scaling methods for extended context."""
    NONE = "none"
    LINEAR = "linear"
    NTK_AWARE = "ntk_aware"
    DYNAMIC_NTK = "dynamic_ntk"
    YARN = "yarn"


@dataclass
class ExtendedRoPEConfig:
    """Configuration for Extended RoPE."""
    d_head: int = 64
    max_seq_len: int = 131072  # 128K context
    base: float = 10000.0
    scaling_type: RoPEScalingType = RoPEScalingType.NTK_AWARE
    # NTK scaling
    ntk_alpha: float = 8.0  # Scale factor for NTK
    # YaRN parameters
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 0.707
    # Original trained length
    original_max_seq_len: int = 4096
    
    @classmethod
    def for_128k(cls, d_head: int = 64) -> "ExtendedRoPEConfig":
        """Create config for 128K context with NTK-aware scaling."""
        return cls(
            d_head=d_head,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=8.0,
        )
    
    @classmethod
    def for_128k_yarn(cls, d_head: int = 64) -> "ExtendedRoPEConfig":
        """Create config for 128K context with YaRN interpolation."""
        return cls(
            d_head=d_head,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.YARN,
            yarn_mscale=0.707,
        )


class ExtendedRotaryPositionalEncoding(nn.Module):
    """
    Extended RoPE with NTK-aware scaling and YaRN interpolation.
    
    Supports:
    - Linear interpolation
    - NTK-aware scaling (DeepSeek-V3 style)
    - Dynamic NTK
    - YaRN (Yet another RoPE extensioN)
    """
    
    def __init__(self, config: ExtendedRoPEConfig):
        super().__init__()
        self.config = config
        self.d_head = config.d_head
        self.base = config.base
        self.scaling_type = config.scaling_type
        
        # Compute scaling factor
        self.scale = self._compute_scale()
        
        # Compute inverse frequencies with scaling
        self.inv_freq = self._compute_inv_freq()
        
        # YaRN mscale for attention logits
        self.mscale = self._compute_mscale()
    
    def _compute_scale(self) -> float:
        """Compute context extension scale factor."""
        ratio = self.config.max_seq_len / self.config.original_max_seq_len
        
        if self.scaling_type == RoPEScalingType.LINEAR:
            return ratio
        elif self.scaling_type == RoPEScalingType.NTK_AWARE:
            # NTK-aware: scale base frequency
            return self.config.ntk_alpha
        elif self.scaling_type == RoPEScalingType.DYNAMIC_NTK:
            return ratio
        elif self.scaling_type == RoPEScalingType.YARN:
            return ratio
        else:
            return 1.0
    
    def _compute_inv_freq(self) -> mx.array:
        """Compute inverse frequencies with scaling applied."""
        dim = self.d_head
        
        if self.scaling_type == RoPEScalingType.NTK_AWARE:
            # Scale the base frequency
            scaled_base = self.base * (self.config.ntk_alpha ** (dim / (dim - 2)))
            inv_freq = 1.0 / (scaled_base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        elif self.scaling_type == RoPEScalingType.LINEAR:
            inv_freq = 1.0 / (self.base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
            inv_freq = inv_freq / self.scale
        elif self.scaling_type == RoPEScalingType.YARN:
            inv_freq = self._compute_yarn_inv_freq()
        else:
            inv_freq = 1.0 / (self.base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        
        return inv_freq
    
    def _compute_yarn_inv_freq(self) -> mx.array:
        """Compute YaRN interpolated frequencies."""
        dim = self.d_head
        ratio = self.config.max_seq_len / self.config.original_max_seq_len
        
        # Standard frequencies
        base_inv_freq = 1.0 / (self.base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        
        # Wavelength thresholds
        low_freq_wavelen = self.config.original_max_seq_len / self.config.yarn_beta_slow
        high_freq_wavelen = self.config.original_max_seq_len / self.config.yarn_beta_fast
        
        # Current wavelengths
        wavelens = 2 * math.pi / base_inv_freq
        
        # Interpolation factor (0 = no scaling, 1 = full linear scaling)
        smooth = mx.clip(
            (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen),
            0.0, 1.0
        )
        
        # Interpolate between original and scaled frequencies
        scaled_inv_freq = base_inv_freq / ratio
        inv_freq = (1 - smooth) * base_inv_freq + smooth * scaled_inv_freq
        
        return inv_freq
    
    def _compute_mscale(self) -> float:
        """Compute magnitude scale for YaRN."""
        if self.scaling_type == RoPEScalingType.YARN:
            ratio = self.config.max_seq_len / self.config.original_max_seq_len
            return self.config.yarn_mscale * math.sqrt(1 + math.log(ratio) / math.log(self.config.original_max_seq_len))
        return 1.0
    
    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """
        Apply extended RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (batch, heads, seq_len, d_head)
            offset: Position offset for KV cache
        
        Returns:
            Rotated tensor of same shape
        """
        batch, heads, seq_len, d_head = x.shape
        
        # Compute positions
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        
        # Compute frequencies
        freqs = mx.outer(positions, self.inv_freq)  # (seq_len, d_head/2)
        
        # Compute cos and sin with mscale
        cos = mx.cos(freqs) * self.mscale
        sin = mx.sin(freqs) * self.mscale
        
        # Reshape for rotation
        x_reshape = x.reshape(batch, heads, seq_len, d_head // 2, 2)
        x_real = x_reshape[..., 0]
        x_imag = x_reshape[..., 1]
        
        # Broadcast cos/sin
        cos = cos.reshape(1, 1, seq_len, d_head // 2)
        sin = sin.reshape(1, 1, seq_len, d_head // 2)
        
        # Apply rotation
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        
        # Stack and reshape
        out = mx.stack([out_real, out_imag], axis=-1)
        return out.reshape(batch, heads, seq_len, d_head)


# ============================================================================
# Original Attention Implementations
# ============================================================================

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.W_q(x).reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.W_k(x).reshape(B, T, 1, self.d_head).transpose(0, 2, 1, 3)
        v = self.W_v(x).reshape(B, T, 1, self.d_head).transpose(0, 2, 1, 3)
        
        # Broadcast K, V across heads
        # MLX handles broadcasting automatically in matmul if dims match or are 1
        
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask
            
        attn = nn.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return self.W_o(out)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_head = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_groups * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, num_groups * self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.W_q(x).reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.W_k(x).reshape(B, T, self.num_groups, self.d_head).transpose(0, 2, 1, 3)
        v = self.W_v(x).reshape(B, T, self.num_groups, self.d_head).transpose(0, 2, 1, 3)
        
        # Repeat K, V for each group
        # num_heads // num_groups times
        repeats = self.num_heads // self.num_groups
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask
            
        attn = nn.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return self.W_o(out)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, d_rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.d_head = d_model // num_heads
        self.d_rope = d_rope
        
        # Compression
        self.W_DK = nn.Linear(d_model, d_latent, bias=False)
        self.W_UV = nn.Linear(d_latent, num_heads * self.d_head, bias=False)
        
        self.W_DQ = nn.Linear(d_model, d_latent, bias=False)
        self.W_UQ = nn.Linear(d_latent, num_heads * self.d_head, bias=False)
        
        if d_rope:
            self.W_KR = nn.Linear(d_model, d_rope, bias=False)
            self.W_QR = nn.Linear(d_model, d_rope * num_heads, bias=False)
            
        self.W_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)
        
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        # Key-Value Compression
        c_kv = self.W_DK(x)
        k_c = self.W_UV(c_kv).reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        v_c = k_c # Shared for K and V in simplified MLA
        
        # Query Compression
        c_q = self.W_DQ(x)
        q_c = self.W_UQ(c_q).reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        
        if self.d_rope:
            k_r = self.W_KR(x).reshape(B, T, 1, self.d_rope).transpose(0, 2, 1, 3)
            q_r = self.W_QR(x).reshape(B, T, self.num_heads, self.d_rope).transpose(0, 2, 1, 3)
            # RoPE application would go here, simplified for now
            
            # Concatenate
            q = mx.concatenate([q_c, q_r], axis=-1)
            k = mx.concatenate([k_c, mx.broadcast_to(k_r, (B, self.num_heads, T, self.d_rope))], axis=-1)
        else:
            q = q_c
            k = k_c
            
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head + (self.d_rope if self.d_rope else 0))
        
        if mask is not None:
            scores = scores + mask
            
        attn = nn.softmax(scores, axis=-1)
        out = (attn @ v_c).transpose(0, 2, 1, 3).reshape(B, T, self.num_heads * self.d_head)
        
        return self.W_o(out)
