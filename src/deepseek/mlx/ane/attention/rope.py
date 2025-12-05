"""
ANE-Optimized Rotary Positional Encoding (RoPE)

This module provides ANE-optimized RoPE implementations with:
- Standard RoPE
- NTK-Aware scaling for extended context
- YaRN interpolation for 128K+ context
- Dynamic NTK scaling

Key optimizations:
- FP16 computation for ANE efficiency
- Pre-computed frequency tables
- Decoupled positional encoding for MLA
"""

import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class RoPEScalingType(Enum):
    """RoPE scaling methods for extended context."""

    NONE = "none"
    LINEAR = "linear"
    NTK_AWARE = "ntk_aware"
    DYNAMIC_NTK = "dynamic_ntk"
    YARN = "yarn"


@dataclass
class ANERoPEConfig:
    """Configuration for ANE-optimized RoPE."""

    d_head: int = 64
    max_seq_len: int = 131072  # 128K context
    base: float = 10000.0
    scaling_type: RoPEScalingType = RoPEScalingType.NTK_AWARE
    # NTK scaling parameters
    ntk_alpha: float = 8.0
    # YaRN parameters
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 0.707
    # Original trained context length
    original_max_seq_len: int = 4096
    # ANE optimization
    use_fp16: bool = True

    @classmethod
    def for_4k(cls, d_head: int = 64) -> "ANERoPEConfig":
        """Create config for 4K context (no scaling)."""
        return cls(
            d_head=d_head,
            max_seq_len=4096,
            scaling_type=RoPEScalingType.NONE,
        )

    @classmethod
    def for_32k_ntk(cls, d_head: int = 64) -> "ANERoPEConfig":
        """Create config for 32K context with NTK-aware scaling."""
        return cls(
            d_head=d_head,
            max_seq_len=32768,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=4.0,
        )

    @classmethod
    def for_128k_ntk(cls, d_head: int = 64) -> "ANERoPEConfig":
        """Create config for 128K context with NTK-aware scaling."""
        return cls(
            d_head=d_head,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=8.0,
        )

    @classmethod
    def for_128k_yarn(cls, d_head: int = 64) -> "ANERoPEConfig":
        """Create config for 128K context with YaRN interpolation."""
        return cls(
            d_head=d_head,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.YARN,
            yarn_mscale=0.707,
        )


class ANERoPE(nn.Module):
    """
    ANE-Optimized Rotary Positional Encoding.

    Supports multiple scaling methods for extended context:
    - None: Standard RoPE (up to trained context length)
    - Linear: Simple frequency scaling (degrades quality)
    - NTK-Aware: Scale base frequency (DeepSeek-V3 style)
    - Dynamic NTK: Runtime adaptive scaling
    - YaRN: Frequency-selective interpolation (best quality)

    Features:
    - FP16 computation for ANE efficiency
    - Pre-computed sin/cos tables
    - Efficient position offset for KV cache
    - Decoupled RoPE support for MLA

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ RoPE Rotation: x_rot = x * cos(θ) + rotate_half(x) * sin(θ)    │
    │                                                                 │
    │ Scaling Methods:                                                │
    │ • NTK: θ'ᵢ = θᵢ × (α^(d/(d-2)))                               │
    │ • YaRN: θ'ᵢ = interpolate(θᵢ, scale, wavelength)              │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: ANERoPEConfig):
        super().__init__()
        self.config = config
        self.d_head = config.d_head
        self.base = config.base
        self.max_seq_len = config.max_seq_len
        self.scaling_type = config.scaling_type
        self.use_fp16 = config.use_fp16

        # Compute inverse frequencies with scaling
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Compute magnitude scale for YaRN
        self.mscale = self._compute_mscale()

        # Pre-compute cos/sin tables for efficiency
        self._build_cos_sin_cache()

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with appropriate scaling."""
        dim = self.d_head

        if self.scaling_type == RoPEScalingType.NONE:
            # Standard RoPE
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, dim, 2).float() / dim)
            )

        elif self.scaling_type == RoPEScalingType.LINEAR:
            # Linear scaling: divide frequencies by scale factor
            scale = self.config.max_seq_len / self.config.original_max_seq_len
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, dim, 2).float() / dim)
            )
            inv_freq = inv_freq / scale

        elif self.scaling_type == RoPEScalingType.NTK_AWARE:
            # NTK-aware: scale the base frequency
            # θ' = θ × (α ^ (d / (d - 2)))
            alpha = self.config.ntk_alpha
            scaled_base = self.base * (alpha ** (dim / (dim - 2)))
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, dim, 2).float() / dim)
            )

        elif self.scaling_type == RoPEScalingType.DYNAMIC_NTK:
            # Dynamic NTK: computed at runtime based on actual sequence length
            # For pre-computation, use max scaling
            scale = self.config.max_seq_len / self.config.original_max_seq_len
            alpha = (scale * self.config.ntk_alpha - 1) / (
                self.config.ntk_alpha - 1
            )
            scaled_base = self.base * (alpha ** (dim / (dim - 2)))
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, dim, 2).float() / dim)
            )

        elif self.scaling_type == RoPEScalingType.YARN:
            # YaRN: frequency-selective interpolation
            inv_freq = self._compute_yarn_inv_freq()

        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

        return inv_freq

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """
        Compute YaRN interpolated frequencies.

        YaRN (Yet another RoPE extensioN) applies different scaling
        to different frequency bands:
        - High frequencies: keep original (local patterns)
        - Low frequencies: interpolate (long-range patterns)
        - Middle frequencies: smooth blend
        """
        dim = self.d_head
        ratio = self.config.max_seq_len / self.config.original_max_seq_len

        # Standard inverse frequencies
        base_inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2).float() / dim)
        )

        # Wavelength thresholds
        low_freq_wavelen = (
            self.config.original_max_seq_len / self.config.yarn_beta_slow
        )
        high_freq_wavelen = (
            self.config.original_max_seq_len / self.config.yarn_beta_fast
        )

        # Current wavelengths: λ = 2π / θ
        wavelens = 2 * math.pi / base_inv_freq

        # Smooth interpolation factor
        # 0 = no scaling (high freq), 1 = full scaling (low freq)
        smooth = torch.clamp(
            (wavelens - high_freq_wavelen)
            / (low_freq_wavelen - high_freq_wavelen),
            0.0,
            1.0,
        )

        # Interpolate between original and scaled frequencies
        scaled_inv_freq = base_inv_freq / ratio
        inv_freq = (1 - smooth) * base_inv_freq + smooth * scaled_inv_freq

        return inv_freq

    def _compute_mscale(self) -> float:
        """
        Compute magnitude scale for YaRN attention.

        YaRN uses a magnitude correction to maintain attention
        distribution quality at extended context lengths.
        """
        if self.scaling_type == RoPEScalingType.YARN:
            ratio = self.config.max_seq_len / self.config.original_max_seq_len
            return self.config.yarn_mscale * math.sqrt(
                1 + math.log(ratio) / math.log(self.config.original_max_seq_len)
            )
        return 1.0

    def _build_cos_sin_cache(self):
        """Pre-compute cos/sin tables for the full sequence length."""
        # Position indices
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)

        # Compute frequencies: θ × position
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, d_head/2)

        # Compute cos/sin with mscale
        cos_cached = torch.cos(freqs) * self.mscale
        sin_cached = torch.sin(freqs) * self.mscale

        # Convert to FP16 if needed
        if self.use_fp16:
            cos_cached = cos_cached.half()
            sin_cached = sin_cached.half()

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor (batch, heads, seq_len, d_head)
            position_offset: Starting position for KV cache inference

        Returns:
            Rotated tensor of same shape
        """
        batch, heads, seq_len, d_head = x.shape

        # Get pre-computed cos/sin for current positions
        end_pos = position_offset + seq_len
        cos = self.cos_cached[position_offset:end_pos]  # (seq_len, d_head/2)
        sin = self.sin_cached[position_offset:end_pos]

        # Reshape for broadcasting: (1, 1, seq_len, d_head/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        return self._apply_rotation(x, cos, sin)

    def _apply_rotation(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary transformation.

        x_rot = x * cos + rotate_half(x) * sin

        Where rotate_half swaps and negates pairs of dimensions:
        [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        """
        # Split into pairs for rotation
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshape[..., 0]  # Even indices
        x_imag = x_reshape[..., 1]  # Odd indices

        # Apply rotation using complex multiplication formula
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        # Interleave back
        out = torch.stack([out_real, out_imag], dim=-1)
        return out.reshape(x.shape)

    def forward_q_k(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to both query and key tensors.

        Optimized for attention computation where both Q and K
        need the same rotation.

        Args:
            q: Query tensor (batch, heads, seq_len, d_head)
            k: Key tensor (batch, heads, seq_len, d_head)
            position_offset: Starting position for KV cache

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        batch, heads, seq_len, d_head = q.shape

        # Get pre-computed cos/sin
        end_pos = position_offset + seq_len
        cos = self.cos_cached[position_offset:end_pos].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[position_offset:end_pos].unsqueeze(0).unsqueeze(0)

        # Apply same rotation to both
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)

        return q_rot, k_rot

    def get_dynamic_ntk_frequencies(
        self, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get dynamically computed frequencies for Dynamic NTK scaling.

        Used when the actual sequence length exceeds the original
        training length, computing appropriate scaling on-the-fly.

        Args:
            seq_len: Current sequence length

        Returns:
            Tuple of (cos, sin) for the sequence
        """
        if seq_len <= self.config.original_max_seq_len:
            # No scaling needed
            return (
                self.cos_cached[:seq_len],
                self.sin_cached[:seq_len],
            )

        # Compute dynamic scaling factor
        scale = seq_len / self.config.original_max_seq_len
        alpha = (scale * self.config.ntk_alpha - 1) / (
            self.config.ntk_alpha - 1
        )
        scaled_base = self.base * (alpha ** (self.d_head / (self.d_head - 2)))

        # Recompute frequencies
        inv_freq = 1.0 / (
            scaled_base
            ** (torch.arange(0, self.d_head, 2, device=self.inv_freq.device).float() / self.d_head)
        )

        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, inv_freq)

        cos = torch.cos(freqs) * self.mscale
        sin = torch.sin(freqs) * self.mscale

        if self.use_fp16:
            cos = cos.half()
            sin = sin.half()

        return cos, sin

    def extra_repr(self) -> str:
        return (
            f"d_head={self.d_head}, max_seq_len={self.max_seq_len}, "
            f"scaling={self.scaling_type.value}"
        )
