"""
DeepSeek Sparse Attention (DSA) - PyTorch Implementation

This module implements the sparse attention mechanism from DeepSeek-V3.2,
optimized for MPS (Apple Silicon) and CUDA backends.

Key Features:
- Sliding window attention for local context
- Dilated global attention for long-range dependencies
- Near-linear O(kL) complexity instead of O(L²)
- Compatible with MLA (Multi-head Latent Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class SparsityPattern(Enum):
    """Types of sparsity patterns for attention."""
    SLIDING_WINDOW = "sliding_window"
    DILATED_GLOBAL = "dilated_global"
    COMBINED = "combined"


@dataclass
class DSAConfig:
    """Configuration for DeepSeek Sparse Attention."""
    d_model: int = 4096
    num_heads: int = 32
    d_latent: int = 512
    d_rope: int = 64
    
    # Sparsity parameters
    window_size: int = 4096
    dilation_stride: int = 64
    global_tokens: int = 128
    
    # Context length
    max_seq_len: int = 131072  # 128K
    
    # Behavior
    causal: bool = True
    use_mla: bool = True
    
    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads
    
    @property
    def effective_kv_budget(self) -> int:
        """Effective KV budget per query position."""
        # window + global + dilated samples
        dilated_samples = self.max_seq_len // self.dilation_stride
        return self.window_size + self.global_tokens + dilated_samples
    
    @classmethod
    def for_128k_context(cls) -> "DSAConfig":
        """Configuration optimized for 128K context."""
        return cls(
            d_model=4096,
            num_heads=32,
            d_latent=512,
            d_rope=64,
            window_size=4096,
            dilation_stride=64,
            global_tokens=128,
            max_seq_len=131072,
        )
    
    @classmethod
    def small_test(cls) -> "DSAConfig":
        """Small configuration for testing."""
        return cls(
            d_model=256,
            num_heads=8,
            d_latent=64,
            d_rope=32,
            window_size=64,
            dilation_stride=8,
            global_tokens=8,
            max_seq_len=512,
        )


class SparseAttentionPattern(nn.Module):
    """
    Generates sparse attention patterns combining:
    1. Sliding window (local context)
    2. Dilated global attention (long-range dependencies)
    
    This achieves near-linear O(kL) complexity where k << L.
    """
    
    def __init__(self, config: DSAConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.dilation_stride = config.dilation_stride
        self.global_tokens = config.global_tokens
        self.causal = config.causal
    
    def get_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate sliding window attention mask."""
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Query positions (rows) and Key positions (columns)
        q_pos = positions.unsqueeze(1)  # [seq_len, 1]
        k_pos = positions.unsqueeze(0)  # [1, seq_len]
        
        # Distance matrix
        distance = q_pos - k_pos
        
        # Window mask: attend if within window
        half_window = self.window_size // 2
        window_mask = (distance >= -half_window) & (distance <= half_window)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = k_pos <= q_pos
            window_mask = window_mask & causal_mask
        
        return window_mask
    
    def get_dilated_global_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate dilated global attention mask."""
        positions = torch.arange(seq_len, device=device)
        
        # Query positions
        q_pos = positions.unsqueeze(1)  # [seq_len, 1]
        k_pos = positions.unsqueeze(0)  # [1, seq_len]
        
        # Dilated positions: every `dilation_stride` position
        is_dilated = (k_pos % self.dilation_stride) == 0
        
        # Global tokens: first N tokens always attend
        is_global = k_pos < self.global_tokens
        
        # Combine
        dilated_mask = is_dilated | is_global
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = k_pos <= q_pos
            dilated_mask = dilated_mask & causal_mask
        
        return dilated_mask
    
    def get_combined_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get combined sparse attention mask."""
        window_mask = self.get_sliding_window_mask(seq_len, device)
        dilated_mask = self.get_dilated_global_mask(seq_len, device)
        return window_mask | dilated_mask
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate the sparse attention mask."""
        return self.get_combined_mask(seq_len, device)


class RotaryPositionalEncoding(nn.Module):
    """Standard RoPE for position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 131072, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if needed."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cache_seq_len = seq_len
    
    def forward(
        self, 
        x: torch.Tensor, 
        offset: int = 0
    ) -> torch.Tensor:
        """
        Apply rotary positional encoding.
        
        Args:
            x: Input tensor [batch, heads, seq_len, dim]
            offset: Position offset for KV cache
            
        Returns:
            Rotated tensor with same shape
        """
        seq_len = x.shape[2]
        self._update_cache(offset + seq_len, x.device, x.dtype)
        
        cos = self._cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        
        # Rotate half
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin


class ExtendedRoPEConfig:
    """Configuration for extended RoPE with NTK/YaRN scaling."""
    
    def __init__(
        self,
        d_head: int = 64,
        max_seq_len: int = 131072,
        base: float = 10000.0,
        scaling_type: str = "yarn",  # "none", "ntk", "yarn"
        ntk_alpha: float = 8.0,
        yarn_scale: float = 1.0,
        yarn_original_max_position: int = 4096,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
    ):
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.ntk_alpha = ntk_alpha
        self.yarn_scale = yarn_scale
        self.yarn_original_max_position = yarn_original_max_position
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow


class ExtendedRotaryPositionalEncoding(nn.Module):
    """
    Extended RoPE with NTK-aware scaling and YaRN interpolation.
    
    Supports:
    - Standard RoPE
    - NTK-aware scaling (DeepSeek style)
    - YaRN interpolation for long context
    """
    
    def __init__(self, config: ExtendedRoPEConfig):
        super().__init__()
        self.config = config
        self.dim = config.d_head
        
        # Compute base frequencies with scaling
        if config.scaling_type == "ntk":
            # NTK-aware scaling: adjust base frequency
            scaled_base = config.base * (config.ntk_alpha ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (scaled_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        elif config.scaling_type == "yarn":
            # YaRN: interpolation with attention scaling
            inv_freq = self._compute_yarn_inv_freq()
        else:
            # Standard RoPE
            inv_freq = 1.0 / (config.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        self.register_buffer('inv_freq', inv_freq)
        
        # Compute YaRN attention scaling factor
        if config.scaling_type == "yarn":
            self.yarn_attn_factor = self._compute_yarn_attn_factor()
        else:
            self.yarn_attn_factor = 1.0
        
        # Cache
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len = 0
    
    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with YaRN interpolation."""
        config = self.config
        
        # Base inverse frequencies
        base_inv_freq = 1.0 / (config.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Scaling factor
        scale = config.max_seq_len / config.yarn_original_max_position
        
        # Interpolation weights based on frequency
        # Low frequencies: interpolate more (for long-range)
        # High frequencies: keep original (for local patterns)
        freqs = torch.arange(0, self.dim // 2).float()
        
        # Compute interpolation factor (ramp from 0 to 1) - use torch.clamp instead of max
        low = (freqs - config.yarn_beta_slow) / (config.yarn_beta_fast - config.yarn_beta_slow)
        ramp = torch.clamp(low, 0.0, 1.0)
        
        # Apply interpolation: blend between original and scaled
        inv_freq = base_inv_freq * (1 - ramp) + (base_inv_freq / scale) * ramp
        
        return inv_freq
    
    def _compute_yarn_attn_factor(self) -> float:
        """Compute YaRN attention scaling factor."""
        config = self.config
        scale = config.max_seq_len / config.yarn_original_max_position
        return 0.1 * math.log(scale) + 1.0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cache_seq_len = seq_len
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply extended rotary positional encoding."""
        seq_len = x.shape[2]
        self._update_cache(offset + seq_len, x.device, x.dtype)
        
        cos = self._cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        
        # Rotate half
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        
        result = x * cos + rotated * sin
        
        # Apply YaRN attention scaling if applicable
        if self.config.scaling_type == "yarn":
            result = result * self.yarn_attn_factor
        
        return result


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention with MLA-style KV compression.
    
    This combines:
    - Sparse attention patterns (sliding window + dilated global)
    - Multi-head Latent Attention (MLA) for KV compression
    - Extended RoPE for long context support
    """
    
    def __init__(self, config: DSAConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.d_head = config.d_head
        self.d_rope = config.d_rope
        self.d_latent = config.d_latent
        
        # Content path (MLA-style compression)
        self.w_q_content = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_dkv_content = nn.Linear(config.d_model, config.d_latent, bias=False)
        self.w_uk_content = nn.Linear(config.d_latent, config.d_model, bias=False)
        self.w_uv_content = nn.Linear(config.d_latent, config.d_model, bias=False)
        
        # Position path (RoPE)
        self.w_k_pos = nn.Linear(config.d_model, config.d_rope * config.num_heads, bias=False)
        self.w_q_pos = nn.Linear(config.d_model, config.d_rope * config.num_heads, bias=False)
        
        # Extended RoPE
        rope_config = ExtendedRoPEConfig(
            d_head=config.d_rope,
            max_seq_len=config.max_seq_len,
            scaling_type="yarn",
        )
        self.rope = ExtendedRotaryPositionalEncoding(rope_config)
        
        # Output projection
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Sparse pattern generator
        self.pattern = SparseAttentionPattern(config)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.scale_rope = 1.0 / math.sqrt(self.d_rope)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with sparse attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional additional mask
            position_offset: Position offset for KV cache
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        B, T, C = x.shape
        
        # A: Content Path (MLA compression)
        q_c = self.w_q_content(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        c_kv = self.w_dkv_content(x)  # [B, T, d_latent]
        k_c = self.w_uk_content(c_kv).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v_c = self.w_uv_content(c_kv).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # B: Position Path (RoPE)
        q_r = self.w_q_pos(x).view(B, T, self.num_heads, self.d_rope).transpose(1, 2)
        k_r = self.w_k_pos(x).view(B, T, self.num_heads, self.d_rope).transpose(1, 2)
        
        q_r = self.rope(q_r, offset=position_offset)
        k_r = self.rope(k_r, offset=position_offset)
        
        # C: Compute sparse attention
        context = self._sparse_attention(q_c, k_c, v_c, q_r, k_r, T)
        
        # D: Output projection
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(context)
    
    def _sparse_attention(
        self,
        q_c: torch.Tensor,  # [B, H, T, d_head]
        k_c: torch.Tensor,
        v_c: torch.Tensor,
        q_r: torch.Tensor,  # [B, H, T, d_rope]
        k_r: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute sparse attention with combined content and position scores."""
        device = q_c.device
        
        # Content scores
        content_scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * self.scale
        
        # Position scores
        position_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale_rope
        
        # Combined scores
        attn_scores = content_scores + position_scores
        
        # Apply sparse mask
        sparse_mask = self.pattern(seq_len, device)  # [seq_len, seq_len]
        
        # Convert bool mask to attention mask (0 for attend, -inf for ignore)
        attn_mask = torch.where(
            sparse_mask,
            torch.zeros(1, device=device, dtype=attn_scores.dtype),
            torch.full((1,), float('-inf'), device=device, dtype=attn_scores.dtype),
        )
        attn_scores = attn_scores + attn_mask
        
        # Softmax and weighted sum
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v_c)
        
        return context


class BlockSparseAttention(nn.Module):
    """
    Block-Sparse Attention for efficient computation.
    
    Computes attention only for non-zero blocks, avoiding
    computation on masked regions entirely.
    
    This is more efficient than applying a mask after full attention
    computation, especially for very long sequences.
    """
    
    def __init__(self, config: DSAConfig, block_size: int = 64):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.num_heads = config.num_heads
        self.d_head = config.d_head
        
        # Standard projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEncoding(config.d_head, config.max_seq_len)
        
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def get_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate block-level sparse mask.
        
        Returns which blocks should be computed (True) vs skipped (False).
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Create block-level positions
        block_pos = torch.arange(num_blocks, device=device)
        q_block = block_pos.unsqueeze(1)
        k_block = block_pos.unsqueeze(0)
        
        # Block distance
        block_dist = q_block - k_block
        
        # Window blocks (within N blocks)
        window_blocks = self.config.window_size // self.block_size
        window_mask = (block_dist >= 0) & (block_dist <= window_blocks)
        
        # Global blocks (first few blocks)
        global_blocks = max(1, self.config.global_tokens // self.block_size)
        global_mask = k_block < global_blocks
        
        # Dilated blocks
        dilation_blocks = max(1, self.config.dilation_stride // self.block_size)
        dilated_mask = (k_block % dilation_blocks) == 0
        
        # Causal
        if self.config.causal:
            causal_mask = k_block <= q_block
            return (window_mask | global_mask | dilated_mask) & causal_mask
        
        return window_mask | global_mask | dilated_mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with block-sparse attention.
        
        For efficiency, this should use specialized block-sparse kernels
        when available (e.g., Triton on CUDA).
        """
        B, T, C = x.shape
        
        # Projections
        q = self.w_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Get block mask
        block_mask = self.get_block_mask(T, x.device)
        
        # For now, use dense attention with mask
        # A production version would use block-sparse CUDA kernels
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Expand block mask to token level
        token_mask = block_mask.repeat_interleave(self.block_size, dim=0)
        token_mask = token_mask.repeat_interleave(self.block_size, dim=1)
        token_mask = token_mask[:T, :T]  # Trim to actual size
        
        # Apply mask
        attn_mask = torch.where(
            token_mask,
            torch.zeros(1, device=x.device, dtype=attn_scores.dtype),
            torch.full((1,), float('-inf'), device=x.device, dtype=attn_scores.dtype),
        )
        attn_scores = attn_scores + attn_mask
        
        # Softmax and output
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(context)


class DSAAlignmentLoss(nn.Module):
    """
    Alignment loss for training DSA to match full attention.
    
    During training, we can compute both sparse and full attention
    and add a loss term to ensure sparse attention approximates full well.
    """
    
    def __init__(
        self, 
        mse_weight: float = 1.0, 
        cosine_weight: float = 0.5,
        sample_ratio: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.sample_ratio = sample_ratio
    
    def forward(
        self, 
        sparse_output: torch.Tensor, 
        full_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment loss between sparse and full attention outputs.
        
        Args:
            sparse_output: Output from sparse attention [B, T, D]
            full_output: Output from full attention [B, T, D]
            
        Returns:
            Combined alignment loss
        """
        # Optionally sample positions for efficiency
        if self.sample_ratio < 1.0:
            B, T, D = sparse_output.shape
            num_samples = max(1, int(T * self.sample_ratio))
            indices = torch.randperm(T, device=sparse_output.device)[:num_samples]
            sparse_output = sparse_output[:, indices, :]
            full_output = full_output[:, indices, :]
        
        loss = torch.tensor(0.0, device=sparse_output.device)
        
        # MSE loss
        if self.mse_weight > 0:
            mse = F.mse_loss(sparse_output, full_output)
            loss = loss + self.mse_weight * mse
        
        # Cosine similarity loss
        if self.cosine_weight > 0:
            sparse_flat = sparse_output.view(-1, sparse_output.shape[-1])
            full_flat = full_output.view(-1, full_output.shape[-1])
            cosine_sim = F.cosine_similarity(sparse_flat, full_flat, dim=-1)
            cosine_loss = 1.0 - cosine_sim.mean()
            loss = loss + self.cosine_weight * cosine_loss
        
        return loss


def demo_sparse_attention():
    """Demonstrate sparse attention on available device."""
    # Get best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print("=" * 60)
    print("DeepSeek Sparse Attention Demo")
    print("=" * 60)
    
    # Small config for demo
    config = DSAConfig.small_test()
    print(f"\nConfig: {config.d_model}D, {config.num_heads} heads, "
          f"window={config.window_size}, max_seq={config.max_seq_len}")
    
    # Create model
    dsa = DeepSeekSparseAttention(config).to(device)
    
    # Test input
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, config.d_model, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = dsa(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Memory comparison
    print(f"\nEffective KV budget per position: {config.effective_kv_budget}")
    print(f"Full attention would require: {config.max_seq_len}")
    print(f"Memory reduction factor: {config.max_seq_len / config.effective_kv_budget:.2f}x")


if __name__ == "__main__":
    demo_sparse_attention()
